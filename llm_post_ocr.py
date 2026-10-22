"""Post-OCR con LLM (Qwen en llama.cpp): corrige errores de OCR sin cambiar el sentido.

Requiere un llama-server en marcha, por ejemplo:
    llama-server -m Qwen.gguf --port 8080 -c 4096 -ngl 99
    llama-server.exe -m Qwen2.5-3B/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 99 --port 9931 -c 4096 -np 2

Variables de entorno opcionales:
    LLAMA_URL    (default http://127.0.0.1:9931)
    LLAMA_MODEL  (default "qwen"; llama-server lo ignora si hay un solo modelo)
    LLAMA_TIMEOUT (segundos, default 60)

Uso como módulo:  from llm_post_ocr import correct_text
Uso CLI:          python llm_post_ocr.py "einco mil dolares exactos"
"""

import csv
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:9931").rstrip("/")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "qwen")
LLAMA_TIMEOUT = float(os.environ.get("LLAMA_TIMEOUT", "60"))

# Si el texto corregido se parece menos que esto al original, se descarta.
MIN_SIMILARITY = 0.5
GEOMETRY_KEYS = ("left", "top", "width", "height")

SYSTEM_PROMPT = (
    "Eres un corrector de texto obtenido por OCR en español (cheques y documentos "
    "bancarios). Tu única tarea es reparar errores de reconocimiento y devolver "
    "el mismo texto ya corregido.\n"
    "Reglas:\n"
    "1. Separa palabras pegadas: 'cincuentaycuatro' -> 'cincuenta y cuatro', "
    "'milquinientos' -> 'mil quinientos', 'dolaresexactos' -> 'dólares exactos'. "
    "Las cifras en letras se escriben separadas, con 'y' entre decenas y unidades "
    "(treinta y uno ... noventa y nueve), excepto del 16 al 29 que van juntas "
    "(dieciséis, veintiuno).\n"
    "2. Repara palabras mal leídas por letras confundidas, eligiendo la palabra "
    "más parecida que tenga sentido en el contexto: 'Drez' -> 'Diez', "
    "'einco' -> 'cinco', 'tresc1entos' -> 'trescientos', 'seis' nunca debe pasar "
    "a 'siete'.\n"
    "3. En tokens numéricos corrige caracteres confundidos: 'i10' -> '110', "
    "'5O0' -> '500'.\n"
    "4. NUNCA cambies el valor ni el sentido: no sustituyas un número por otro "
    "distinto, no agregues ni quites palabras o cifras, no traduzcas, no inventes "
    "nombres ni fechas. 'cinco' no es 'cincuenta'.\n"
    "5. Si el texto ya es correcto o dudas, devuélvelo idéntico.\n"
    "Responde SOLO con el texto corregido en una línea, sin explicaciones, "
    "comillas ni formato."
)

FEW_SHOT = [
    ("cincuentaycuatro", "cincuenta y cuatro"),
    ("Drez", "Diez"),
    ("einco mil dolares exactos", "cinco mil dolares exactos"),
    ("ciento veintitresmil", "ciento veintitrés mil"),
    ("i10 pesos", "110 pesos"),
    ("veintiunodolares", "veintiun dolares"),
    ("einco mil", "cinco mil"),
    ("Banco Pichincha", "Banco Pichincha"),
    ("dolav", "dolar"),
]


SELECT_PROMPT = (
    "Recibes dos lecturas OCR del MISMO fragmento de un cheque o documento bancario "
    "en español: 'A' (OCR local) y 'B' (TrOCR). Cada una puede tener errores.\n"
    "Tu tarea: decidir cuál tiene más sentido semántico (o combinarlas si se "
    "complementan) y devolver el texto final corregido.\n"
    "Reglas:\n"
    "1. Separa palabras pegadas: 'cincuentaycuatro' -> 'cincuenta y cuatro'. Las "
    "cifras en letras van separadas, con 'y' entre decenas y unidades (31-99); "
    "del 16 al 29 van juntas (dieciséis, veintiuno).\n"
    "2. Repara letras confundidas eligiendo la palabra más parecida con sentido: "
    "'Drez' -> 'Diez', 'einco' -> 'cinco'. En números corrige 'i10' -> '110'.\n"
    "3. Si las lecturas difieren, prefiere la que forme una palabra o cifra válida "
    "y coherente; no inventes contenido que no esté en ninguna.\n"
    "4. NUNCA cambies el valor ni el sentido: no sustituyas un número por otro "
    "distinto ('cinco' no es 'cincuenta'), no agregues ni quites palabras o cifras.\n"
    "Responde SOLO con el texto final en una línea, sin explicaciones, comillas "
    "ni etiquetas."
)

SELECT_FEW_SHOT = [
    (("cincuentaycuatro", "cincuenta y cuatro"), "cincuenta y cuatro"),
    (("Drez", "Diez"), "Diez"),
    (("einco mil", "cinco mil"), "cinco mil"),
    (("cinco mil dolares", "cinco mi1 d0lares"), "cinco mil dolares"),
    (("i10", "110"), "110"),
    (("veintiunodolares", "veintiun d0lares"), "veintiun dolares"),
    (("Banco Pichincha", ""), "Banco Pichincha"),
    (("Banco Pichincha", "Banco Pichincha"), "Banco Pichincha"),
    (("LASUMADE", "LA5UMADE"), "LA SUMA DE"),
    (("dolav", "dolar"), "dolar"),
    (("dolaves", "d0lares"), "dolares"),
    (("Nn dolav", "un dolav"), "Un dolar"),
    (("LASUMADE", "LA5UMADE"), "LA SUMA DE")
]


def _format_pair(original: str, trocr: str) -> str:
    return f"A: {original}\nB: {trocr}"


def _chat(text: str, messages: list[dict] | None = None) -> str:
    payload = {
        "model": LLAMA_MODEL,
        "messages": messages
        or [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[
                {"role": role, "content": content}
                for source, target in FEW_SHOT
                for role, content in (("user", source), ("assistant", target))
            ],
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": max(64, len(text) * 2),
        # Desactiva el modo "thinking" de Qwen3 (ignorado por otros modelos).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=LLAMA_TIMEOUT) as response:
        data = json.load(response)
    content = data["choices"][0]["message"]["content"]
    # print(f"[LLM] entrada={text!r} -> respuesta={content!r}", flush=True)
    return content


def correct_text(text: str) -> str:
    """Devuelve el texto corregido por el LLM; ante cualquier fallo, el original."""
    if not text or not text.strip():
        return text
    try:
        corrected = _chat(text).strip()
    except Exception as error:
        print(f"[llm_post_ocr] fallo, se conserva el original: {error}", file=sys.stderr)
        return text
    if "</think>" in corrected:
        corrected = corrected.split("</think>")[-1].strip()
    corrected = corrected.strip('"\'`')
    if not corrected or "\n" in corrected and "\n" not in text:
        return text
    if SequenceMatcher(None, text.lower(), corrected.lower()).ratio() < MIN_SIMILARITY:
        return text
    return corrected


def resolve_text(original: str, trocr: str) -> str:
    """El LLM elige entre la lectura OCR original y la de TrOCR, y la corrige.

    Ante cualquier fallo o respuesta dudosa se usa TrOCR (o el original si no hay).
    """
    candidates = [t for t in (trocr, original) if t and t.strip()]
    if not candidates:
        return trocr or original
    fallback = candidates[0]
    pair = _format_pair(original, trocr)
    messages = [
        {"role": "system", "content": SELECT_PROMPT},
        *[
            {"role": role, "content": content}
            for (a, b), target in SELECT_FEW_SHOT
            for role, content in (("user", _format_pair(a, b)), ("assistant", target))
        ],
        {"role": "user", "content": pair},
    ]
    try:
        answer = _chat(pair, messages).strip()
    except Exception as error:
        print(f"[llm_post_ocr] fallo, se usa '{fallback}': {error}", file=sys.stderr)
        return fallback
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()
    answer = answer.strip('"\'`')
    for label in ("A:", "B:"):
        if answer.startswith(label):
            answer = answer[len(label):].strip()
    if not answer or "\n" in answer:
        return fallback
    best = max(
        SequenceMatcher(None, c.lower(), answer.lower()).ratio() for c in candidates
    )
    return answer if best >= MIN_SIMILARITY else fallback


def refine_folder(tsv_path: Path, json_path: Path, workers: int = 2) -> int:
    """Resuelve cada detección con el LLM y guarda el resultado en el JSON (texto.llm)
    y en la columna text del TSV (mismo archivo). Devuelve cuántas filas del TSV cambiaron.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    detections = data["detecciones"]
    pairs = {
        (d["texto"].get("ocr_original", ""), d["texto"].get("trocr", ""))
        for d in detections
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        resolved = dict(zip(pairs, executor.map(lambda p: resolve_text(*p), pairs)))

    by_geometry = {}
    for d in detections:
        pair = (d["texto"].get("ocr_original", ""), d["texto"].get("trocr", ""))
        d["texto"]["llm"] = resolved[pair]
        by_geometry[tuple(str(d[k]) for k in GEOMETRY_KEYS)] = resolved[pair]
    Path(json_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with Path(tsv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0]
    text_index = header.index("text")
    geometry_index = [header.index(k) for k in GEOMETRY_KEYS]
    changed = 0
    for row in rows[1:]:
        new = by_geometry.get(tuple(row[i] for i in geometry_index))
        if new is not None:
            changed += new != row[text_index]
            row[text_index] = new
    temp_path = Path(tsv_path).with_suffix(".tsv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
    temp_path.replace(tsv_path)
    return changed


def refine_all(output_dir: Path, json_dir: Path) -> int:
    """Aplica refine_folder a cada TSV de output_dir. Devuelve cuántos se procesaron."""
    done = 0
    tsv_files = sorted(Path(output_dir).rglob("*.tsv"))
    for tsv_file in tsv_files:
        json_file = Path(json_dir) / f"{tsv_file.parent.name.removesuffix('_ocr')}.json"
        try:
            changed = refine_folder(tsv_file, json_file)
            done += 1
            print(f"[LLM] {tsv_file.parent.name}: {changed} filas cambiadas")
        except (OSError, ValueError, KeyError) as error:
            print(f"[LLM][ERROR] {tsv_file}: {error}")
    if not tsv_files:
        print(f"[LLM] Sin TSV en: {output_dir}")
    return done


if __name__ == "__main__":
    for line in sys.argv[1:] or sys.stdin.read().splitlines():
        print(correct_text(line))
