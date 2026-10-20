"""Post-OCR con LLM (Qwen en llama.cpp): corrige errores de OCR sin cambiar el sentido.

Requiere un llama-server en marcha, por ejemplo:
    llama-server -m Qwen.gguf --port 8080 -c 4096 -ngl 99

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
    "3. Restaura tildes ('dolares' -> 'dólares') y conserva las mayúsculas del "
    "original: si empieza en minúscula, la respuesta también.\n"
    "4. En tokens numéricos corrige caracteres confundidos: 'i10' -> '110', "
    "'5O0' -> '500'.\n"
    "5. NUNCA cambies el valor ni el sentido: no sustituyas un número por otro "
    "distinto, no agregues ni quites palabras o cifras, no traduzcas, no inventes "
    "nombres ni fechas. 'cinco' no es 'cincuenta'.\n"
    "6. Si el texto ya es correcto o dudas, devuélvelo idéntico.\n"
    "Responde SOLO con el texto corregido en una línea, sin explicaciones, "
    "comillas ni formato."
)

FEW_SHOT = [
    ("cincuentaycuatro", "cincuenta y cuatro"),
    ("Drez", "Diez"),
    ("einco mil dolares exactos", "cinco mil dólares exactos"),
    ("ciento veintitresmil", "ciento veintitrés mil"),
    ("i10 pesos", "110 pesos"),
    ("veintiunodolares", "veintiún dólares"),
    ("einco mil", "cinco mil"),
    ("Banco Pichincha", "Banco Pichincha"),
]


def _chat(text: str) -> str:
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
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
    print(f"[LLM] entrada={text!r} -> respuesta={content!r}", flush=True)
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


def clean_tsv(tsv_path: Path, column: str = "text", workers: int = 2) -> int:
    """Corrige con el LLM la columna de texto del TSV, sobrescribiendo el mismo archivo.

    Devuelve cuántas filas cambiaron. Usar workers <= slots (-np) de llama-server.
    """
    tsv_path = Path(tsv_path)
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or column not in rows[0]:
        raise ValueError(f"{tsv_path} no tiene la columna '{column}'")
    index = rows[0].index(column)
    body = [row for row in rows[1:] if len(row) > index]

    unique_texts = {row[index] for row in body if row[index].strip()}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        corrections = dict(zip(unique_texts, executor.map(correct_text, unique_texts)))

    changed = 0
    for row in body:
        new = corrections.get(row[index], row[index])
        changed += new != row[index]
        row[index] = new

    temp_path = tsv_path.with_suffix(".tsv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
    temp_path.replace(tsv_path)
    return changed


def clean_all(output_dir: Path) -> None:
    tsv_files = sorted(Path(output_dir).rglob("*.tsv"))
    for tsv_file in tsv_files:
        try:
            changed = clean_tsv(tsv_file)
            print(f"[LLM] {tsv_file.parent.name}/{tsv_file.name}: {changed} filas corregidas")
        except (OSError, ValueError) as error:
            print(f"[LLM][ERROR] {tsv_file}: {error}")
    if not tsv_files:
        print(f"[LLM] Sin TSV en: {output_dir}")


if __name__ == "__main__":
    for line in sys.argv[1:] or sys.stdin.read().splitlines():
        print(correct_text(line))
