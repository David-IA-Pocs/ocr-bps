"""Flujo completo: recortes + TSV -> API OCR (rellena el texto) -> post-OCR con LLM.

Uso:
    python pipeline.py [carpeta_de_imagenes]      # detecta, recorta, API y LLM
    python pipeline.py --skip-ocr                 # reutiliza output/ ya generado
    python pipeline.py --skip-api                 # solo limpia con el LLM

Requiere llama-server activo (ver llm_post_ocr.py).
"""

import argparse
import csv
import json
from pathlib import Path

import llm_post_ocr
import send_to_api
from ocr import OUTPUT_DIR, OUTPUT_OCR_DIR, process_folder

DEFAULT_IMAGES = Path(__file__).resolve().parent / "muestras_venus"
GEOMETRY_KEYS = ("left", "top", "width", "height")


def sync_json(output_dir: Path, key: str) -> None:
    """Copia la columna text de cada TSV al JSON de la imagen, en texto.<key>."""
    for tsv_path in sorted(Path(output_dir).rglob("*.tsv")):
        json_path = OUTPUT_OCR_DIR / f"{tsv_path.parent.name.removesuffix('_ocr')}.json"
        if not json_path.is_file():
            continue
        with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        texts = {tuple(row[k] for k in GEOMETRY_KEYS): row["text"] for row in rows}
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for detection in data["detecciones"]:
            geometry = tuple(str(detection[k]) for k in GEOMETRY_KEYS)
            if geometry in texts:
                detection["texto"][key] = texts[geometry]
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(images: Path, skip_ocr: bool, skip_api: bool, workers: int = 2) -> None:
    if not skip_ocr:
        print("== 1/3 Detección y recortes ==")
        process_folder(images, max_workers=workers)
    if not skip_api:
        print("== 2/3 API OCR ==")
        send_to_api.process_all(OUTPUT_DIR)
        sync_json(OUTPUT_DIR, "api")
    print("== 3/3 Post-OCR con LLM sobre los TSV ==")
    llm_post_ocr.clean_all(OUTPUT_DIR)
    sync_json(OUTPUT_DIR, "llm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flujo OCR + API + post-OCR LLM.")
    parser.add_argument("folder", nargs="?", default=DEFAULT_IMAGES, type=Path)
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.folder, args.skip_ocr, args.skip_api, args.workers)
