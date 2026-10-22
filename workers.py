import argparse
from pathlib import Path

from pipeline import run

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta el flujo completo: OCR + recortes -> TrOCR -> post-OCR con LLM."
    )
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de workers a usar.")
    parser.add_argument("--skip-ocr", action="store_true", help="Reutiliza output/ ya generado.")
    parser.add_argument("--skip-trocr", action="store_true", help="No ejecuta TrOCR.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="muestras_venus",
        help="Carpeta con imágenes a procesar.",
    )
    args = parser.parse_args()
    run(Path(args.folder), args.skip_ocr, args.skip_trocr, args.workers)


if __name__ == "__main__":
    main()