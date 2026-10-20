import argparse
from pathlib import Path

from pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta el flujo completo: OCR + recortes -> API -> post-OCR con LLM."
    )
    parser.add_argument("--workers", type=int, default=6, help="Cantidad de workers a usar.")
    parser.add_argument("--skip-ocr", action="store_true", help="Reutiliza output/ ya generado.")
    parser.add_argument("--skip-api", action="store_true", help="No llama a la API.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="muestras_venus",
        help="Carpeta con imágenes a procesar.",
    )
    args = parser.parse_args()
    run(Path(args.folder), args.skip_ocr, args.skip_api, args.workers)


if __name__ == "__main__":
    main()