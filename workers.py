import argparse

from ocr import process_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesa una carpeta de imágenes con OCR.")
    parser.add_argument("--workers", type=int, default=6, help="Cantidad de workers a usar.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="muestras_venus",
        help="Carpeta con imágenes a procesar.",
    )
    args = parser.parse_args()
    process_folder(args.folder, max_workers=args.workers)


if __name__ == "__main__":
    main()