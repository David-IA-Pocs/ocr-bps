import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_generated_files(folder: Path) -> list[Path]:
    generated_files: list[Path] = []

    for path in folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() == ".txt":
            generated_files.append(path)
            continue

        if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.endswith("_boxes"):
            generated_files.append(path)

    return sorted(generated_files)


def clean_generated_files(folder_path: str | Path, dry_run: bool = False) -> int:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {folder}")

    generated_files = find_generated_files(folder)
    if not generated_files:
        print(f"No se encontraron archivos generados en: {folder}")
        return 0

    action = "[DRY-RUN] Se borrarían" if dry_run else "Borrado"
    print(f"{action} {len(generated_files)} archivos en: {folder}")

    removed_count = 0
    for path in generated_files:
        print(f"  - {path}")
        if dry_run:
            continue

        path.unlink()
        removed_count += 1

    if not dry_run:
        print(f"Archivos eliminados: {removed_count}")

    return len(generated_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpia los archivos generados por OCR en una carpeta."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="muestras_venus",
        help="Carpeta a limpiar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra qué archivos se borrarían sin eliminarlos.",
    )
    args = parser.parse_args()
    clean_generated_files(args.folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()