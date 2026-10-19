import argparse
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Carpetas cuyo contenido se elimina. Agrega aquí más rutas para limpiarlas también.
CLEANUP_PATHS: list[Path] = [
    BASE_DIR / "output",
    BASE_DIR / "output_ocr",
]


def clean_folder(folder_path: str | Path, dry_run: bool = False) -> int:
    """Elimina el contenido de una carpeta (la carpeta se conserva)."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        print(f"No existe, se omite: {folder}")
        return 0
    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {folder}")

    entries = sorted(folder.iterdir())
    if not entries:
        print(f"Nada que limpiar en: {folder}")
        return 0

    action = "[DRY-RUN] Se borrarían" if dry_run else "Borrando"
    print(f"{action} {len(entries)} elementos en: {folder}")
    for entry in entries:
        print(f"  - {entry}")
        if dry_run:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    return len(entries)


def clean_paths(paths: list[Path] = CLEANUP_PATHS, dry_run: bool = False) -> int:
    return sum(clean_folder(path, dry_run=dry_run) for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpia el contenido de las carpetas generadas por el OCR."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra qué se borraría sin eliminarlo.",
    )
    args = parser.parse_args()
    total = clean_paths(dry_run=args.dry_run)
    print(f"Total de elementos {'a borrar' if args.dry_run else 'eliminados'}: {total}")


if __name__ == "__main__":
    main()
