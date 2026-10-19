import argparse
from pathlib import Path

import requests

API_URL = "https://lkn8mbsb-8000.use.devtunnels.ms"
ENDPOINT = "/predict-batch"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Evita la página de advertencia de los dev tunnels de Microsoft.
HEADERS = {"X-Tunnel-Skip-AntiPhishing-Page": "true"}
MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def build_form_data() -> dict[str, str]:
    """Campos de formulario que acompañan a los archivos. Editar aquí si la API cambia."""
    return {"tsv_path": "", "actualizar_tsv": "true", "devolver_tsv": "true"}


def send_folder(folder: Path, timeout: int = 600) -> requests.Response:
    """Envía en una sola petición todos los archivos (recortes y TSV) de una carpeta."""
    paths = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and (p.suffix.lower() in IMAGE_EXTENSIONS or p.suffix.lower() == ".tsv")
    )
    if not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in paths):
        raise FileNotFoundError(f"Sin imágenes en: {folder}")

    handles = [p.open("rb") for p in paths]
    try:
        files = [
            ("files", (p.name, h, MIME_TYPES.get(p.suffix.lower(), "application/octet-stream")))
            for p, h in zip(paths, handles)
        ]
        return requests.post(
            API_URL + ENDPOINT,
            headers=HEADERS,
            files=files,
            data=build_form_data(),
            timeout=timeout,
        )
    finally:
        for handle in handles:
            handle.close()


def process_all(output_dir: Path = OUTPUT_DIR) -> None:
    folders = sorted(p for p in output_dir.iterdir() if p.is_dir()) if output_dir.is_dir() else []
    if not folders:
        print(f"No hay carpetas en: {output_dir}")
        return

    ok = 0
    for folder in folders:
        try:
            response = send_folder(folder)
            if not response.ok:
                raise requests.RequestException(f"{response.status_code}: {response.text[:300]}")
        except (requests.RequestException, FileNotFoundError) as error:
            print(f"[ERROR] {folder.name}: {error}")
            continue

        tsv_files = sorted(folder.glob("*.tsv"))
        target = tsv_files[0] if tsv_files else folder / f"{folder.name}.tsv"
        target.write_bytes(response.content)
        ok += 1
        print(f"[OK] {folder.name} -> {response.status_code} ({target.name} actualizado)")

    print(f"Carpetas enviadas correctamente: {ok}/{len(folders)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Envía cada carpeta de output a la API OCR.")
    parser.add_argument("folder", nargs="?", default=OUTPUT_DIR, type=Path)
    process_all(parser.parse_args().folder)
