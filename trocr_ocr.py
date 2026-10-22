"""Segundo OCR local con TrOCR: lee los recortes de cada carpeta y rellena el TSV.

Para cada carpeta output/<imagen>_ocr/ lee los recortes (<left>_<top>_<width>_<height>.<ext>), ejecuta TrOCR y escribe
text y conf en la fila del TSV con ese bbox, en el mismo archivo.
"""

import re
import threading
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from tsv_utils import guardar_tsv, leer_tsv

MODEL_NAME = "microsoft/trocr-base-printed"
BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_COLUMNS = {"text", "conf", "left", "top", "width", "height"}

_state: dict = {}
_load_lock = threading.Lock()
_gpu_lock = threading.Lock()


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    try:
        import torch_directml

        if torch_directml.is_available():
            return torch_directml.device(), f"DirectML ({torch_directml.device_name(0)})"
    except ImportError:
        pass
    return torch.device("cpu"), "CPU"


def load_model():
    with _load_lock:
        if not _state:
            device, name = _select_device()
            print(f"[TrOCR] Cargando {MODEL_NAME} en {name}")
            processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=True)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).eval()
            # fp16 en CUDA: la T4 tiene tensor cores y la inferencia rinde ~2x.
            if device.type == "cuda":
                model = model.half()
            model = model.to(device)
            _state.update(device=device, processor=processor, model=model)
    return _state["device"], _state["processor"], _state["model"]


def _bbox_from_name(path: Path) -> tuple[str, str, str, str] | None:
    numbers = re.findall(r"\d+", path.stem)
    return tuple(numbers[-4:]) if len(numbers) >= 4 else None


def _bbox_key(left: str, top: str, width: str, height: str) -> tuple[str, str, str, str]:
    def norm(value: str) -> str:
        try:
            return str(int(float((value or "").strip())))
        except ValueError:
            return (value or "").strip()

    return norm(left), norm(top), norm(width), norm(height)


def recognize(pixel_values: torch.Tensor) -> list[tuple[str, float | None]]:
    """Devuelve (texto, confianza) por recorte ya preprocesado (CPU), con la misma métrica que la API."""
    device, processor, model = load_model()
    pixel_values = pixel_values.to(device, dtype=model.dtype, non_blocking=True)
    # La GPU es una sola: la inferencia se serializa, el resto (E/S, preproceso) corre en paralelo.
    with _gpu_lock, torch.inference_mode():
        output = model.generate(
            pixel_values, return_dict_in_generate=True, output_scores=True
        )
    texts = processor.batch_decode(output.sequences, skip_special_tokens=True)
    # Log-prob de cada token generado (sin el token inicial), igual que la API.
    generated = output.sequences[:, 1:]
    log_probs = torch.stack(
        [torch.log_softmax(step.float(), dim=-1) for step in output.scores], dim=1
    )
    token_scores = log_probs.gather(2, generated.unsqueeze(-1)).squeeze(-1)
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = model.config.decoder.pad_token_id
    results = []
    for text, tokens, scores in zip(texts, generated, token_scores):
        valid = scores[tokens != pad_id] if pad_id is not None else scores
        confidence = float(torch.exp(valid.mean()).item()) if valid.numel() else None
        results.append((text, confidence))
    return results


def process_folder(folder: Path) -> int:
    """Rellena text/conf del TSV de una carpeta. Devuelve cuántas filas se actualizaron."""
    tsv_files = sorted(folder.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"Sin TSV en: {folder}")
    tsv_file = tsv_files[0]
    rows = leer_tsv(tsv_file, encoding="utf-8-sig")
    fieldnames = list(rows[0].keys()) if rows else []
    if rows and not REQUIRED_COLUMNS.issubset(fieldnames):
        raise ValueError(f"{tsv_file.name} debe tener: {sorted(REQUIRED_COLUMNS)}")

    index: dict[tuple, list[dict]] = {}
    for row in rows:
        index.setdefault(
            _bbox_key(row["left"], row["top"], row["width"], row["height"]), []
        ).append(row)

    _, processor, _ = load_model()
    keys = []
    images = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        bbox = _bbox_from_name(path)
        if bbox is None or _bbox_key(*bbox) not in index:
            continue
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
            keys.append(_bbox_key(*bbox))
        except (UnidentifiedImageError, OSError) as error:
            print(f"[TrOCR][ERROR] {path.name}: {error}")

    updated = 0
    for start in range(0, len(images), BATCH_SIZE):
        # Preproceso en CPU fuera del candado de GPU: se solapa con la inferencia de otras carpetas.
        pixel_values = processor(images[start : start + BATCH_SIZE], return_tensors="pt").pixel_values
        batch_keys = keys[start : start + BATCH_SIZE]
        for key, (text, confidence) in zip(batch_keys, recognize(pixel_values)):
            for row in index[key]:
                row["text"] = text
                row["conf"] = f"{confidence:.6f}" if confidence is not None else ""
                updated += 1

    guardar_tsv(tsv_file, rows, fieldnames=fieldnames)
    return updated


def process_all(output_dir: Path) -> int:
    """Procesa cada subcarpeta de output_dir. Devuelve cuántas se procesaron bien."""
    folders = sorted(p for p in Path(output_dir).iterdir() if p.is_dir()) if Path(output_dir).is_dir() else []
    if not folders:
        print(f"No hay carpetas en: {output_dir}")
        return 0
    ok = 0
    for folder in folders:
        try:
            updated = process_folder(folder)
            ok += 1
            print(f"[TrOCR] {folder.name}: {updated} filas actualizadas")
        except (OSError, ValueError, FileNotFoundError) as error:
            print(f"[TrOCR][ERROR] {folder.name}: {error}")
    print(f"Carpetas procesadas con TrOCR: {ok}/{len(folders)}")
    return ok


if __name__ == "__main__":
    import argparse

    from ocr import OUTPUT_DIR

    parser = argparse.ArgumentParser(description="TrOCR local sobre los recortes de output/.")
    parser.add_argument("folder", nargs="?", default=OUTPUT_DIR, type=Path)
    process_all(parser.parse_args().folder)
