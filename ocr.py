import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
import numpy as np
import cv2
import onnxruntime as ort

from rapidocr import RapidOCR
from rapidocr.inference_engine.onnxruntime.provider_config import EP, ProviderConfig


from tsv_export import compute_geometry, write_tsv


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_OCR_DIR = Path(__file__).resolve().parent / "output_ocr"
worker_context = local()


def _patch_dml_provider_name() -> None:
    # rapidocr pasa (nombre, {}) a DirectML y onnxruntime lo rechaza; requiere solo el nombre.
    original = ProviderConfig.get_ep_list

    def patched(self) -> list:
        return [
            EP.DIRECTML_EP.value if ep[0] == EP.DIRECTML_EP.value else ep
            for ep in original(self)
        ]

    ProviderConfig.get_ep_list = patched


def _select_provider() -> str:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "CUDA"
    if "DmlExecutionProvider" in available:
        return "DirectML"
    return "CPU"


def _build_ocr_engine() -> RapidOCR:
    # Carga CUDA/cuDNN desde paquetes pip nvidia-* o desde el PATH si existen.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    ocr_engine = RapidOCR()
    provider = _select_provider()
    onnx_cfg = ocr_engine.cfg.EngineConfig.onnxruntime
    if provider == "CUDA":
        onnx_cfg.use_cuda = True
    elif provider == "DirectML":
        _patch_dml_provider_name()
        onnx_cfg.use_dml = True
        onnx_cfg.dml_ep_cfg = {}

    # Los modelos se cargan en la primera inferencia; se fuerza para verificar el dispositivo real.
    ocr_engine(np.full((64, 256, 3), 255, dtype=np.uint8))
    active = ocr_engine.text_det.session.session.get_providers()[0]
    device = "CPU" if active == "CPUExecutionProvider" else f"GPU ({active})"
    print(f"\n===== [pid {os.getpid()}] USANDO: {device} =====")
    if provider != "CPU" and device == "CPU":
        print(f"===== Se pidió {provider} pero se cayó a CPU: faltan DLLs de CUDA/cuDNN (revisa el error de onnxruntime arriba) =====")
    return ocr_engine


def _get_worker_ocr_engine() -> RapidOCR:
    if not hasattr(worker_context, "ocr_engine"):
        worker_context.ocr_engine = _build_ocr_engine()
    return worker_context.ocr_engine


def process_image(image_path: Path, ocr_engine: RapidOCR, worker_id: int) -> float:
    """Procesa una imagen y guarda su imagen anotada y su texto extraído."""
    start_time = time.perf_counter()
    OUTPUT_OCR_DIR.mkdir(parents=True, exist_ok=True)
    boxes_image_path = OUTPUT_OCR_DIR / f"{image_path.stem}_boxes{image_path.suffix}"
    json_path = OUTPUT_OCR_DIR / f"{image_path.stem}.json"
    result = ocr_engine(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"No se pudo leer la imagen: {image_path}")

    boxes = result.boxes if result.boxes is not None else []
    texts = result.txts if result.txts is not None else []
    confidences = result.scores if result.scores is not None else []

    original = image.copy()
    crops_dir = OUTPUT_DIR / f"{image_path.stem}_ocr"
    crops_dir.mkdir(parents=True, exist_ok=True)
    height, width = original.shape[:2]
    detections = []
    json_detections = []

    for box, text, confidence in zip(boxes, texts, confidences):
        points = [[int(round(x)), int(round(y))] for x, y in box]
        contour = cv2.convexHull(np.array(points, dtype="int32"))
        cv2.polylines(image, [contour], True, (0, 255, 0), 2)

        label = f"{text} ({float(confidence):.2%})"
        x, y = points[0]
        cv2.putText(
            image,
            label,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        geometry = compute_geometry(points)
        x_min, y_min = geometry["left"], geometry["top"]
        x_max, y_max = x_min + geometry["width"], y_min + geometry["height"]
        cx_min, cy_min = max(x_min, 0), max(y_min, 0)
        cx_max, cy_max = min(x_max, width), min(y_max, height)
        if cx_max > cx_min and cy_max > cy_min:
            crop_name = (
                f"{geometry['left']}_{geometry['top']}"
                f"_{geometry['width']}_{geometry['height']}{image_path.suffix}"
            )
            cv2.imwrite(
                str(crops_dir / crop_name),
                original[cy_min:cy_max, cx_min:cx_max],
            )
        detections.append({"text": text, "confidence": confidence, "points": points})
        json_detections.append(
            {
                "texto": {"ocr_original": text, "trocr": "", "llm": ""},
                "confianza": round(float(confidence), 6),
                **geometry,
                "puntos": points,
            }
        )

    write_tsv(crops_dir / f"{image_path.stem}.tsv", detections)
    json_path.write_text(
        json.dumps(
            {"imagen": image_path.name, "detecciones": json_detections},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not cv2.imwrite(str(boxes_image_path), image):
        raise OSError(f"No se pudo guardar la imagen anotada: {boxes_image_path}")

    print(f"Worker {worker_id}: {image_path}")
    print(f"  Imagen con boxes: {boxes_image_path}")
    print(f"  JSON (OCR original y post-OCR): {json_path}")
    elapsed_time = time.perf_counter() - start_time
    print(f"  Tiempo: {elapsed_time:.2f} segundos")
    return elapsed_time


def process_image_task(arguments: tuple[Path, int]) -> tuple[int, Path, float]:
    """Crea un lector por worker y procesa una imagen."""
    image_path, worker_id = arguments
    elapsed_time = process_image(image_path, _get_worker_ocr_engine(), worker_id)
    return worker_id, image_path, elapsed_time


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("_boxes")
    )


def iter_process_images(folder_path: str | Path, max_workers: int = 2):
    """Procesa las imágenes en paralelo y cede (imagen, segundos) en cuanto cada una termina.

    Al ceder una imagen ya existen su carpeta de recortes, su TSV y su JSON.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {folder}")
    if max_workers < 1:
        raise ValueError("max_workers debe ser mayor que cero")

    image_paths = list_images(folder)
    if not image_paths:
        print(f"No se encontraron imágenes en: {folder}")
        return

    executor_class = ProcessPoolExecutor if max_workers > 1 else ThreadPoolExecutor
    with executor_class(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image_task, (path, index % max_workers + 1)): path
            for index, path in enumerate(image_paths)
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, _, elapsed = future.result()
            except Exception as error:
                print(f"[OCR][ERROR] {path.name}: {error}")
                continue
            yield path, elapsed


def process_folder(folder_path: str | Path, max_workers: int = 2) -> None:
    """Procesa todas las imágenes que se encuentran directamente en una carpeta."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {folder}")
    if max_workers < 1:
        raise ValueError("max_workers debe ser mayor que cero")

    image_paths = list_images(folder)
    if not image_paths:
        print(f"No se encontraron imágenes en: {folder}")
        return

    start_time = time.perf_counter()
    tasks = (
        (image_path, index % max_workers + 1)
        for index, image_path in enumerate(image_paths)
    )
    executor_class = ProcessPoolExecutor if max_workers > 1 else ThreadPoolExecutor
    with executor_class(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                process_image_task,
                tasks,
            )
        )

    total_time = time.perf_counter() - start_time
    image_times = [result[2] for result in results]
    average_time = sum(image_times) / len(image_times)
    print("\n--- Reporte final de tiempos ---")
    print(f"Carpeta: {folder}")
    print(f"Imágenes procesadas: {len(image_paths)}")
    print(f"Tiempo total: {total_time:.2f} segundos")
    print(f"Tiempo promedio por imagen: {average_time:.2f} segundos")
    for worker_id in range(1, max_workers + 1):
        worker_images = [
            str(image_path)
            for current_worker, image_path, _ in results
            if current_worker == worker_id
        ]
        print(f"Worker {worker_id}: {len(worker_images)} imágenes")
        # for image_path in worker_images:
        #     print(f"  - {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae texto de todas las imágenes de una carpeta."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=Path(__file__).resolve().parent / "muestras_venus",
        help="Ruta de la carpeta que contiene las imágenes.",
    )
    args = parser.parse_args()
    process_folder(args.folder)
