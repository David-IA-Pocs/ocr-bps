import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from threading import local
import numpy as np
import cv2

from rapidocr import RapidOCR
from rapidocr.inference_engine.onnxruntime.provider_config import EP, ProviderConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
worker_context = local()


def _patch_rapidocr_for_directml() -> None:
    if getattr(ProviderConfig, "_bps_directml_patched", False):
        return

    def patched_get_ep_list(self) -> list:
        results = [(EP.CPU_EP.value, self.cpu_ep_cfg())]

        if self.is_cuda_available():
            results.insert(0, (EP.CUDA_EP.value, self.cuda_ep_cfg()))

        if self.is_dml_available():
            results.insert(0, EP.DIRECTML_EP.value)

        if self.is_cann_available():
            results.insert(0, (EP.CANN_EP.value, self.cann_ep_cfg()))

        if self.is_coreml_available():
            results.insert(0, (EP.COREML_EP.value, self.coreml_ep_cfg()))

        return results

    ProviderConfig.get_ep_list = patched_get_ep_list
    ProviderConfig._bps_directml_patched = True


def _build_ocr_engine() -> RapidOCR:
    _patch_rapidocr_for_directml()

    ocr_engine = RapidOCR()
    ocr_engine.cfg.EngineConfig.onnxruntime.use_dml = True
    ocr_engine.cfg.EngineConfig.onnxruntime.dml_ep_cfg = {}
    return ocr_engine


def _get_worker_ocr_engine() -> RapidOCR:
    if not hasattr(worker_context, "ocr_engine"):
        worker_context.ocr_engine = _build_ocr_engine()
    return worker_context.ocr_engine


def process_image(image_path: Path, ocr_engine: RapidOCR, worker_id: int) -> float:
    """Procesa una imagen y guarda su imagen anotada y su texto extraído."""
    start_time = time.perf_counter()
    boxes_image_path = image_path.with_name(
        f"{image_path.stem}_boxes{image_path.suffix}"
    )
    text_path = image_path.with_suffix(".txt")
    result = ocr_engine(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"No se pudo leer la imagen: {image_path}")

    boxes = result.boxes if result.boxes is not None else []
    texts = result.txts if result.txts is not None else []
    confidences = result.scores if result.scores is not None else []

    with text_path.open("w", encoding="utf-8") as text_file:
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
            text_file.write(f"{text}\tconfianza: {float(confidence):.6f}\n")

    if not cv2.imwrite(str(boxes_image_path), image):
        raise OSError(f"No se pudo guardar la imagen anotada: {boxes_image_path}")

    print(f"Worker {worker_id}: {image_path}")
    print(f"  Imagen con boxes: {boxes_image_path}")
    print(f"  Texto y confianza: {text_path}")
    elapsed_time = time.perf_counter() - start_time
    print(f"  Tiempo: {elapsed_time:.2f} segundos")
    return elapsed_time


def process_image_task(arguments: tuple[Path, int]) -> tuple[int, Path, float]:
    """Crea un lector por worker y procesa una imagen."""
    image_path, worker_id = arguments
    elapsed_time = process_image(image_path, _get_worker_ocr_engine(), worker_id)
    return worker_id, image_path, elapsed_time


def process_folder(folder_path: str | Path, max_workers: int = 2) -> None:
    """Procesa todas las imágenes que se encuentran directamente en una carpeta."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {folder}")
    if max_workers < 1:
        raise ValueError("max_workers debe ser mayor que cero")

    image_paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("_boxes")
    )
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
