"""Flujo completo en paralelo (por etapas con colas), sin esperar a que termine cada etapa:

    detección+recortes (varios procesos) -> TrOCR (GPU) -> LLM (llama-server)

En cuanto una imagen termina una etapa pasa a la siguiente, mientras las demás
siguen en las etapas anteriores.

Uso:
    python pipeline.py [carpeta_de_imagenes]      # detecta, recorta, TrOCR y LLM
    python pipeline.py --skip-ocr                 # reutiliza output/ ya generado
    python pipeline.py --skip-trocr               # sin TrOCR (el LLM usa solo el OCR original)

Requiere llama-server activo (ver llm_post_ocr.py).
"""

import argparse
import csv
import json
import queue
import threading
import time
from pathlib import Path

from ocr import OUTPUT_DIR, OUTPUT_OCR_DIR, iter_process_images

DEFAULT_IMAGES = Path(__file__).resolve().parent / "muestras_venus"
GEOMETRY_KEYS = ("left", "top", "width", "height")
QUEUE_SIZE = 32  # límite por cola: evita acumular miles de imágenes en memoria
TROCR_THREADS = 3  # preproceso/E-S en paralelo; la inferencia en GPU se serializa en trocr_ocr
LLM_THREADS = 1  # refine_folder ya usa 2 hebras internas (= -np 2 de llama-server)
END = None


def _folder(stem: str) -> Path:
    return OUTPUT_DIR / f"{stem}_ocr"


def _tsv(stem: str) -> Path:
    return _folder(stem) / f"{stem}.tsv"


def _json(stem: str) -> Path:
    return OUTPUT_OCR_DIR / f"{stem}.json"


def sync_json(tsv_path: Path, json_path: Path, key: str) -> None:
    """Copia la columna text del TSV al JSON de la imagen, en texto.<key>."""
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    texts = {tuple(row[k] for k in GEOMETRY_KEYS): row["text"] for row in rows}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for detection in data["detecciones"]:
        geometry = tuple(str(detection[k]) for k in GEOMETRY_KEYS)
        if geometry in texts:
            detection["texto"][key] = texts[geometry]
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stage_seconds: dict[str, list[float]] = {}
        self.image_seconds: dict[str, float] = {}
        self.failed: dict[str, str] = {}
        self.completed: set[str] = set()

    def add(self, stage: str, stem: str, seconds: float) -> None:
        with self.lock:
            self.stage_seconds.setdefault(stage, []).append(seconds)
            self.image_seconds[stem] = self.image_seconds.get(stem, 0.0) + seconds


def _stage_worker(name, handler, inbox, outbox, stats: Stats, last: bool, pending: list) -> None:
    # `pending` es un contador compartido (lista de 1 elemento) con la cantidad de hebras activas de la etapa.
    while True:
        stem = inbox.get()
        if stem is END:
            break
        started = time.perf_counter()
        try:
            handler(stem)
        except Exception as error:
            print(f"[{name}][ERROR] {stem}: {error}", flush=True)
            with stats.lock:
                stats.failed[stem] = name
            continue
        stats.add(name, stem, time.perf_counter() - started)
        print(f"[{name}] {stem} listo", flush=True)
        if last:
            with stats.lock:
                stats.completed.add(stem)
        else:
            outbox.put(stem)
    with stats.lock:
        pending[0] -= 1
        remaining = pending[0]
    if remaining:
        inbox.put(END)  # despierta a las hebras hermanas
    elif outbox is not None:
        outbox.put(END)  # la última hebra cierra la etapa siguiente


def run(images: Path, skip_ocr: bool, skip_trocr: bool, workers: int = 2) -> None:
    import llm_post_ocr

    stats = Stats()
    stages = []
    if not skip_trocr:
        import trocr_ocr

        def trocr_handler(stem: str) -> None:
            trocr_ocr.process_folder(_folder(stem))
            sync_json(_tsv(stem), _json(stem), "trocr")

        stages.append(("TrOCR", trocr_handler, TROCR_THREADS))

    def llm_handler(stem: str) -> None:
        llm_post_ocr.refine_folder(_tsv(stem), _json(stem))

    stages.append(("LLM", llm_handler, LLM_THREADS))

    queues = [queue.Queue(maxsize=QUEUE_SIZE) for _ in stages]
    threads = []
    for index, (name, handler, count) in enumerate(stages):
        last = index == len(stages) - 1
        outbox = None if last else queues[index + 1]
        pending = [count]
        for n in range(count):
            threads.append(
                threading.Thread(
                    target=_stage_worker,
                    args=(name, handler, queues[index], outbox, stats, last, pending),
                    name=f"{name}-{n}",
                )
            )

    # Calentamiento: carga de modelos (TrOCR, LLM, motores OCR) fuera del tiempo medido.
    def warm_trocr() -> None:
        from PIL import Image

        _, processor, _ = trocr_ocr.load_model()
        blank = Image.new("RGB", (256, 64), "white")
        trocr_ocr.recognize(processor([blank], return_tensors="pt").pixel_values)

    def warm_llm() -> None:
        try:
            llm_post_ocr._chat("prueba")
        except Exception as error:
            print(f"[LLM][AVISO] calentamiento falló: {error}", flush=True)

    warmups = [threading.Thread(target=warm_llm, daemon=True)]
    if not skip_trocr:
        warmups.append(threading.Thread(target=warm_trocr, daemon=True))
    print("Cargando modelos (no cuenta en el tiempo medido)...", flush=True)
    for thread in warmups + threads:
        thread.start()

    timer = {}

    def on_ready() -> None:
        for warm in warmups:
            warm.join()
        timer["start"] = time.perf_counter()
        print("Modelos listos. Iniciando medición.", flush=True)

    # Etapa 1 (esta hebra): cada imagen entra a la cola en cuanto termina de recortarse.
    try:
        if skip_ocr:
            on_ready()
            stems = sorted(p.name.removesuffix("_ocr") for p in OUTPUT_DIR.glob("*_ocr") if p.is_dir())
            for stem in stems:
                queues[0].put(stem)
        else:
            for image_path, seconds in iter_process_images(images, workers, on_ready):
                stats.add("OCR+recortes", image_path.stem, seconds)
                queues[0].put(image_path.stem)
    finally:
        queues[0].put(END)
        for thread in threads:
            thread.join()

    total = time.perf_counter() - timer["start"] if "start" in timer else 0.0
    processed = len(stats.completed)
    print("\n========== INFORME FINAL ==========")
    print(f"Imágenes procesadas:        {processed}")
    if stats.failed:
        print(f"Imágenes con error:         {len(stats.failed)} ({', '.join(sorted(stats.failed))})")
    print(f"Tiempo total del proceso:   {total:.2f} s")
    if processed:
        per_image = [stats.image_seconds[s] for s in stats.completed]
        print(f"Tiempo promedio por imagen: {sum(per_image) / processed:.2f} s (suma de etapas)")
        print(f"Tiempo efectivo por imagen: {total / processed:.2f} s (total / imágenes, con paralelismo)")
    for name, values in stats.stage_seconds.items():
        print(f"  - {name}: {sum(values) / len(values):.2f} s promedio, {sum(values):.2f} s acumulado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flujo OCR + TrOCR + post-OCR LLM.")
    parser.add_argument("folder", nargs="?", default=DEFAULT_IMAGES, type=Path)
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-trocr", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.folder, args.skip_ocr, args.skip_trocr, args.workers)
