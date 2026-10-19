"""Exporta las detecciones OCR de una imagen a un archivo TSV.

Para cambiar la estructura del TSV basta con editar COLUMNS (nombre de columna
y función que obtiene su valor a partir de una detección).
"""

import csv
from pathlib import Path
from typing import Callable

# Una detección es un dict: {"text": str, "confidence": float, "points": [[x, y], ...]}
Detection = dict


def compute_geometry(points: list[list[int]]) -> dict[str, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, top = min(xs), min(ys)
    return {"left": left, "top": top, "width": max(xs) - left, "height": max(ys) - top}


def geometry_column(key: str) -> Callable[[Detection], str]:
    return lambda d: str(compute_geometry(d["points"])[key])


COLUMNS: list[tuple[str, Callable[[Detection], str]]] = [
    ("text", lambda d: ""),
    ("conf", lambda d: ""),
    ("left", geometry_column("left")),
    ("top", geometry_column("top")),
    ("width", geometry_column("width")),
    ("height", geometry_column("height")),
]


def write_tsv(tsv_path: Path, detections: list[Detection]) -> None:
    with Path(tsv_path).open("w", encoding="utf-8", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
        writer.writerow([name for name, _ in COLUMNS])
        for detection in detections:
            writer.writerow([getter(detection) for _, getter in COLUMNS])
