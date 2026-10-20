import csv
from pathlib import Path

TsvRow = dict[str, str]


def leer_tsv(ruta: str | Path, encoding: str = "utf-8") -> list[TsvRow]:
    """Lee un archivo TSV y devuelve una lista de filas como diccionarios."""
    with Path(ruta).open("r", encoding=encoding, newline="") as archivo:
        return [dict(fila) for fila in csv.DictReader(archivo, delimiter="\t")]


def guardar_tsv(ruta: str | Path, filas: list[TsvRow], fieldnames: list[str]) -> None:
    """Guarda las filas en formato TSV con las columnas indicadas."""
    with Path(ruta).open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=fieldnames, delimiter="\t")
        escritor.writeheader()
        escritor.writerows(filas)
