"""Exportación de resultados.

La interfaz :class:`ResultExporter` permite sustituir el CSV por SQLite,
PostgreSQL o PDF en el futuro sin tocar el pipeline de análisis.

Los exportadores se cargan de forma perezosa para que el módulo de dominio
``DetectionRecord`` no fuerce la importación de pandas.
"""

from .records import DetectionRecord

__all__ = ["CsvExporter", "DetectionRecord", "PandasCsvExporter", "ResultExporter"]


def __getattr__(name: str):
    """Resolución lazy de los exportadores (evita dependencias innecesarias)."""
    if name in {"CsvExporter", "PandasCsvExporter", "ResultExporter", "export_records"}:
        from . import csv_exporter

        return getattr(csv_exporter, name)
    raise AttributeError(f"módulo {__name__!r} no tiene el atributo {name!r}")
