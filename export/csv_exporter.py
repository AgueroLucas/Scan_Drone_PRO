"""Exportación CSV compatible con Microsoft Excel.

Se usa ``pandas`` (requisito del proyecto) y se escriben los ficheros de forma
atómica: primero un ``*.tmp.csv`` y después ``os.replace``. Así nunca se lee un
CSV a medias aunque la aplicación se cierre durante una exportación.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .records import DetectionRecord

_LOGGER = logging.getLogger("dpc.export")


class ResultExporter(ABC):
    """Contrato de salida de resultados (punto de extensión para BD/PDF)."""

    @abstractmethod
    def export(self, records: Iterable[DetectionRecord], destination: Path | str) -> Path:
        """Escribe los registros y devuelve la ruta del fichero generado."""
        raise NotImplementedError


class PandasCsvExporter(ResultExporter):
    """Escribe ``ID;Fecha;Hora;Timestamp;Confianza`` con BOM UTF-8."""

    COLUMNS: tuple[str, ...] = ("ID", "Fecha", "Hora", "Timestamp", "Confianza")

    def export(self, records: Iterable[DetectionRecord], destination: Path | str) -> Path:
        """Genera el CSV. Crea el directorio destino si no existe."""
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)

        rows = [record.to_row() for record in records]
        frame = pd.DataFrame(rows, columns=list(self.COLUMNS))

        # El BOM (utf-8-sig) hace que Excel detecte los acentos correctamente y
        # el separador ';' respeta la configuración regional en español.
        temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
        try:
            frame.to_csv(
                temporary,
                sep=";",
                encoding="utf-8-sig",
                index=False,
                float_format="%.4f",
                lineterminator="\r\n",
            )
            os.replace(temporary, target)
        except OSError as exc:
            _LOGGER.error("No se pudo exportar %s: %s", target, exc, exc_info=True)
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            raise
        _LOGGER.info("Exportados %d registros a %s", len(frame), target)
        return target


# Alias público solicitado en el enunciado.
CsvExporter = PandasCsvExporter


def export_records(
    records: Sequence[DetectionRecord], destination: Path | str = "detecciones.csv"
) -> Path:
    """Función de conveniencia para uso desde scripts o tests."""
    return PandasCsvExporter().export(records, destination)
