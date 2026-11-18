"""Contador de personas por sesión.

Reglas de negocio implementadas:

    * ``Personas actuales`` = tracks con detección fresca en el frame actual.
    * ``Total únicas`` = número de track-IDs distintos vistos en la sesión.
    * Un mismo ID se contabiliza una sola vez; las visitas posteriores sólo
      actualizan la confianza máxima y la última fecha/hora.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Iterable, Sequence

from export.records import DetectionRecord

from .types import TrackView


class PeopleCounter:
    """Contador thread-safe e independiente del origen de vídeo."""

    def __init__(self) -> None:
        self._records: dict[int, DetectionRecord] = {}
        self._visible: set[int] = set()
        self._lock = threading.RLock()

    def update(
        self,
        tracks: Sequence[TrackView],
        timestamp: str,
        now: float | None = None,
    ) -> tuple[int, int, list[int]]:
        """Incorpora los tracks de un frame.

        Args:
            tracks: tracks confirmados por ByteTrack en este frame.
            timestamp: marca de tiempo del vídeo (``HH:MM:SS``).
            now: epoch opcional para tests; por defecto ``time.time()``.

        Returns:
            ``(visibles, total_únicas, ids_nuevos)``.
        """
        current = float(now if now is not None else time.time())
        moment = datetime.fromtimestamp(current)

        # Formato exigido: dd/mm/YYYY y HH:MM:SS (compatible con Excel ES).
        date_text = moment.strftime("%d/%m/%Y")
        time_text = moment.strftime("%H:%M:%S")

        visible_ids: set[int] = set()
        new_ids: list[int] = []

        with self._lock:
            for track in tracks:
                if track.state != "TRACKED":
                    continue
                visible_ids.add(track.track_id)
                record = self._records.get(track.track_id)
                if record is None:
                    record = DetectionRecord(
                        track_id=track.track_id,
                        fecha=date_text,
                        hora=time_text,
                        timestamp=timestamp,
                        confidence=float(track.confidence),
                        first_seen=current,
                        last_seen=current,
                    )
                    self._records[track.track_id] = record
                    new_ids.append(track.track_id)
                else:
                    # La persona ya fue contada: no se añade otro registro.
                    record.confidence = max(record.confidence, float(track.confidence))
                    record.last_seen = current

            self._visible = visible_ids
            return len(visible_ids), len(self._records), new_ids

    def snapshot(self) -> list[DetectionRecord]:
        """Copia de los registros para exportar sin bloquear el pipeline."""
        with self._lock:
            return [DetectionRecord(**vars(record)) for record in self._records.values()]

    def clear_visible(self) -> None:
        """Pone a cero el contador de visibles (pérdida de señal/parada)."""
        with self._lock:
            self._visible.clear()

    def reset(self) -> None:
        """Reinicia por completo la sesión de conteo."""
        with self._lock:
            self._records.clear()
            self._visible.clear()

    @property
    def visible_count(self) -> int:
        """Personas visibles en el último frame procesado."""
        with self._lock:
            return len(self._visible)

    @property
    def total_unique(self) -> int:
        """Personas únicas detectadas desde el inicio de la sesión."""
        with self._lock:
            return len(self._records)
