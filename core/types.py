"""Tipos de dominio compartidos por detección, tracking, GUI y exportación.

Todos los datos que cruzan hilos son dataclasses simples (sin QObject), de modo
que pueden emitirse mediante ``pyqtSignal(object)`` sin problemas de afinidad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """Convierte ``[x, y, w, h]`` a ``[cx, cy, aspect_ratio, height]``.

    Es el formato de estado usado por el filtro de Kalman de ByteTrack.
    """
    box = np.asarray(tlwh, dtype=np.float64)
    return np.array(
        [box[0] + box[2] / 2.0, box[1] + box[3] / 2.0, box[2] / max(box[3], 1e-6), box[3]],
        dtype=np.float64,
    )


def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
    """Convierte ``[cx, cy, aspect_ratio, height]`` a ``[x, y, w, h]``."""
    state = np.asarray(xyah, dtype=np.float64)
    width = state[2] * state[3]
    return np.array(
        [state[0] - width / 2.0, state[1] - state[3] / 2.0, width, state[3]],
        dtype=np.float64,
    )


def tlwh_to_tlbr(tlwh: np.ndarray) -> tuple[int, int, int, int]:
    """Convierte a coordenadas ``(x1, y1, x2, y2)`` enteras para OpenCV."""
    box = np.asarray(tlwh, dtype=np.float64)
    return (
        int(round(box[0])),
        int(round(box[1])),
        int(round(box[0] + box[2])),
        int(round(box[1] + box[3])),
    )


def bbox_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Matriz IoU vectorizada entre dos arrays ``(N, 4)`` en formato tlwh."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2]) * np.maximum(0.0, a[:, 3])
    area_b = np.maximum(0.0, b[:, 2]) * np.maximum(0.0, b[:, 3])
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


# ---------------------------------------------------------------------------
# Tiempo
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float | None, with_milliseconds: bool = False) -> str:
    """Formatea segundos como ``HH:MM:SS`` (o ``HH:MM:SS.mmm``)."""
    if seconds is None or not math.isfinite(float(seconds)):
        return "--:--:--"
    total_ms = max(0, int(float(seconds) * 1000.0))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}.{millis:03d}" if with_milliseconds else base


# ---------------------------------------------------------------------------
# Datos de dominio
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    """Una detección cruda de la clase ``Person``."""

    tlwh: np.ndarray
    confidence: float
    class_id: int = 0

    def __post_init__(self) -> None:
        self.tlwh = np.asarray(self.tlwh, dtype=np.float64)
        self.confidence = float(self.confidence)

    @property
    def xyah(self) -> np.ndarray:
        return tlwh_to_xyah(self.tlwh)

    @property
    def area(self) -> float:
        return float(self.tlwh[2] * self.tlwh[3])


@dataclass(frozen=True, slots=True)
class TrackView:
    """Instantánea pública de un track (lo que consume contador y render)."""

    track_id: int
    tlwh: np.ndarray
    confidence: float
    age: int = 0
    state: str = "TRACKED"
    class_id: int = 0


@dataclass(slots=True)
class SessionStats:
    """Indicadores mostrados en el panel derecho de la GUI."""

    fps: float = 0.0
    visible: int = 0
    total_unique: int = 0
    frames_processed: int = 0
    dropped_frames: int = 0
    status: str = "🔴 Sin señal"
    device_mode: str = "Modo CPU"
    source_name: str = "Sin fuente"
    timestamp: str = "--:--:--"
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class ProcessedFrame:
    """Resultado de procesar un frame; viaja del worker a la GUI."""

    image: np.ndarray
    tracks: tuple[TrackView, ...] = ()
    stats: SessionStats = field(default_factory=SessionStats)
    timestamp: str = "--:--:--"
    # True cuando el frame se descartó por el límite de FPS objetivo; en ese
    # caso ``image`` es la última imagen anotada para que la GUI no parpadee.
    skipped: bool = False
    skipped: bool = False


class SessionState(Enum):
    """Estados del ciclo de vida de una sesión de análisis."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
