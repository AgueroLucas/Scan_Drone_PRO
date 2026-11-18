"""Medición de rendimiento del pipeline."""

from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """FPS media calculada sobre una ventana temporal deslizante.

    Se prefieren measurements reales a ``1 / tiempo_de_inferencia`` porque
    incluyen lectura de fuente, tracking y renderizado.
    """

    def __init__(self, window_seconds: float = 1.0) -> None:
        self._window = max(0.1, float(window_seconds))
        self._samples: deque[float] = deque()

    def tick(self, now: float | None = None) -> float:
        """Registra un frame y devuelve los FPS actuales."""
        current = float(now if now is not None else time.monotonic())
        self._samples.append(current)
        cutoff = current - self._window
        while self._samples and self._samples[0] < cutoff:
            self._samples.popleft()
        if len(self._samples) < 2:
            return 0.0
        span = self._samples[-1] - self._samples[0]
        return 0.0 if span <= 0 else (len(self._samples) - 1) / span

    def reset(self) -> None:
        """Reinicia la ventana de medidas."""
        self._samples.clear()

    @property
    def value(self) -> float:
        """Último valor calculado sin registrar un nuevo frame."""
        if len(self._samples) < 2:
            return 0.0
        span = self._samples[-1] - self._samples[0]
        return 0.0 if span <= 0 else (len(self._samples) - 1) / span
