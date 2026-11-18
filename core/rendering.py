"""Dibujo de anotaciones y HUD sobre el frame.

Responsabilidad aislada: si mañana se quiere un render sin HUD (para grabar el
vídeo limpio) basta con inyectar otra implementación en la sesión.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .types import SessionStats, TrackView

_COLORS: tuple[tuple[int, int, int], ...] = (
    (46, 204, 113),  # verde
    (52, 152, 219),  # azul
    (241, 196, 15),  # amarillo
    (230, 126, 34),  # naranja
    (155, 89, 182),  # violeta
    (26, 188, 156),  # turquesa
)


class FrameRenderer:
    """Pinta bounding boxes, IDs, confianza e indicadores."""

    def __init__(self, show_hud: bool = True) -> None:
        self.show_hud = bool(show_hud)

    @staticmethod
    def _color_for(track_id: int) -> tuple[int, int, int]:
        return _COLORS[track_id % len(_COLORS)]

    def render(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackView],
        stats: SessionStats,
        timestamp: str,
    ) -> np.ndarray:
        """Devuelve una copia anotada del frame (no muta la entrada)."""
        output = frame.copy()
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        for track in tracks:
            x1, y1, x2, y2 = (
                int(track.tlwh[0]),
                int(track.tlwh[1]),
                int(track.tlwh[0] + track.tlwh[2]),
                int(track.tlwh[1] + track.tlwh[3]),
            )
            color = self._color_for(track.track_id)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            percentage = int(round(float(track.confidence) * 100))
            label = f"Persona ID {track.track_id} {percentage}%"
            self._draw_label(output, label, x1, y1, color)

        if self.show_hud:
            self._draw_hud(output, stats, timestamp)
        return output

    @staticmethod
    def _draw_label(
        frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]
    ) -> None:
        """Dibuja una etiqueta con fondo semitransparente legible."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        top = max(0, y - text_height - baseline - 6)
        left = max(0, x)
        right = min(frame.shape[1], left + text_width + 8)
        bottom = min(frame.shape[0], top + text_height + baseline + 6)
        if right <= left or bottom <= top:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), color, thickness=cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(
            frame,
            text,
            (left + 4, bottom - baseline - 3),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_hud(frame: np.ndarray, stats: SessionStats, timestamp: str) -> None:
        """Panel superior con contadores, FPS, modo y marca de tiempo."""
        lines = (
            f"Actuales: {stats.visible}    Total unicas: {stats.total_unique}",
            f"FPS: {stats.fps:.1f}    {stats.device_mode}    T: {timestamp}",
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        widths = [cv2.getTextSize(text, font, scale, thickness)[0][0] for text in lines]
        panel_width = min(frame.shape[1], max(widths) + 20)
        panel_height = 62
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (15, 20, 25), cv2.FILLED)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (18, 32 + index * 24),
                font,
                scale,
                (240, 240, 240),
                thickness,
                cv2.LINE_AA,
            )
