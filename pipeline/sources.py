"""Fuentes de vídeo: archivo, RTSP, UDP, capturadora HDMI y cámara USB.

Todas las fuentes respetan el contrato :class:`FrameSource`, de modo que la
sesión de análisis no sabe (ni necesita saber) si está leyendo un MP4 o un
stream RTSP. Esto permite añadir más cámaras en el futuro creando otra fuente.
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config.settings import VideoSettings

_LOGGER = logging.getLogger("dpc.sources")

SUPPORTED_FILE_EXTENSIONS = {".mp4", ".mov", ".avi"}


class SourceError(RuntimeError):
    """Error al abrir o leer una fuente de vídeo."""


class FrameSource(ABC):
    """Contrato común de todas las fuentes."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._capture: cv2.VideoCapture | None = None
        self._opened_at = 0.0

    @property
    def is_live(self) -> bool:
        """``True`` en streams/cámaras; ``False`` en vídeos grabados."""
        return False

    @property
    def is_open(self) -> bool:
        """Indica si el dispositivo está abierto."""
        return self._capture is not None and bool(self._capture.isOpened())

    @abstractmethod
    def open(self) -> None:
        """Abre la fuente; lanza :class:`SourceError` si falla."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Devuelve el siguiente frame o ``None`` si no hay señal/EOF."""
        raise NotImplementedError

    def timestamp(self) -> float:
        """Marca temporal en segundos (vídeo: posición; live: tiempo transcurrido)."""
        if self.is_live:
            return max(0.0, time.monotonic() - self._opened_at)
        if self._capture is not None:
            milliseconds = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if milliseconds > 0:
                return milliseconds / 1000.0
        return 0.0

    def close(self) -> None:
        """Libera el handle de OpenCV."""
        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                self._capture = None

    def describe(self) -> str:
        """Descripción corta para la GUI."""
        return self.label


class VideoFileSource(FrameSource):
    """Lector de vídeos MP4 / MOV / AVI."""

    def __init__(self, path: Path | str, settings: VideoSettings) -> None:
        super().__init__(Path(path).name)
        self.path = Path(path)
        self._settings = settings
        self._fps = 30.0
        self._frame_number = 0

    @property
    def is_live(self) -> bool:
        return False

    def open(self) -> None:
        """Valida extensión y abre el archivo con el backend FFmpeg."""
        suffix = self.path.suffix.lower()
        if suffix not in SUPPORTED_FILE_EXTENSIONS:
            raise SourceError(
                f"Formato '{suffix}' no soportado. Usa MP4, MOV o AVI."
            )
        if not self.path.is_file():
            raise SourceError(f"No existe el archivo: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not self.is_open:
            self.close()
            raise SourceError(f"OpenCV no pudo abrir el vídeo: {self.path}")

        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = reported_fps if 0.1 < reported_fps < 1000 else 30.0
        self._apply_resolution()
        self._opened_at = time.monotonic()
        self._frame_number = 0
        _LOGGER.info("Vídeo abierto: %s (%.2f FPS)", self.path, self._fps)

    def _apply_resolution(self) -> None:
        """Aplica la resolución configurada si no es ``auto``."""
        assert self._capture is not None
        requested = self._settings.resolution_tuple()
        if requested is None:
            return
        width, height = requested
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        """Lee un frame; devuelve ``None`` al llegar al final del archivo."""
        if self._capture is None:
            return None
        success, frame = self._capture.read()
        if not success or frame is None:
            return None
        self._frame_number += 1
        return frame

    def timestamp(self) -> float:
        """Usa CAP_PROP_POS_MSEC y, si falla, estima por número de frame."""
        if self._capture is not None:
            milliseconds = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if milliseconds > 0:
                return milliseconds / 1000.0
        return self._frame_number / self._fps if self._fps > 0 else 0.0

    def describe(self) -> str:
        return f"Vídeo: {self.path.name}"


class StreamSource(FrameSource):
    """Receptor genérico de RTSP, UDP, HDMI (capturadora) y USB.

    Formatos aceptados en el cuadro de diálogo:
        * ``rtsp://usuario:password@ip:554/stream1``
        * ``udp://@0.0.0.0:5000``
        * ``0`` / ``1`` -> cámara USB o capturadora HDMI en ese índice
        * ``hdmi:0`` -> alias legible del caso anterior
    """

    _INDEX_PATTERN = re.compile(r"^\d+$")

    def __init__(self, target: str, settings: VideoSettings) -> None:
        super().__init__(target)
        self.target = target.strip()
        self._settings = settings
        self._index: int | None = None
        self._url: str | None = None
        self._parse_target()

    def _parse_target(self) -> None:
        """Normaliza el destino a índice de cámara o URL de red."""
        lower = self.target.lower()
        if lower.startswith("hdmi:"):
            suffix = lower.split(":", 1)[1]
            self._index = int(suffix) if suffix.isdigit() else 0
        elif self._INDEX_PATTERN.match(self.target):
            self._index = int(self.target)
        else:
            self._url = self.target

    @property
    def is_live(self) -> bool:
        return True

    def open(self) -> None:
        """Configura el buffer bajo y abre el stream con el backend adecuado."""
        if self._url and self._url.lower().startswith("rtsp"):
            # Variable de entorno leída por el backend FFmpeg de OpenCV.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self._settings.rtsp_transport}|stimeout;5000000"
            )

        backend = cv2.CAP_ANY
        if self._index is not None and os.name == "nt":
            # CAP_DSHOW reduce el retraso de apertura en Windows.
            backend = cv2.CAP_DSHOW

        source: Any = self._index if self._index is not None else self._url
        self._capture = cv2.VideoCapture(source, backend)
        if self._capture is not None:
            # Si falla CAP_DSHOW (algunas capturadoras), se reintenta con ANY.
            if not self._capture.isOpened() and backend == cv2.CAP_DSHOW:
                self._capture.release()
                self._capture = cv2.VideoCapture(source, cv2.CAP_ANY)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, int(self._settings.buffer_size))

        if not self.is_open:
            self.close()
            raise SourceError(
                f"No se pudo conectar con la fuente: {self.target}. "
                "Comprueba URL, permisos y que ninguna otra app esté usando la cámara."
            )

        self._apply_resolution()
        self._opened_at = time.monotonic()
        _LOGGER.info("Stream abierto: %s", self.target)

    def _apply_resolution(self) -> None:
        """Aplica resolución configurada en cámaras (los streams RTSP la ignoran)."""
        assert self._capture is not None
        requested = self._settings.resolution_tuple()
        if requested is None or self._index is None:
            return
        width, height = requested
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        """Lee un frame; en directo reintenta brevemente ante pérdidas puntuales."""
        if self._capture is None:
            return None
        success, frame = self._capture.read()
        if not success or frame is None:
            if not self.is_live:
                return None
            # Un paquete UDP perdido no debe cortar la sesión.
            for _ in range(3):
                time.sleep(0.03)
                success, frame = self._capture.read()
                if success and frame is not None:
                    return frame
            return None
        return frame

    def describe(self) -> str:
        kind = "Cámara/HDMI" if self._index is not None else "Stream"
        return f"{kind}: {self.target}"
