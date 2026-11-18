"""Hilo de análisis (QThread) que ejecuta la sesión sin bloquear la GUI.

La comunicación con la ventana se hace con señales Qt; el pipeline no conoce
PyQt6, por lo que puede reutilizarse en otros contextos (CLI, servicio, tests).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

from pipeline.session import DetectionSession, SessionResult

_LOGGER = logging.getLogger("dpc.worker")


class AnalysisWorker(QThread):
    """Bucle de procesamiento en un hilo separado."""

    frame_ready = pyqtSignal(object)      # ProcessedFrame
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    session_finished = pyqtSignal(object)  # SessionResult

    def __init__(self, session_factory: Callable[[], DetectionSession], parent=None) -> None:
        super().__init__(parent)
        self._session_factory = session_factory
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._session: DetectionSession | None = None

    def run(self) -> None:
        """Abre la fuente y procesa frames hasta parada, fin de vídeo o error."""
        session: DetectionSession | None = None
        reason = "finalizado"
        try:
            session = self._session_factory()
            self._session = session
            session.open()
            session.start()
            self.status_changed.emit("🟡 Procesando")

            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    self.msleep(50)  # Espera cooperativa; no consume CPU.
                    continue
                processed = session.process_next()
                if processed is None:
                    reason = "fin de vídeo o pérdida de señal"
                    self.status_changed.emit("🔴 Sin señal")
                    break
                self.frame_ready.emit(processed)
            else:
                reason = "detenido por el usuario"
        except Exception as exc:  # pragma: no cover - captura de última línea
            reason = f"error: {exc}"
            _LOGGER.error("Error en el hilo de análisis: %s", exc, exc_info=True)
            self.status_changed.emit("🔴 Sin señal")
            self.error_occurred.emit(str(exc))
        finally:
            if session is None:
                # Falló la creación de dependencias (por ejemplo, el modelo):
                # se notifica igualmente para no dejar la GUI bloqueada.
                self.session_finished.emit(None)
                return
            try:
                session.stop(reason)
                # close() guarda el CSV final antes de construir el resumen,
                # así ``result.export_path`` siempre apunta a un fichero real.
                session.close()
                result: SessionResult = session.result(reason)
                self.session_finished.emit(result)
            except Exception as exc:  # pragma: no cover - defensivo
                _LOGGER.error("Error al cerrar la sesión: %s", exc, exc_info=True)
                self.error_occurred.emit(f"Error al cerrar la sesión: {exc}")

    def pause(self) -> None:
        """Pausa el bucle de forma cooperativa."""
        self._pause_event.set()
        if self._session is not None:
            self._session.pause()
        self.status_changed.emit("🟡 Procesando (pausado)")

    def resume(self) -> None:
        """Reanuda el bucle."""
        self._pause_event.clear()
        if self._session is not None:
            self._session.resume()
        self.status_changed.emit("🟢 Sistema activo")

    def stop(self) -> None:
        """Solicita la parada del bucle."""
        self._stop_event.set()
        self._pause_event.clear()
        if self._session is not None:
            self._session.stop("detenido por el usuario")
        self.status_changed.emit("🔴 Sin señal")

    @property
    def is_paused(self) -> bool:
        """Indica si el worker está pausado."""
        return self._pause_event.is_set()
