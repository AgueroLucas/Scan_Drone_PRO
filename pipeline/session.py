"""Sesión de análisis: orquesta fuente -> detección -> tracking -> conteo -> salida.

La sesión es un objeto de dominio puro (sin Qt). El worker de la GUI se limita
a llamar a :meth:`DetectionSession.process_next` en bucle. Esta separación
permite:

    * ejecutar la misma sesión en tests o en un servicio;
    * crear varias sesiones simultáneas en el futuro (una por dron);
    * cambiar de fuente, modelo o exportador por inyección de dependencias.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config.settings import AppConfig
from core.counter import PeopleCounter
from core.events import EventBus, Events
from core.performance import FpsMeter
from core.rendering import FrameRenderer
from core.types import Detection, ProcessedFrame, SessionState, SessionStats, TrackView, format_timestamp
from detection.detector import PersonDetector
from detection.hardware import DeviceInfo
from export.csv_exporter import ResultExporter
from export.records import DetectionRecord
from tracking.byte_track import ByteTrackTracker

from .sources import FrameSource

_LOGGER = logging.getLogger("dpc.session")


@dataclass(frozen=True)
class SessionResult:
    """Resumen devuelto al terminar una sesión."""

    records: tuple[DetectionRecord, ...]
    frames_processed: int
    dropped_frames: int
    total_unique: int
    elapsed_seconds: float
    export_path: Path | None
    reason: str


class DetectionSession:
    """Una sesión de conteo para una única fuente de vídeo."""

    def __init__(
        self,
        *,
        config: AppConfig,
        device: DeviceInfo,
        source: FrameSource,
        detector: PersonDetector,
        tracker: ByteTrackTracker,
        counter: PeopleCounter,
        exporter: ResultExporter,
        renderer: FrameRenderer,
        bus: EventBus,
    ) -> None:
        self.config = config
        self.device = device
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.exporter = exporter
        self.renderer = renderer
        self.bus = bus

        # ByteTrack usa la confianza de configuración para separar detecciones
        # "altas" (crean IDs) de "bajas" (sólo recuperan tracks existentes).
        self.tracker.configure_score_threshold(config.detection.min_confidence)

        self._state = SessionState.IDLE
        self._state_lock = threading.RLock()
        self._fps_meter = FpsMeter(window_seconds=1.0)
        self._frames_processed = 0
        self._dropped_frames = 0
        self._last_frame_time: float | None = None
        self._last_image = None
        self._last_autosave = time.monotonic()
        self._started_at = time.monotonic()
        self._last_timestamp = "--:--:--"
        self._reason = ""

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    @property
    def state(self) -> SessionState:
        """Estado actual de la sesión."""
        with self._state_lock:
            return self._state

    def open(self) -> None:
        """Abre la fuente. Lanza ``SourceError`` si no hay señal."""
        self.source.open()
        with self._state_lock:
            self._state = SessionState.IDLE
        self._started_at = time.monotonic()
        self._last_autosave = self._started_at

    def start(self) -> None:
        """Marca la sesión como activa y publica el evento de inicio."""
        with self._state_lock:
            self._state = SessionState.RUNNING
        self._reason = ""
        _LOGGER.info("Análisis iniciado | %s | %s", self.source.describe(), self.device.description())
        self.bus.emit(Events.SESSION_STARTED, {"source": self.source.describe()})

    def pause(self) -> None:
        """Pausa el procesamiento (el worker deja de pedir frames)."""
        with self._state_lock:
            if self._state == SessionState.RUNNING:
                self._state = SessionState.PAUSED
        self.bus.emit(Events.SESSION_PAUSED)

    def resume(self) -> None:
        """Reanuda el procesamiento."""
        with self._state_lock:
            if self._state == SessionState.PAUSED:
                self._state = SessionState.RUNNING
        self.bus.emit(Events.SESSION_RESUMED)

    def stop(self, reason: str = "detenido por el usuario") -> None:
        """Detiene la sesión sin cerrar los recursos todavía."""
        with self._state_lock:
            self._state = SessionState.STOPPED
        self._reason = reason
        self.counter.clear_visible()

    def close(self) -> None:
        """Guarda el CSV final, libera la fuente y publica el fin de sesión."""
        self.counter.clear_visible()
        with self._state_lock:
            if self._state not in (SessionState.STOPPED, SessionState.ERROR):
                self._state = SessionState.STOPPED
        try:
            self.export_csv()
        except Exception as exc:  # pragma: no cover - defensivo
            _LOGGER.error("Error al guardar el CSV final: %s", exc, exc_info=True)
            self.bus.emit(Events.ERROR, {"message": str(exc)})
        self.source.close()
        _LOGGER.info("Análisis finalizado (%s)", self._reason or "sin motivo registrado")
        self.bus.emit(
            Events.SESSION_STOPPED,
            {
                "reason": self._reason,
                "frames": self._frames_processed,
                "unicos": self.counter.total_unique,
            },
        )

    # ------------------------------------------------------------------
    # Procesamiento de un frame
    # ------------------------------------------------------------------
    def process_next(self) -> ProcessedFrame | None:
        """Lee, detecta, rastrea, cuenta, dibuja y devuelve un frame.

        Returns:
            ``None`` cuando la fuente se agotó o se perdió la señal.
        """
        if self.state not in (SessionState.RUNNING, SessionState.PAUSED):
            return None

        frame = self.source.read()
        if frame is None:
            self.stop("fin de vídeo o pérdida de señal")
            return None

        now = time.monotonic()
        elapsed = now - self._started_at

        # Limitación de FPS: en directo DESCARTAMOS el frame (no dormimos) para
        # no acumular latencia; en vídeo también reduce CPU innecesaria.
        target_fps = int(self.config.video.target_fps)
        if target_fps > 0 and self._last_frame_time is not None:
            minimum_interval = 1.0 / float(target_fps)
            if (now - self._last_frame_time) < minimum_interval:
                self._dropped_frames += 1
                stats = self._build_stats(fps=self._fps_meter.value, elapsed=elapsed)
                return ProcessedFrame(
                    image=self._last_image if self._last_image is not None else frame,
                    stats=stats,
                    timestamp=self._last_timestamp,
                    skipped=True,
                )

        self._last_frame_time = now

        # --- 1) Detección de personas -------------------------------------
        detections: Sequence[Detection] = self.detector.detect(frame)

        # --- 2) Tracking ByteTrack ----------------------------------------
        tracks: Sequence[TrackView] = self.tracker.update(detections)

        # --- 3) Conteo único ----------------------------------------------
        timestamp_seconds = self.source.timestamp()
        timestamp_text = format_timestamp(timestamp_seconds)
        visible, total, new_ids = self.counter.update(tracks, timestamp_text)
        for track_id in new_ids:
            self.bus.emit(
                Events.PERSON_DETECTED,
                {
                    "id": track_id,
                    "timestamp": timestamp_text,
                    "total": total,
                },
            )

        # --- 4) Render + métricas ----------------------------------------
        interim_stats = self._build_stats(fps=self._fps_meter.value, elapsed=elapsed)
        interim_stats.visible = visible
        interim_stats.total_unique = total
        annotated = self.renderer.render(frame, tracks, interim_stats, timestamp_text)
        self._last_image = annotated
        self._last_timestamp = timestamp_text

        self._frames_processed += 1
        fps = self._fps_meter.tick(now)
        stats = self._build_stats(fps=fps, elapsed=elapsed)
        stats.visible = visible
        stats.total_unique = total

        # --- 5) Autoguardado del CSV --------------------------------------
        autosave = float(self.config.video.autosave_seconds)
        if (now - self._last_autosave) >= autosave:
            if self.counter.total_unique > 0 or not self._default_destination().exists():
                try:
                    self.export_csv()
                except Exception as exc:  # pragma: no cover - defensivo
                    _LOGGER.error("Autoguardado CSV fallido: %s", exc, exc_info=True)
            self._last_autosave = now

        return ProcessedFrame(
            image=annotated,
            tracks=tuple(tracks),
            stats=stats,
            timestamp=timestamp_text,
        )

    def _build_stats(self, fps: float, elapsed: float) -> SessionStats:
        """Construye el snapshot de indicadores para la GUI y el HUD."""
        with self._state_lock:
            if self._state == SessionState.RUNNING:
                status = "🟢 Sistema activo"
            elif self._state == SessionState.PAUSED:
                status = "🟡 Procesando (pausado)"
            else:
                status = "🔴 Sin señal"
        return SessionStats(
            fps=fps,
            visible=self.counter.visible_count,
            total_unique=self.counter.total_unique,
            frames_processed=self._frames_processed,
            dropped_frames=self._dropped_frames,
            status=status,
            device_mode=self.device.mode_label,
            source_name=self.source.describe(),
            timestamp=self._last_timestamp,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Exportación y resultado
    # ------------------------------------------------------------------
    def _default_destination(self) -> Path:
        """Ruta ``<export_path>/<export_filename>``."""
        return self.config.export_directory() / self.config.export_filename

    def export_csv(self, destination: Path | str | None = None) -> Path:
        """Exporta el estado actual del contador a CSV y publica el evento."""
        records = self.counter.snapshot()
        target = Path(destination) if destination is not None else self._default_destination()
        path = self.exporter.export(records, target)
        self.bus.emit(Events.EXPORT_COMPLETED, {"path": str(path), "registros": len(records)})
        return path

    def result(self, reason: str = "") -> SessionResult:
        """Genera el resumen final de la sesión."""
        records = self.counter.snapshot()
        destination = self._default_destination()
        return SessionResult(
            records=tuple(records),
            frames_processed=self._frames_processed,
            dropped_frames=self._dropped_frames,
            total_unique=len(records),
            elapsed_seconds=max(0.0, time.monotonic() - self._started_at),
            export_path=destination if destination.exists() else None,
            reason=reason or self._reason,
        )
