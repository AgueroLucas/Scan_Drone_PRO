"""Ventana principal de Drone People Counter Pro.

Panel izquierdo: control y selección de fuente.
Panel central: vídeo anotado en tiempo real.
Panel derecho: indicadores (únicas, visibles, FPS, estado, hardware).

Toda la lógica pesada se delega al worker; aquí sólo se actualizan widgets.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from config.settings import ConfigManager
from core.counter import PeopleCounter
from core.events import EventBus
from core.rendering import FrameRenderer
from core.types import SessionStats
from detection.detector import PersonDetector
from detection.hardware import detect_hardware
from export.csv_exporter import PandasCsvExporter
from gui.settings_dialog import SettingsDialog
from gui.style import DARK_STYLE
from gui.worker import AnalysisWorker
from logs.logging_setup import EventLogger, log_error, log_event
from pipeline.session import DetectionSession
from pipeline.sources import FrameSource, SourceError, StreamSource, VideoFileSource
from tracking.byte_track import ByteTrackTracker

_LOGGER = logging.getLogger("dpc.gui")


class MainWindow(QMainWindow):
    """Interfaz gráfica principal."""

    def __init__(self) -> None:
        super().__init__()
        self._config_manager = ConfigManager()
        self._config = self._config_manager.load()
        self._device = detect_hardware(self._config.use_gpu)

        self._source: FrameSource | None = None
        self._session: DetectionSession | None = None
        self._worker: AnalysisWorker | None = None
        self._busy = False

        self.setWindowTitle("Drone People Counter Pro")
        self.resize(1360, 800)
        self.setStyleSheet(DARK_STYLE)
        self._build_interface()
        self._refresh_device_label()
        self._set_status("🔴 Sin señal")
        self.statusBar().showMessage("Listo. Abra un vídeo o conecte un stream para comenzar.")
        log_event("Interfaz gráfica inicializada | %s", self._device.description())

    # ------------------------------------------------------------------
    # Construcción de la UI
    # ------------------------------------------------------------------
    def _build_interface(self) -> None:
        """Crea los tres paneles y la barra de menús."""
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        root.addWidget(self._build_control_panel(), stretch=0)
        root.addWidget(self._build_video_panel(), stretch=1)
        root.addWidget(self._build_indicators_panel(), stretch=0)

        self.setStatusBar(QStatusBar(self))
        self._build_menu()

    def _build_control_panel(self) -> QWidget:
        """Panel izquierdo con todos los botones de acción."""
        panel = QGroupBox("Control")
        panel.setFixedWidth(230)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        self.button_open_video = QPushButton("📂  Abrir vídeo")
        self.button_connect_stream = QPushButton("📡  Conectar stream")
        self.button_start = QPushButton("▶  Iniciar análisis")
        self.button_pause = QPushButton("⏸  Pausar")
        self.button_stop = QPushButton("⏹  Detener")
        self.button_export = QPushButton("💾  Exportar CSV")
        self.button_settings = QPushButton("⚙  Configuración")
        self.button_settings.setObjectName("secondary")
        self.button_stop.setObjectName("danger")

        self.button_open_video.clicked.connect(self._open_video)
        self.button_connect_stream.clicked.connect(self._connect_stream)
        self.button_start.clicked.connect(self._start_analysis)
        self.button_pause.clicked.connect(self._toggle_pause)
        self.button_stop.clicked.connect(self._stop_analysis)
        self.button_export.clicked.connect(self._export_csv)
        self.button_settings.clicked.connect(self._open_settings)

        for button in (
            self.button_open_video,
            self.button_connect_stream,
            self.button_start,
            self.button_pause,
            self.button_stop,
            self.button_export,
            self.button_settings,
        ):
            layout.addWidget(button)

        self.label_source = QLabel("Fuente: ninguna")
        self.label_source.setWordWrap(True)
        self.label_source.setObjectName("metric-title")
        layout.addStretch(1)
        layout.addWidget(self.label_source)

        self.button_pause.setEnabled(False)
        self.button_stop.setEnabled(False)
        return panel

    def _build_video_panel(self) -> QWidget:
        """Panel central con el vídeo escalado manteniendo proporción."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("Sin señal")
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_label)
        return container

    def _build_indicators_panel(self) -> QWidget:
        """Panel derecho con los indicadores solicitados."""
        panel = QGroupBox("Indicadores")
        panel.setFixedWidth(260)
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)

        layout.addWidget(self._metric_title("Total únicas detectadas"))
        self.label_total = self._metric_value("0", green=True)
        layout.addWidget(self.label_total)

        layout.addWidget(self._metric_title("Personas visibles actualmente"))
        self.label_visible = self._metric_value("0")
        layout.addWidget(self.label_visible)

        layout.addWidget(self._metric_title("FPS"))
        self.label_fps = self._metric_value("0.0")
        layout.addWidget(self.label_fps)

        layout.addSpacing(10)
        layout.addWidget(self._metric_title("Estado del sistema"))
        self.label_status = QLabel("🔴 Sin señal")
        self.label_status.setWordWrap(True)
        layout.addWidget(self.label_status)

        layout.addWidget(self._metric_title("Hardware de inferencia"))
        self.label_device = QLabel("Modo CPU")
        self.label_device.setWordWrap(True)
        layout.addWidget(self.label_device)

        layout.addWidget(self._metric_title("Timestamp"))
        self.label_timestamp = QLabel("--:--:--")
        layout.addWidget(self.label_timestamp)

        layout.addWidget(self._metric_title("Frames procesados"))
        self.label_frames = QLabel("0")
        layout.addWidget(self.label_frames)

        layout.addStretch(1)
        return panel

    @staticmethod
    def _metric_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("metric-title")
        return label

    @staticmethod
    def _metric_value(text: str, green: bool = False) -> QLabel:
        label = QLabel(text)
        label.setObjectName("metric-value-green" if green else "metric-value")
        return label

    def _build_menu(self) -> None:
        """Menú de archivo y acceso a configuración."""
        file_menu = self.menuBar().addMenu("Archivo")
        action_video = QAction("Abrir vídeo...", self)
        action_video.triggered.connect(self._open_video)
        action_stream = QAction("Conectar stream...", self)
        action_stream.triggered.connect(self._connect_stream)
        action_export = QAction("Exportar CSV...", self)
        action_export.triggered.connect(self._export_csv)
        action_exit = QAction("Salir", self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_video)
        file_menu.addAction(action_stream)
        file_menu.addAction(action_export)
        file_menu.addSeparator()
        file_menu.addAction(action_exit)

        settings_menu = self.menuBar().addMenu("Herramientas")
        action_settings = QAction("Configuración...", self)
        action_settings.triggered.connect(self._open_settings)
        settings_menu.addAction(action_settings)

    # ------------------------------------------------------------------
    # Selección de fuente
    # ------------------------------------------------------------------
    def _ensure_idle(self) -> bool:
        """Devuelve ``True`` si no hay análisis en marcha."""
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Análisis activo", "Detenga el análisis antes de cambiar la fuente.")
            return False
        return True

    def _open_video(self) -> None:
        """Selecciona un MP4, MOV o AVI."""
        if not self._ensure_idle():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar vídeo grabado",
            "",
            "Vídeos (*.mp4 *.MOV *.mov *.avi *.AVI)",
        )
        if not path:
            return
        try:
            self._source = VideoFileSource(Path(path), self._config.video)
            self.label_source.setText(f"Fuente: {Path(path).name}")
            self._set_status("🟡 Procesando (en espera)")
            self.statusBar().showMessage(f"Vídeo seleccionado: {path}")
            log_event("Vídeo seleccionado: %s", path)
        except SourceError as exc:
            self._show_error(str(exc))

    def _connect_stream(self) -> None:
        """Solicita RTSP, UDP, índice USB o ``hdmi:N``."""
        if not self._ensure_idle():
            return
        target, accepted = QInputDialog.getText(
            self,
            "Conectar stream",
            "URL RTSP/UDP, índice de cámara o hdmi:0:",
            text="rtsp://usuario:password@192.168.0.10:554/stream1",
        )
        if not accepted or not target.strip():
            return
        try:
            self._source = StreamSource(target, self._config.video)
            self.label_source.setText(f"Fuente: {self._source.describe()}")
            self._set_status("🟡 Procesando (en espera)")
            self.statusBar().showMessage(f"Stream configurado: {target}")
            log_event("Stream seleccionado: %s", target)
        except (SourceError, ValueError) as exc:
            self._show_error(str(exc))

    # ------------------------------------------------------------------
    # Control del análisis
    # ------------------------------------------------------------------
    def _build_session_factory(self, source: FrameSource):
        """Crea la fábrica de sesiones inyectando todas las dependencias."""
        config = self._config_manager.load()
        device = detect_hardware(config.use_gpu)
        self._config = config
        self._device = device
        self._refresh_device_label()

        def factory() -> DetectionSession:
            bus = EventBus()
            bus.subscribe("*", EventLogger())
            session = DetectionSession(
                config=config,
                device=device,
                source=source,
                detector=PersonDetector(config.detection, device),
                tracker=ByteTrackTracker(config.tracking),
                counter=PeopleCounter(),
                exporter=PandasCsvExporter(),
                renderer=FrameRenderer(show_hud=True),
                bus=bus,
            )
            self._session = session
            return session

        return factory

    def _start_analysis(self) -> None:
        """Arranca el worker con la fuente seleccionada."""
        if self._busy or (self._worker is not None and self._worker.isRunning()):
            return
        if self._source is None:
            QMessageBox.information(
                self, "Sin fuente", "Abra un vídeo o conecte un stream antes de iniciar."
            )
            return

        self._busy = True
        self.button_start.setEnabled(False)
        self.button_pause.setEnabled(True)
        self.button_pause.setText("⏸  Pausar")
        self.button_stop.setEnabled(True)
        self.video_label.setText("Conectando...")

        factory = self._build_session_factory(self._source)
        worker = AnalysisWorker(factory, parent=self)
        worker.frame_ready.connect(self._on_frame)
        worker.status_changed.connect(self._set_status)
        worker.error_occurred.connect(self._on_error)
        worker.session_finished.connect(self._on_session_finished)
        self._worker = worker
        worker.start()

    def _toggle_pause(self) -> None:
        """Pausa o reanuda el análisis."""
        if self._worker is None or not self._worker.isRunning():
            return
        if self._worker.is_paused:
            self._worker.resume()
            self.button_pause.setText("⏸  Pausar")
        else:
            self._worker.pause()
            self.button_pause.setText("▶  Reanudar")

    def _stop_analysis(self) -> None:
        """Detiene el bucle; el worker guarda el CSV final automáticamente."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.button_pause.setEnabled(False)
            self.statusBar().showMessage("Deteniendo análisis y guardando resultados...")

    def _export_csv(self) -> None:
        """Exporta manualmente los registros acumulados."""
        if self._session is None:
            QMessageBox.information(self, "Sin datos", "Todavía no hay una sesión de análisis.")
            return
        try:
            path = self._session.export_csv()
            self.statusBar().showMessage(f"CSV exportado: {path}")
            QMessageBox.information(self, "Exportación completada", f"Fichero generado:\n{path}")
        except OSError as exc:
            self._show_error(f"No se pudo exportar el CSV: {exc}")

    def _open_settings(self) -> None:
        """Edita y guarda ``config.json``."""
        dialog = SettingsDialog(self._config, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        updated = dialog.updated_config(self._config)
        try:
            self._config_manager.save(updated)
        except OSError as exc:
            self._show_error(f"No se pudo guardar config.json: {exc}")
            return
        self._config = updated
        self._device = detect_hardware(updated.use_gpu)
        self._refresh_device_label()
        self.statusBar().showMessage(
            "Configuración guardada. Reinicie el análisis para aplicar todos los cambios."
        )
        log_event("Configuración actualizada desde la interfaz")

    # ------------------------------------------------------------------
    # Slots de señales del worker
    # ------------------------------------------------------------------
    def _on_frame(self, processed) -> None:
        """Convierte el frame BGR a ``QPixmap`` y actualiza indicadores."""
        frame = processed.image
        if frame is None or frame.size == 0:
            return
        height, width = frame.shape[:2]
        bytes_per_line = int(frame.strides[0])
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(image.copy())
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self._update_stats(processed.stats)

    def _update_stats(self, stats: SessionStats) -> None:
        """Refresca los QLabel del panel derecho."""
        self.label_total.setText(str(stats.total_unique))
        self.label_visible.setText(str(stats.visible))
        self.label_fps.setText(f"{stats.fps:.1f}")
        self.label_timestamp.setText(stats.timestamp)
        self.label_frames.setText(str(stats.frames_processed))

    def _set_status(self, text: str) -> None:
        """Aplica el indicador visual 🟢 / 🟡 / 🔴."""
        self.label_status.setText(text)
        if "🟢" in text:
            self.label_status.setObjectName("status-active")
        elif "🟡" in text:
            self.label_status.setObjectName("status-waiting")
        else:
            self.label_status.setObjectName("status-error")
        self.label_status.style().polish(self.label_status)
        self.statusBar().showMessage(text)

    def _on_error(self, message: str) -> None:
        """Muestra un error del worker y deja el sistema en ``Sin señal``."""
        log_error("Error mostrado en la GUI: %s", message)
        self._set_status("🔴 Sin señal")
        QMessageBox.critical(self, "Error de análisis", message)

    def _on_session_finished(self, result) -> None:
        """Rehabilita controles y muestra el resumen final."""
        if result is not None:
            self._update_stats(
                SessionStats(
                    total_unique=result.total_unique,
                    visible=0,
                    elapsed_seconds=result.elapsed_seconds,
                    status="🔴 Sin señal",
                )
            )
            message = (
                f"Sesión finalizada ({result.reason}). "
                f"Personas únicas: {result.total_unique}."
            )
            if result.export_path is not None:
                message += f" CSV: {result.export_path}"
            self.statusBar().showMessage(message)
            log_event(
                "Sesión finalizada | motivo=%s | únicas=%d | frames=%d",
                result.reason,
                result.total_unique,
                result.frames_processed,
            )
        self._busy = False
        self.button_start.setEnabled(True)
        self.button_pause.setEnabled(False)
        self.button_stop.setEnabled(False)
        self.button_pause.setText("⏸  Pausar")
        self._set_status("🔴 Sin señal")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def _refresh_device_label(self) -> None:
        """Muestra ``Modo GPU`` o ``Modo CPU`` con detalle del dispositivo."""
        self.label_device.setText(self._device.description())

    def _show_error(self, message: str) -> None:
        """Error de usuario con registro en ``errores.log``."""
        log_error("%s", message)
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event) -> None:
        """Detiene el worker antes de cerrar la ventana."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(5000)
        event.accept()
