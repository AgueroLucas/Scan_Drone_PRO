"""Diálogo de configuración editable por el usuario.

Todos los campos se validan antes de guardar; el ``ConfigManager`` vuelve a
acotar los valores por si el fichero ``config.json`` se edita a mano.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from config.settings import AppConfig


class SettingsDialog(QDialog):
    """Formulario modal de ``config.json``."""

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configuración — Drone People Counter Pro")
        self.setMinimumWidth(460)

        self._use_gpu = QCheckBox("Usar GPU NVIDIA (CUDA) si está disponible")
        self._use_gpu.setChecked(config.use_gpu)

        self._model = QLineEdit(config.detection.model_path)
        self._min_confidence = QDoubleSpinBox()
        self._min_confidence.setRange(0.01, 0.99)
        self._min_confidence.setSingleStep(0.05)
        self._min_confidence.setDecimals(2)
        self._min_confidence.setValue(config.detection.min_confidence)

        self._image_size = QSpinBox()
        self._image_size.setRange(160, 2560)
        self._image_size.setSingleStep(32)
        self._image_size.setValue(config.detection.image_size)

        self._fp16 = QCheckBox("Inferencia FP16 en GPU (más rápido)")
        self._fp16.setChecked(config.detection.fp16)

        self._target_fps = QSpinBox()
        self._target_fps.setRange(0, 240)
        self._target_fps.setValue(config.video.target_fps)

        self._resolution = QLineEdit(config.video.resolution)
        self._resolution.setPlaceholderText("auto o 1280x720")

        self._transport = QLineEdit(config.video.rtsp_transport)
        self._transport.setPlaceholderText("udp o tcp")

        self._max_lost = QSpinBox()
        self._max_lost.setRange(1, 300)
        self._max_lost.setValue(config.tracking.max_lost)

        self._export_path = QLineEdit(config.export_path)
        self._browse = QPushButton("...")
        self._browse.setFixedWidth(32)
        self._browse.clicked.connect(self._choose_directory)

        export_row = QHBoxLayout()
        export_row.addWidget(self._export_path)
        export_row.addWidget(self._browse)

        form = QFormLayout()
        form.addRow(self._use_gpu)
        form.addRow("Modelo YOLO:", self._model)
        form.addRow("Confianza mínima:", self._min_confidence)
        form.addRow("Resolución modelo:", self._image_size)
        form.addRow(self._fp16)
        form.addRow("FPS objetivo (0 = sin límite):", self._target_fps)
        form.addRow("Resolución captura:", self._resolution)
        form.addRow("Transporte RTSP:", self._transport)
        form.addRow("Frames máx. perdidos (ByteTrack):", self._max_lost)
        form.addRow("Ruta de exportación:", export_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _choose_directory(self) -> None:
        """Permite elegir la carpeta donde se guardará ``detecciones.csv``."""
        directory = QFileDialog.getExistingDirectory(
            self, "Carpeta de exportación", self._export_path.text()
        )
        if directory:
            self._export_path.setText(directory)

    def _on_accept(self) -> None:
        """Valida los campos antes de cerrar con ``Accepted``."""
        transport = self._transport.text().strip().lower()
        if transport not in {"udp", "tcp"}:
            transport = "udp"
        resolution = self._resolution.text().strip().lower().replace("×", "x")
        if resolution and resolution != "auto" and "x" not in resolution:
            self._resolution.setText("auto")
        self.accept()

    def updated_config(self, base: AppConfig) -> AppConfig:
        """Devuelve una copia de ``base`` con los valores del formulario."""
        base.use_gpu = self._use_gpu.isChecked()
        base.detection.model_path = self._model.text().strip() or "yolo11n.pt"
        base.detection.min_confidence = float(self._min_confidence.value())
        base.detection.image_size = int(self._image_size.value())
        base.detection.fp16 = self._fp16.isChecked()
        base.video.target_fps = int(self._target_fps.value())
        base.video.resolution = self._resolution.text().strip() or "auto"
        base.video.rtsp_transport = self._transport.text().strip().lower()
        base.tracking.max_lost = int(self._max_lost.value())
        base.export_path = str(Path(self._export_path.text().strip() or "exports"))
        return base
