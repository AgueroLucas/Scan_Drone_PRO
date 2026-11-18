"""Wrapper de Ultralytics YOLO11/YOLOv8 especializado en la clase ``Person``.

El detector entrega también detecciones de baja confianza porque ByteTrack las
necesita en su segunda fase de asociación. La decisión de qué detecciones crean
un ID nuevo se toma en el tracker, no aquí (separación de responsabilidades).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config.settings import RESOURCE_ROOT, USER_ROOT, DetectionSettings
from core.types import Detection
from detection.hardware import DeviceInfo

_LOGGER = logging.getLogger("dpc.detection")

COCO_PERSON_CLASS_ID = 0  # COCO: 0 = person


class ModelLoadError(RuntimeError):
    """Error fatal al cargar los pesos del modelo."""


class PersonDetector:
    """Detecta exclusivamente personas usando YOLO11 (o YOLOv8 como fallback)."""

    def __init__(self, settings: DetectionSettings, device: DeviceInfo) -> None:
        self._settings = settings
        self._device = device
        self._lock = threading.Lock()
        self._low_confidence = max(
            0.01,
            float(settings.min_confidence) * float(settings.low_score_factor),
        )
        # FP16 sólo en CUDA; en CPU se fuerza FP32 por compatibilidad.
        self._half = bool(settings.fp16 and device.using_gpu)
        self._model = self._load_model()
        self._warm_up()

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------
    def _resolve_local_path(self, configured: str) -> Path | None:
        """Resuelve una ruta local (absoluta o dentro de ``models/``)."""
        candidate = Path(configured).expanduser()
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        # En el exe, primero se busca junto al ejecutable (modelos del usuario)
        # y después dentro del bundle (modelos incluidos en la distribución).
        for root in (USER_ROOT, RESOURCE_ROOT):
            inside_models = root / "models" / candidate
            if inside_models.is_file():
                return inside_models
            local = root / candidate
            if local.is_file():
                return local
        return None

    def _load_model(self) -> Any:
        """Carga YOLO desde ``models/`` o desde el hub de Ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependencia obligatoria
            raise ModelLoadError(
                "Ultralytics no está instalado. Ejecuta: pip install -r requirements.txt"
            ) from exc

        local_path = self._resolve_local_path(self._settings.model_path)
        source = str(local_path) if local_path else self._settings.model_path
        _LOGGER.info("Cargando modelo YOLO: %s en %s", source, self._device.description())

        try:
            model = YOLO(source)
        except Exception as exc:
            # Si el usuario configuró un modelo que no existe, se intenta el
            # fallback oficial para que la aplicación siga siendo utilizable.
            fallback = "yolov8n.pt"
            if source != fallback:
                _LOGGER.error("No se pudo cargar %s: %s. Probando %s", source, exc, fallback)
                try:
                    model = YOLO(fallback)
                except Exception as second_exc:  # pragma: no cover - red/drivers
                    raise ModelLoadError(
                        f"No se pudo cargar ningún modelo YOLO. "
                        f"Detalle: {exc} | fallback: {second_exc}"
                    ) from second_exc
            else:
                raise ModelLoadError(f"No se pudo cargar el modelo {source}: {exc}") from exc

        try:
            model.to(self._device.device)
        except Exception as exc:  # pragma: no cover - GPU sin FP16, etc.
            _LOGGER.warning("No se pudo mover el modelo a %s: %s. Se usa CPU.", self._device.device, exc)
            self._half = False
            model.to("cpu")
        return model

    def _warm_up(self) -> None:
        """Ejecuta una inferencia mínima para fijar kernels CUDA y evitar el primer frame lento."""
        try:
            dummy = np.zeros((96, 96, 3), dtype=np.uint8)
            self._model.predict(
                source=dummy,
                imgsz=96,
                conf=self._settings.min_confidence,
                iou=self._settings.iou,
                classes=[COCO_PERSON_CLASS_ID],
                device=self._device.device,
                half=False,  # Warm-up en FP32: algunos motores CUDA no soportan FP16 vacío.
                verbose=False,
            )
            _LOGGER.info("Modelo YOLO inicializado y calentado correctamente.")
        except Exception as exc:  # pragma: no cover - no bloquea el arranque
            _LOGGER.warning("El warm-up del modelo falló (se continúa): %s", exc)

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Devuelve todas las detecciones de persona (alta y baja confianza)."""
        if frame is None or frame.size == 0:
            return []
        results: Sequence[Any]
        with self._lock:
            results = self._model.predict(
                source=frame,
                imgsz=int(self._settings.image_size),
                conf=self._low_confidence,
                iou=float(self._settings.iou),
                classes=[COCO_PERSON_CLASS_ID],
                device=self._device.device,
                half=self._half,
                max_det=int(self._settings.max_detections),
                verbose=False,
            )

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xywh = boxes.xywh
            confidences = boxes.conf
            if hasattr(xywh, "cpu"):
                xywh = xywh.cpu().numpy()
                confidences = confidences.cpu().numpy()
            for box, confidence in zip(np.asarray(xywh), np.asarray(confidences)):
                detections.append(
                    Detection(
                        tlwh=np.asarray(box, dtype=np.float64)[:4],
                        confidence=float(confidence),
                        class_id=COCO_PERSON_CLASS_ID,
                    )
                )
        return detections

    @property
    def low_confidence_threshold(self) -> float:
        """Umbral mínimo real usado en la inferencia."""
        return self._low_confidence
