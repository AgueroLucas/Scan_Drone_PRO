"""Detección de personas y selección de hardware."""

from .detector import ModelLoadError, PersonDetector
from .hardware import DeviceInfo, detect_hardware

__all__ = ["DeviceInfo", "ModelLoadError", "PersonDetector", "detect_hardware"]
