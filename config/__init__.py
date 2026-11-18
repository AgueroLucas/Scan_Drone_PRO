"""Paquete de configuración de Drone People Counter Pro.

Expone una única API pública (:class:`AppConfig` y :class:`ConfigManager`) para
que el resto de módulos no dependan del formato físico del fichero ``config.json``.
"""

from .settings import (
    AppConfig,
    ConfigManager,
    DetectionSettings,
    TrackingSettings,
    VideoSettings,
    load_config,
    save_config,
)

__all__ = [
    "AppConfig",
    "ConfigManager",
    "DetectionSettings",
    "TrackingSettings",
    "VideoSettings",
    "load_config",
    "save_config",
]
