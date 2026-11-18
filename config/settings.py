"""Gestión typed de la configuración de la aplicación.

Responsabilidades:
    * Definir la estructura de ``config.json`` mediante dataclasses.
    * Validar y acotar todos los valores leídos del disco.
    * Crear el fichero por defecto si no existe (primera ejecución).

El resto de la aplicación consume :class:`AppConfig`, nunca diccionarios crudos.
Esto permite añadir nuevas secciones sin romper los módulos existentes (SOLID:
principio de abierto/cerrado).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------
# En modo desarrollo ambas raíces coinciden con la carpeta del proyecto.
# Al empaquetar con PyInstaller:
#   * RESOURCE_ROOT = carpeta interna donde se extraen datos (config/modelos).
#   * USER_ROOT     = carpeta donde está el .exe (config editable, logs, CSV).
IS_FROZEN: bool = bool(getattr(sys, "frozen", False))
RESOURCE_ROOT: Path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
USER_ROOT: Path = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parents[1]
# Alias conservado por compatibilidad con módulos existentes.
PROJECT_ROOT: Path = USER_ROOT
DEFAULT_CONFIG_PATH: Path = USER_ROOT / "config.json"
_LOGGER = logging.getLogger("dpc.config")


# ---------------------------------------------------------------------------
# Funciones de conversión segura
# ---------------------------------------------------------------------------
def _as_float(value: Any, default: float, low: float, high: float) -> float:
    """Convierte a float y lo limita al rango [low, high]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(low, min(high, number))


def _as_int(value: Any, default: int, low: int, high: int) -> int:
    """Convierte a int y lo limita al rango [low, high]."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _as_bool(value: Any, default: bool) -> bool:
    """Acepta booleanos reales y representaciones textuales habituales."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "si", "sí", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _as_str(value: Any, default: str) -> str:
    """Devuelve un string limpio o el valor por defecto."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Devuelve una sección del JSON; si no es un objeto, devuelve {}."""
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Estructuras de configuración
# ---------------------------------------------------------------------------
@dataclass
class DetectionSettings:
    """Parámetros del modelo YOLO11/YOLOv8."""

    # Nombre del modelo (Ultralytics lo descarga) o ruta a un *.pt local.
    model_path: str = "yolo11n.pt"
    # Confianza mínima para considerar una detección "alta" y crear tracks.
    min_confidence: float = 0.50
    # ByteTrack necesita detecciones de baja confianza para recuperar identities:
    # el detector entrega desde min_confidence * low_score_factor.
    low_score_factor: float = 0.50
    # IoU usado por el post-proceso NMS de Ultralytics.
    iou: float = 0.45
    # Tamaño de entrada de la red (múltiplo de 32 recomendado).
    image_size: int = 640
    # Precisión FP16 sólo tiene sentido en GPU NVIDIA compatible.
    fp16: bool = True
    # Número máximo de cajas por imagen.
    max_detections: int = 300

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DetectionSettings":
        defaults = cls()
        return cls(
            model_path=_as_str(data.get("model_path"), defaults.model_path),
            min_confidence=_as_float(data.get("min_confidence"), defaults.min_confidence, 0.01, 0.99),
            low_score_factor=_as_float(data.get("low_score_factor"), defaults.low_score_factor, 0.05, 1.0),
            iou=_as_float(data.get("iou"), defaults.iou, 0.01, 0.99),
            image_size=_as_int(data.get("image_size"), defaults.image_size, 160, 2560),
            fp16=_as_bool(data.get("fp16"), defaults.fp16),
            max_detections=_as_int(data.get("max_detections"), defaults.max_detections, 1, 2000),
        )


@dataclass
class TrackingSettings:
    """Parámetros del algoritmo ByteTrack."""

    # Umbral IoU-cost para la primera asociación (detecciones de alta confianza).
    match_threshold: float = 0.80
    # Umbral IoU-cost para la segunda asociación (detecciones de baja confianza).
    low_match_threshold: float = 0.70
    # Frames que un track puede permanecer "perdido" antes de borrarse.
    max_lost: int = 30

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrackingSettings":
        defaults = cls()
        return cls(
            match_threshold=_as_float(data.get("match_threshold"), defaults.match_threshold, 0.1, 0.99),
            low_match_threshold=_as_float(data.get("low_match_threshold"), defaults.low_match_threshold, 0.1, 0.99),
            max_lost=_as_int(data.get("max_lost"), defaults.max_lost, 1, 300),
        )


@dataclass
class VideoSettings:
    """Parámetros de captura y rendimiento."""

    # Cadencia máxima de procesamiento (0 = sin límite).
    target_fps: int = 30
    # "auto" conserva la resolución de la fuente o "ANCHOxALTO" (ej. 1280x720).
    resolution: str = "auto"
    # Buffer de OpenCV. 1 = latencia mínima en directo.
    buffer_size: int = 1
    # Transporte para fuentes RTSP: "udp" (menos latencia) o "tcp" (más estable).
    rtsp_transport: str = "udp"
    # Intervalo de guardado automático del CSV mientras la sesión está activa.
    autosave_seconds: float = 3.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VideoSettings":
        defaults = cls()
        transport = _as_str(data.get("rtsp_transport"), defaults.rtsp_transport).lower()
        if transport not in {"udp", "tcp"}:
            transport = defaults.rtsp_transport
        return cls(
            target_fps=_as_int(data.get("target_fps"), defaults.target_fps, 0, 240),
            resolution=_as_str(data.get("resolution"), defaults.resolution),
            buffer_size=_as_int(data.get("buffer_size"), defaults.buffer_size, 1, 30),
            rtsp_transport=transport,
            autosave_seconds=_as_float(data.get("autosave_seconds"), defaults.autosave_seconds, 0.5, 60.0),
        )

    def resolution_tuple(self) -> tuple[int, int] | None:
        """Convierte ``1280x720`` en ``(1280, 720)``; ``None`` para ``auto``."""
        text = self.resolution.strip().lower().replace("×", "x")
        if text in {"", "auto", "original", "nativo"}:
            return None
        parts = text.split("x")
        if len(parts) != 2:
            return None
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        if width < 64 or height < 64:
            return None
        return width, height


@dataclass
class AppConfig:
    """Configuración raíz de la aplicación."""

    use_gpu: bool = True
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    # Directorio destino de ``detecciones.csv`` (relativo a la raíz del proyecto).
    export_path: str = "exports"
    # Nombre del fichero de resultados.
    export_filename: str = "detecciones.csv"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        """Construye una configuración válida a partir de un diccionario JSON."""
        defaults = cls()
        return cls(
            use_gpu=_as_bool(data.get("use_gpu"), defaults.use_gpu),
            detection=DetectionSettings.from_mapping(_section(data, "detection")),
            tracking=TrackingSettings.from_mapping(_section(data, "tracking")),
            video=VideoSettings.from_mapping(_section(data, "video")),
            export_path=_as_str(data.get("export_path"), defaults.export_path),
            export_filename=_as_str(data.get("export_filename"), defaults.export_filename),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializa la configuración a un diccionario JSON-compatible."""
        return asdict(self)

    def export_directory(self) -> Path:
        """Ruta absoluta de exportación (relativa a la raíz si es relativa)."""
        candidate = Path(self.export_path).expanduser()
        return candidate if candidate.is_absolute() else USER_ROOT / candidate


# ---------------------------------------------------------------------------
# Gestor de ficheros
# ---------------------------------------------------------------------------
class ConfigManager:
    """Carga, valida y guarda ``config.json``.

    Se encarga también de crear el fichero con valores por defecto la primera
    vez que se ejecuta la aplicación.
    """

    def __init__(self, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self.path = Path(path)

    def load(self) -> AppConfig:
        """Lee la configuración; ante cualquier error usa y persiste los defaults."""
        if not self.path.exists():
            defaults = AppConfig()
            # En el ejecutable, el config.json original viaja dentro del bundle;
            # se copia junto al .exe para que el usuario pueda editarlo.
            bundled = RESOURCE_ROOT / "config.json"
            if IS_FROZEN and bundled.is_file():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bundled, self.path)
                _LOGGER.info("Copiado config.json desde el bundle a %s", self.path)
                return AppConfig.from_mapping(json.loads(self.path.read_text(encoding="utf-8")))
            self.save(defaults)
            _LOGGER.info("Creado config.json con valores por defecto en %s", self.path)
            return defaults
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError("La raíz de config.json debe ser un objeto JSON")
            return AppConfig.from_mapping(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _LOGGER.error("No se pudo leer %s: %s. Se usan valores por defecto.", self.path, exc)
            defaults = AppConfig()
            self.save(defaults)
            return defaults

    def save(self, config: AppConfig) -> None:
        """Escribe la configuración de forma atómica."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(config.to_dict(), stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            temporary.replace(self.path)
        except OSError as exc:
            _LOGGER.error("No se pudo guardar %s: %s", self.path, exc)
            raise


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Atajo funcional para cargar la configuración."""
    return ConfigManager(path).load()


def save_config(config: AppConfig, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    """Atajo funcional para guardar la configuración."""
    ConfigManager(path).save(config)
