"""Pipeline de procesamiento (fuentes de vídeo + sesión de análisis)."""

from .session import DetectionSession, SessionResult
from .sources import FrameSource, SourceError, StreamSource, VideoFileSource

__all__ = [
    "DetectionSession",
    "FrameSource",
    "SessionResult",
    "SourceError",
    "StreamSource",
    "VideoFileSource",
]
