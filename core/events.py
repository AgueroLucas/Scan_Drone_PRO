"""Bus de eventos interno (observer pattern) desacoplado de Qt.

Cada sesión recibe su propia instancia. Esto es clave para la ampliación futura
de múltiples cámaras: cada cámara publica en su bus y un agregador opcional puede
suscribirse a todos sin que el pipeline conozca a la GUI.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

EventHandler = Callable[[str, Any], None]


class Events:
    """Nombres canónicos de los eventos publicados por el pipeline."""

    SESSION_STARTED = "session.started"
    SESSION_PAUSED = "session.paused"
    SESSION_RESUMED = "session.resumed"
    SESSION_STOPPED = "session.stopped"
    PERSON_DETECTED = "person.detected"
    EXPORT_COMPLETED = "export.completed"
    ERROR = "system.error"


class EventBus:
    """Bus de eventos thread-safe con suscriptores múltiples."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event: str, handler: EventHandler) -> None:
        """Registra un handler para un evento concreto (o ``*`` para todos)."""
        with self._lock:
            self._handlers.setdefault(event, []).append(handler)

    def unsubscribe(self, event: str, handler: EventHandler) -> None:
        """Elimina un handler previamente registrado."""
        with self._lock:
            handlers = self._handlers.get(event, [])
            if handler in handlers:
                handlers.remove(handler)

    def emit(self, event: str, payload: Any = None) -> None:
        """Notifica el evento a todos sus suscriptores.

        Un handler que falla no interrumpe el procesamiento de vídeo: el error
        se registra y se continúa.
        """
        with self._lock:
            targets = list(self._handlers.get(event, [])) + list(self._handlers.get("*", []))
        for handler in targets:
            try:
                handler(event, payload)
            except Exception as exc:  # pragma: no cover - defensivo
                import logging

                logging.getLogger("dpc.events").exception("Falló handler del evento %s: %s", event, exc)

    def clear(self) -> None:
        """Elimina todos los suscriptores."""
        with self._lock:
            self._handlers.clear()
