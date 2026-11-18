"""Configuración centralizada de logs.

Genera dos ficheros dentro de ``logs/``:

    * ``eventos.log`` -> inicio, fin, exportaciones y eventos de negocio.
    * ``errores.log`` -> errores y trazas completas.

Se usa :class:`logging.handlers.RotatingFileHandler` para que el disco no crezca
sin límite durante sesiones largas de grabación con drones.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from config.settings import USER_ROOT

EVENT_LOG_NAME = "eventos.log"
ERROR_LOG_NAME = "errores.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB por fichero
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
_DATE_FORMAT = "%d/%m/%Y %H:%M:%S"

_configured = False


def configure_logging(base_dir: Path | str | None = None, console: bool = False) -> Path:
    """Configura los handlers globales y devuelve el directorio de logs.

    La función es idempotente: llamarla dos veces no duplica líneas en los logs.
    """
    global _configured
    directory = Path(base_dir) if base_dir is not None else USER_ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)

    if _configured:
        return directory

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    event_handler = RotatingFileHandler(
        directory / EVENT_LOG_NAME,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    event_handler.setLevel(logging.INFO)
    event_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        directory / ERROR_LOG_NAME,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    # Los errores también aparecen en eventos.log para tener la cronología completa.
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root.addHandler(event_handler)
    root.addHandler(error_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    _configured = True
    logging.getLogger("dpc.events").info("Subsistema de logging inicializado en %s", directory)
    return directory


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger con el prefijo estándar de la aplicación."""
    return logging.getLogger(f"dpc.{name}")


def log_event(message: str, *args: Any) -> None:
    """Registra un evento de negocio (inicio, fin, exportación, etc.)."""
    get_logger("events").info(message, *args)


def log_error(message: str, *args: Any, exc_info: bool = False) -> None:
    """Registra un error. ``exc_info=True`` incluye la traza completa."""
    get_logger("errors").error(message, *args, exc_info=exc_info)


class EventLogger:
    """Suscriptor del :class:`~core.events.EventBus` que vuelca eventos a disco.

    Es un callable simple para que el bus no dependa de Qt ni de logging.
    """

    def __call__(self, event: str, payload: Any = None) -> None:
        if payload is None:
            log_event("EVENTO %s", event)
        elif isinstance(payload, dict):
            details = ", ".join(f"{key}={value}" for key, value in payload.items())
            log_event("EVENTO %s | %s", event, details)
        else:
            log_event("EVENTO %s | %s", event, payload)
