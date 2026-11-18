"""Subsistema de logging de Drone People Counter Pro."""

from .logging_setup import (
    ERROR_LOG_NAME,
    EVENT_LOG_NAME,
    EventLogger,
    configure_logging,
    get_logger,
    log_error,
    log_event,
)

__all__ = [
    "ERROR_LOG_NAME",
    "EVENT_LOG_NAME",
    "EventLogger",
    "configure_logging",
    "get_logger",
    "log_error",
    "log_event",
]
