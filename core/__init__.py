"""Núcleo de dominio de Drone People Counter Pro.

Contiene tipos inmutables, contador de personas, bus de eventos, renderizado y
métricas. Los módulos se exponen de forma perezosa para que importar, por
ejemplo, ``core.types`` no obligue a tener OpenCV instalado.
"""

from importlib import import_module

__all__ = [
    "Detection",
    "EventBus",
    "Events",
    "FpsMeter",
    "FrameRenderer",
    "PeopleCounter",
    "ProcessedFrame",
    "SessionState",
    "SessionStats",
    "TrackView",
]

_MODULE_BY_NAME = {
    "Detection": "types",
    "EventBus": "events",
    "Events": "events",
    "FpsMeter": "performance",
    "FrameRenderer": "rendering",
    "PeopleCounter": "counter",
    "ProcessedFrame": "types",
    "SessionState": "types",
    "SessionStats": "types",
    "TrackView": "types",
}


def __getattr__(name: str):
    """Carga bajo demanda el símbolo solicitado."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"módulo {__name__!r} no tiene el atributo {name!r}")
    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)
