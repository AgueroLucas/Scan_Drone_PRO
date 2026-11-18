"""Detección automática de hardware al inicio de la aplicación.

Si el usuario pide GPU pero PyTorch no ve CUDA (drivers ausentes, tarjeta no
compatible o ``use_gpu=false``), la aplicación degrada a CPU sin detenerse y lo
informa en la interfaz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

_LOGGER = logging.getLogger("dpc.hardware")


@dataclass(frozen=True)
class DeviceInfo:
    """Resumen del dispositivo de inferencia seleccionado."""

    device: str
    cuda_available: bool
    gpu_name: str | None
    torch_version: str
    requested_gpu: bool

    @property
    def using_gpu(self) -> bool:
        """``True`` cuando la inferencia se ejecutará en NVIDIA CUDA."""
        return self.device.startswith("cuda")

    @property
    def mode_label(self) -> str:
        """Texto exacto mostrado en la GUI: ``Modo GPU`` / ``Modo CPU``."""
        return "Modo GPU" if self.using_gpu else "Modo CPU"

    def description(self) -> str:
        """Descripción ampliada para el log y el panel de estado."""
        if self.using_gpu:
            return f"Modo GPU | {self.gpu_name or 'CUDA'} | torch {self.torch_version}"
        reason = "desactivado en config.json" if not self.requested_gpu else "CUDA no disponible"
        return f"Modo CPU | {reason} | torch {self.torch_version}"


def detect_hardware(request_gpu: bool = True) -> DeviceInfo:
    """Inspecciona PyTorch y devuelve el dispositivo a usar."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependencia obligatoria
        raise RuntimeError(
            "PyTorch no está instalado. Instala requirements.txt (ultralytics lo incluye)."
        ) from exc

    cuda_available = bool(torch.cuda.is_available())
    gpu_name: str | None = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - defensivo
            _LOGGER.warning("No se pudo consultar el nombre de la GPU: %s", exc)

    if request_gpu and cuda_available:
        device = "cuda:0"
    else:
        device = "cpu"
        if request_gpu and not cuda_available:
            _LOGGER.warning("CUDA no disponible; se usa CPU automáticamente.")

    info = DeviceInfo(
        device=device,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        torch_version=str(getattr(torch, "__version__", "desconocida")),
        requested_gpu=bool(request_gpu),
    )
    _LOGGER.info("Hardware detectado: %s", info.description())
    return info
