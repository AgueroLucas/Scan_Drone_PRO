"""Modelo de dato de una persona detectada.

Una instancia por track-id por sesión: aunque la persona se pierda y vuelva a
aparecer con el mismo ID, se actualiza el mismo registro (contador único).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class DetectionRecord:
    """Registro persistente exportado a ``detecciones.csv``."""

    track_id: int
    fecha: str
    hora: str
    timestamp: str
    confidence: float
    first_seen: float
    last_seen: float

    @property
    def confidence_text(self) -> str:
        """Confianza con 4 decimales (ej. ``0.9400``)."""
        return f"{float(self.confidence):.4f}"

    def to_row(self) -> dict[str, Any]:
        """Fila ordenada según las columnas exigidas en el CSV."""
        return {
            "ID": self.track_id,
            "Fecha": self.fecha,
            "Hora": self.hora,
            "Timestamp": self.timestamp,
            "Confianza": self.confidence_text,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DetectionRecord":
        """Reconstruye un registro (útil para tests o importaciones futuras)."""
        return cls(
            track_id=int(data["ID"]),
            fecha=str(data["Fecha"]),
            hora=str(data["Hora"]),
            timestamp=str(data["Timestamp"]),
            confidence=float(data["Confianza"]),
            first_seen=float(data.get("first_seen", 0.0)),
            last_seen=float(data.get("last_seen", 0.0)),
        )
