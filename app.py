"""Punto de entrada de Drone People Counter Pro.

Uso:
    python app.py

La aplicación detecta hardware, configura logs y muestra la interfaz PyQt6.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from logs.logging_setup import configure_logging, log_error, log_event


def main() -> int:
    """Inicializa la aplicación y entra en el bucle de eventos de Qt."""
    configure_logging(console=False)
    log_event("=== Inicio de Drone People Counter Pro ===")

    try:
        # Evita que Windows escalone de forma distinta los monitores HiDPI.
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:  # pragma: no cover - versiones antiguas de PyQt6
        pass

    application = QApplication(sys.argv)
    application.setApplicationName("Drone People Counter Pro")
    application.setApplicationVersion("1.0.0")
    application.setOrganizationName("Scan_Dron_PRO")

    window = MainWindow()
    window.show()

    try:
        return int(application.exec())
    except Exception as exc:  # pragma: no cover - captura global de última línea
        log_error("Error fatal en el bucle de la aplicación: %s", exc, exc_info=True)
        return 1
    finally:
        log_event("=== Fin de Drone People Counter Pro ===")


if __name__ == "__main__":
    raise SystemExit(main())
