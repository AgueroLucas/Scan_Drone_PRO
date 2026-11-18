"""Interfaz gráfica de escritorio construida con PyQt6."""

from .main_window import MainWindow
from .worker import AnalysisWorker

__all__ = ["AnalysisWorker", "MainWindow"]
