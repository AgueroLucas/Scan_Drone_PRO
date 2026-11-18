"""Hoja de estilos Qt (tema oscuro) de Drone People Counter Pro."""

DARK_STYLE = """
QWidget {
    background-color: #12161c;
    color: #e8edf2;
    font-family: 'Segoe UI', 'Segoe UI Emoji', sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: #0e1116;
}
QGroupBox {
    border: 1px solid #2c3643;
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #7fd1ff;
}
QPushButton {
    background-color: #1f78d1;
    border: none;
    border-radius: 6px;
    padding: 9px 12px;
    color: white;
    font-weight: 600;
    text-align: left;
}
QPushButton:hover { background-color: #2a8be8; }
QPushButton:pressed { background-color: #1766b5; }
QPushButton:disabled {
    background-color: #2a323c;
    color: #7c8794;
}
QPushButton#secondary {
    background-color: #2c3643;
}
QPushButton#secondary:hover { background-color: #3a4757; }
QPushButton#danger {
    background-color: #c0392b;
}
QPushButton#danger:hover { background-color: #d44a3b; }
QLabel#video {
    background-color: #05070a;
    border: 1px solid #2c3643;
    border-radius: 6px;
}
QLabel#metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #7fd1ff;
}
QLabel#metric-value-green {
    font-size: 28px;
    font-weight: 700;
    color: #4fd18b;
}
QLabel#metric-title {
    color: #9aa7b5;
    font-size: 12px;
}
QLabel#status-active { color: #4fd18b; font-weight: 700; font-size: 15px; }
QLabel#status-waiting { color: #f1c40f; font-weight: 700; font-size: 15px; }
QLabel#status-error { color: #e74c3c; font-weight: 700; font-size: 15px; }
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1b222b;
    border: 1px solid #34404e;
    border-radius: 4px;
    padding: 5px;
}
QStatusBar { color: #9aa7b5; }
QMenuBar { background-color: #0e1116; }
QMenuBar::item:selected { background-color: #2c3643; }
QMenu { background-color: #1b222b; border: 1px solid #34404e; }
QMenu::item:selected { background-color: #1f78d1; }
"""
