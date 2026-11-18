# -*- mode: python ; coding: utf-8 -*-
"""Especificación de PyInstaller para Drone People Counter Pro.

Genera una carpeta ``dist/DronePeopleCounterPro`` con el ejecutable y todas
sus dependencias. Se usa ``--onedir`` en vez de ``--onefile`` porque PyTorch,
Ultralytics y OpenCV son muy grandes: así el arranque es inmediato y no hay
que descomprimir ~1 GB en cada ejecución.
"""

import glob
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

project_root = os.path.abspath(os.getcwd())

# ---------------------------------------------------------------------------
# Datos que deben viajar dentro del ejecutable
# ---------------------------------------------------------------------------
datas = [
    (os.path.join(project_root, "config.json"), "."),
    (os.path.join(project_root, "README.md"), "."),
]

# Pesos locales (si el usuario ha copiado modelos en models/).
for model in glob.glob(os.path.join(project_root, "models", "*.pt")):
    datas.append((model, "models"))

# YAML de configuración de Ultralytics (default.yaml, etc.).
try:
    datas += collect_data_files("ultralytics", include_py_files=False)
except Exception:  # pragma: no cover - defensivo
    pass

# ---------------------------------------------------------------------------
# Imports ocultos (Ultralytics carga modelos dinámicamente)
# ---------------------------------------------------------------------------
hiddenimports = [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtNetwork",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "torchvision",
    "ultralytics",
    "ultralytics.cfg",
    "ultralytics.data",
    "ultralytics.engine",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.models.yolo.detect.predict",
    "ultralytics.nn",
    "ultralytics.nn.tasks",
    "ultralytics.utils",
    "ultralytics.utils.plotting",
]

try:
    hiddenimports += collect_submodules("ultralytics")
except Exception:  # pragma: no cover - defensivo
    pass
# Se eliminan duplicados manteniendo el orden.
hiddenimports = list(dict.fromkeys(hiddenimports))

analysis = Analysis(
    ["app.py"],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="DronePeopleCounterPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,               # Sin ventana de consola al abrir el .exe.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DronePeopleCounterPro",
)
