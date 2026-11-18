# Drone People Counter Pro

Aplicación de escritorio para **Windows 10/11** que detecta, rastrea y cuenta personas en vídeos grabados o transmisiones en vivo de drones DJI (Air 2S, Avata 1, FPV), cámaras USB y capturadoras HDMI.

- **YOLO11** (YOLOv8 como fallback) filtrando únicamente la clase `person`.
- **ByteTrack** propio (Kalman + asociación en dos etapas) para IDs estables.
- **Contador único por sesión**: `Personas actuales` y `Total únicas`.
- **Exportación CSV** automática y manual, compatible con Excel (`utf-8-sig`, `;`).
- **GPU NVIDIA CUDA** con degradación automática a CPU.
- Interfaz **PyQt6** con indicadores 🟢 / 🟡 / 🔴 y logs en `logs/`.

---

## 1. Requisitos previos

| Componente | Versión recomendada |
|------------|---------------------|
| Windows    | 10 o 11 (64 bits)   |
| Python     | 3.10 – 3.12 (64 bits) |
| GPU (opcional) | NVIDIA con driver actual y CUDA habilitado |
| RAM        | 8 GB mínimo, 16 GB recomendado |

> Sin GPU la aplicación funciona correctamente en CPU; sólo disminuye el FPS.

---

## 2. Instalación

Abrir **PowerShell** en la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 2.1 Instalación CUDA (opcional)

Comprobar soporte tras la instalación:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Si devuelve `False`, instalar el driver NVIDIA y ejecutar:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

La GUI muestra **Modo GPU** o **Modo CPU** automáticamente al arrancar.

---

## 3. Ejecución

### 3.1 Ejecutable .exe (recomendado)

Ya se ha generado el ejecutable en:

```text
dist\DronePeopleCounterPro\DronePeopleCounterPro.exe
```

Haz doble clic en ese archivo para abrir la aplicación. **No copies sólo el `.exe` a otra carpeta**: debe estar siempre dentro de `dist\DronePeopleCounterPro\`, porque ahí está `_internal\` con Python, PyQt6, OpenCV, PyTorch y el modelo `yolo11n.pt`.

La primera vez, junto al `.exe` se crean automáticamente:

```text
config.json      <- editable con el botón ⚙ Configuración
logs\            <- eventos.log / errores.log
exports\         <- detecciones.csv
```

Para crear un acceso directo: clic derecho sobre `DronePeopleCounterPro.exe` → **Enviar a → Escritorio (crear acceso directo)**.

Si modificas el código, vuelve a generarlo con:

```powershell
pip install -r requirements-build.txt
.\build_exe.bat
```

> Se usa `--onedir` en lugar de `--onefile` porque el conjunto PyTorch + Ultralytics + OpenCV ocupa ~1,1 GB. Con `onefile` habría que descomprimir todo en cada arranque y la apertura sería muy lenta.

### 3.2 Desde código fuente

```powershell
python app.py
```

Primera ejecución:

1. Se crea `config.json` con valores por defecto (si no existe).
2. Se descarga `yolo11n.pt` desde Ultralytics (si no está en `models/`).
3. Se crean `logs/eventos.log`, `logs/errores.log` y la carpeta `exports/`.

---

## 4. Uso

### Modo 1 — Vídeo grabado

1. Pulsar **Abrir vídeo** y seleccionar `.mp4`, `.mov` o `.avi`.
2. Pulsar **Iniciar análisis**.
3. Ver anotaciones: `Persona ID 17 92%`.
4. **Pausar / Reanudar / Detener** cuando se necesite.
5. El CSV se guarda automáticamente cada pocos segundos y al detener.

### Modo 2 — Tiempo real

Pulsar **Conectar stream** e introducir una de estas fuentes:

| Fuente | Ejemplo |
|--------|---------|
| RTSP DJI / NVR | `rtsp://usuario:password@192.168.0.10:554/stream1` |
| UDP | `udp://@0.0.0.0:5000` |
| Cámara USB | `0`, `1`, `2` ... |
| Capturadora HDMI | `hdmi:0` |

Para latencia mínima, `video.buffer_size=1` y `rtsp_transport=udp`.  
Si el stream RTSP se corta, cambiar a `tcp` en Configuración.

### Grabación con drones DJI

- **Air 2S / Avata 1**: use el stream RTSP del dispositivo o de una estación receptora en la red.
- **DJI FPV / FPV 2**: la salida HDMI de las gafas se captura con una capturadora USB (`hdmi:0`).

---

## 5. Configuración (`config.json`)

Editable desde **⚙ Configuración** o directamente en el fichero.

| Clave | Por defecto | Descripción |
|-------|-------------|-------------|
| `use_gpu` | `true` | Intenta CUDA; si falla usa CPU. |
| `detection.model_path` | `yolo11n.pt` | Modelo local o de Ultralytics. |
| `detection.min_confidence` | `0.5` | Confianza mínima para crear un ID. |
| `detection.low_score_factor` | `0.5` | Fracción baja que ByteTrack usa para recuperar tracks. |
| `detection.iou` | `0.45` | IoU del NMS. |
| `detection.image_size` | `640` | Resolución de entrada del modelo. |
| `detection.fp16` | `true` | FP16 sólo en GPU. |
| `tracking.match_threshold` | `0.8` | Coste IoU máximo en la 1.ª asociación. |
| `tracking.low_match_threshold` | `0.7` | Coste IoU máximo en la 2.ª asociación. |
| `tracking.max_lost` | `30` | Frames que un ID se mantiene tras perder la persona. |
| `video.target_fps` | `30` | Cadencia máxima (`0` = sin límite). |
| `video.resolution` | `auto` | `auto` o `ANCHOxALTO`. |
| `video.buffer_size` | `1` | Buffer de OpenCV (latencia mínima). |
| `video.rtsp_transport` | `udp` | `udp` (rápido) o `tcp` (fiable). |
| `video.autosave_seconds` | `3.0` | Intervalo de guardado automático del CSV. |
| `export_path` | `exports` | Carpeta de salida. |
| `export_filename` | `detecciones.csv` | Nombre del fichero. |

---

## 6. Resultados y logs

### CSV

`exports/detecciones.csv` con columnas exactas:

```
ID;Fecha;Hora;Timestamp;Confianza
8;18/11/2026;14:33:21;00:05:42;0.9400
```

- `Confianza` es la **máxima** observada para ese ID durante la sesión.
- `Timestamp` es la posición del vídeo (`HH:MM:SS`) o el tiempo transcurrido en directo.
- Codificación `UTF-8 con BOM` y separador `;` => doble clic abre en Excel en español.

### Logs

```
logs/eventos.log   -> inicio, fin, exportaciones, cambios de configuración
logs/errores.log   -> errores y trazas
```

Rotación automática a 5 MB (3 copias).

---

## 7. Estructura del proyecto

```
project/
├── app.py                  # Punto de entrada
├── config.json             # Configuración editable
├── requirements.txt
├── requirements-build.txt  # PyInstaller (sólo para generar el .exe)
├── build_exe.bat           # Genera dist\DronePeopleCounterPro\...exe
├── DronePeopleCounterPro.spec
├── README.md
├── config/                 # Dataclasses y gestor de config.json
│   ├── __init__.py
│   └── settings.py
├── core/                   # Dominio: contador, eventos, tipos, render, FPS
│   ├── __init__.py
│   ├── counter.py
│   ├── events.py
│   ├── performance.py
│   ├── rendering.py
│   └── types.py
├── detection/              # YOLO + hardware
│   ├── __init__.py
│   ├── detector.py
│   └── hardware.py
├── tracking/               # ByteTrack (Kalman + Hungarian)
│   ├── __init__.py
│   └── byte_track.py
├── export/                 # Contrato de salida + CSV con pandas
│   ├── __init__.py
│   ├── csv_exporter.py
│   └── records.py
├── logs/                   # Logging rotativo
│   ├── __init__.py
│   └── logging_setup.py
├── pipeline/               # Fuentes y sesión de análisis
│   ├── __init__.py
│   ├── session.py
│   └── sources.py
├── gui/                    # PyQt6
│   ├── __init__.py
│   ├── main_window.py
│   ├── settings_dialog.py
│   ├── style.py
│   └── worker.py
├── models/                 # Pesos .pt opcionales
│   └── README.md
└── exports/                # CSV generado (se crea solo)
```

---

## 8. Arquitectura y ampliaciones futuras

La arquitectura aplica SOLID y **inyección de dependencias**:

- `FrameSource` (ABC): añadir una cámara nueva no toca el pipeline.
- `ResultExporter` (ABC): cambiar CSV por SQLite/PostgreSQL/PDF es sustituir el exportador.
- `EventBus`: cada sesión publica eventos; un agregador podrá combinar varias cámaras.
- `DetectionSession`: una instancia por fuente => **múltiples drones/cámaras** sin refactorizar.
- `TrackView` y `DetectionRecord`: contratos estables para zonas de conteo, entradas/salidas y mapas de calor.
- La GUI consume `ProcessedFrame` por señales => se puede añadir un **panel web** que consuma los mismos eventos.

Funcionalidades **no implementadas** pero preparadas: heatmaps, zonas, conteo direccional, panel web remoto, base de datos y PDF.

---

## 9. Solución de problemas

| Síntoma | Acción |
|---------|--------|
| `Modo CPU` inesperado | Actualizar driver NVIDIA; comprobar `torch.cuda.is_available()`. |
| `🔴 Sin señal` con RTSP | Probar `tcp`, verificar URL y que el stream no esté en uso. |
| ID cambia mucho | Aumentar `tracking.max_lost` o bajar `video.target_fps`. |
| Pocos FPS en CPU | `image_size=416`, `yolo11n.pt`, `fp16=false`. |
| Excel muestra raros los acentos | El CSV ya incluye BOM; abrir con doble clic, no "Importar datos". |
| `No module named ultralytics` | Activar el venv y reinstalar `requirements.txt`. |

---

## 10. Licencia y notas

Proyecto de uso local. Los pesos YOLO se distribuyen según la licencia de Ultralytics (AGPL-3.0 para el ecosistema Ultralytics). El contenido de `models/` no incluye pesos por defecto.
