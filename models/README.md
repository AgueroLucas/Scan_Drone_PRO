# models/

Esta carpeta guarda pesos locales opcionales (`.pt`).

Por defecto `config.json` usa `yolo11n.pt`. Ultralytics descarga el modelo la
primera vez que se ejecuta la aplicación y lo guarda en su caché de usuario.

Si se quiere trabajar **sin conexión**, copie aquí el peso deseado, por ejemplo:

```
models/yolo11n.pt
```

Y deje `detection.model_path` como `yolo11n.pt` (el resolver busca primero en
esta carpeta) o indique la ruta absoluta.

Modelos probados:

| Modelo      | Exactitud | Velocidad | Recomendado para |
|-------------|-----------|-----------|------------------|
| yolo11n.pt  | Media     | Muy alta  | CPU / drones     |
| yolo11s.pt  | Alta      | Alta      | GPU NVIDIA       |
| yolov8n.pt  | Media     | Muy alta  | Fallback         |

La aplicación detecta la clase `person` (ID 0 de COCO) y descarta el resto.
