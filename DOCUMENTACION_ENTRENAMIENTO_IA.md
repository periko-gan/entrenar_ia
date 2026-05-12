# Documentación técnica de `entrenamiento ia`

Este documento describe, de forma práctica y mantenible, qué hay dentro de `entrenamiento ia`, para qué sirve cada bloque de archivos, qué hace cada función de los scripts en `tools/`, y cómo interpretar/usar los resultados de entrenamiento, prueba (test) y validación.

> Alcance: solo `entrenamiento ia`.
> Nota: no se listan miles de imágenes una por una; se documenta su estructura por carpetas y su rol en el flujo.

## 1. Mapa del directorio

Raíz: `entrenamiento ia/`

- `.venv/`
  - Entorno virtual Python local del proyecto.
- `.pytest_cache/`
  - Caché de `pytest` (solo soporte de pruebas).
- `archive/`
  - Datos originales/históricos (estructura fuente del dataset).
- `archive.zip`
  - Copia comprimida del dataset fuente.
- `dataset/`
  - Dataset operativo para YOLOv8 (`images/`, `labels/`, `data.yaml`, reportes).
- `tools/`
  - Scripts CLI de conversión, entrenamiento, evaluación y pipeline.
- `tests/`
  - Pruebas de humo y unitarias de los scripts.
- `runs/`
  - Salidas de ejecución (train, eval_predict, pipeline).
- `README.md`
  - Guía de uso diaria (instalación y comandos).
- `requirements.txt`
  - Dependencias del proyecto (sin forzar `torch`, se instala según CPU/CUDA).
- `yolov8n.pt`, `yolo26n.pt`
  - Pesos base/preentrenados de partida.
- `best.pt`, `last.pt`
  - Checkpoints en raíz (si existen, suelen ser copias o resultados movidos manualmente).

## 2. Qué hay dentro de cada directorio relevante

## `dataset/`

- `dataset/images/train`, `dataset/images/val`, `dataset/images/test`
  - Imágenes por split.
- `dataset/labels/train`, `dataset/labels/val`, `dataset/labels/test`
  - Etiquetas YOLO (`.txt`) y, según flujo, CSV de origen (`_annotations.csv`).
- `dataset/data.yaml`
  - Manifiesto de dataset usado por Ultralytics.
- `dataset/conversion_report.json`
  - Resumen de calidad de conversión CSV -> YOLO.

## `tools/`

- `tools/csv_to_yolo.py`
  - Convierte anotaciones CSV a formato YOLO y genera `data.yaml`.
- `tools/device_resolver.py`
  - Resuelve `--device` (CPU/GPU) con fallback seguro.
- `tools/train_yolov8.py`
  - Entrena YOLOv8 sobre `data.yaml`.
- `tools/eval_predict_yolov8.py`
  - Ejecuta validación (`val`), predicción (`predict`) o ambos (`both`).
- `tools/run_train_eval_predict_yolov8.py`
  - Orquesta el pipeline completo train -> eval/predict.

## `tests/`

- `tests/test_csv_to_yolo_smoke.py`
  - Verifica conversión básica CSV -> etiquetas YOLO + `data.yaml` + reporte.
- `tests/test_train_yolov8_smoke.py`
  - Verifica `train_yolov8.py --dry-run`.
- `tests/test_eval_predict_yolov8_smoke.py`
  - Verifica `eval_predict_yolov8.py --dry-run`.
- `tests/test_eval_predict_yolov8_extract_metrics.py`
  - Valida normalización de métricas (`extract_metrics`).
- `tests/test_run_train_eval_predict_yolov8_smoke.py`
  - Verifica pipeline en `--dry-run` y creación de `pipeline_report.json`.

## `runs/`

- `runs/train/`
  - Salidas de entrenamiento: pesos (`best.pt`, `last.pt`) y gráficas.
- `runs/eval_predict/`
  - Salidas de validación/predicción: métricas, imágenes y `run_report.json`.
- `runs/pipeline/`
  - Reportes globales de ejecuciones end-to-end (`pipeline_report.json`).

## `archive/`

- `archive/train`, `archive/valid`, `archive/test`
  - Dataset fuente/histórico. Suele usarse como origen para conversión, no como salida de entrenamiento.

## 3. Detalle por archivo de `tools/` y funciones

## `tools/device_resolver.py`

Objetivo: transformar el `--device` solicitado en un valor utilizable por Ultralytics/Torch.

Funciones y estructuras:

- `DeviceResolution`
  - Dataclass inmutable con `requested`, `resolved` y `warning`.
- `_safe_torch_info()`
  - Intenta leer estado CUDA (`is_available`, `device_count`) sin romper el flujo si `torch` falla.
- `resolve_device(requested_device)`
  - Aplica reglas:
    - `auto` -> usa GPU `0` si existe; si no, `cpu`.
    - `cpu` -> fuerza CPU.
    - `0`, `0,1`, etc. -> valida índices; si no hay CUDA, cae a CPU con warning.

## `tools/csv_to_yolo.py`

Objetivo: pasar de anotaciones CSV a formato YOLOv8 y dejar el dataset listo para entrenar.

Funciones:

- `parse_args()`
  - Lee argumentos CLI (`--dataset-root`, `--csv-name`, `--output-labels-dir`, `--data-yaml`).
- `collect_classes(dataset_root, csv_name)`
  - Detecta clases únicas recorriendo los CSV de todos los splits.
- `list_images(images_dir)`
  - Lista imágenes válidas en disco por extensión.
- `row_to_box(row, class_to_id)`
  - Convierte una fila CSV en caja YOLO normalizada y valida coordenadas.
- `write_label_file(path, boxes)`
  - Escribe un `.txt` YOLO por imagen.
- `convert_split(dataset_root, split, csv_name, class_to_id, output_labels_root)`
  - Ejecuta la conversión completa de un split y acumula estadísticas.
- `write_data_yaml(dataset_root, data_yaml_name, class_names)`
  - Genera `data.yaml` con rutas y clases.
- `write_report(dataset_root, report)`
  - Guarda `conversion_report.json`.
- `main()`
  - Orquesta todo el flujo de conversión extremo a extremo.

## `tools/train_yolov8.py`

Objetivo: entrenar un detector YOLOv8 con parámetros CLI.

Funciones:

- `_resolve_project_dir(project_arg)`
  - Garantiza que rutas relativas de salida queden dentro de `entrenamiento ia`.
- `parse_args()`
  - Define parámetros de entrenamiento (`data`, `model`, `epochs`, `imgsz`, `batch`, `device`, etc.).
- `build_train_kwargs(args)`
  - Traduce argumentos CLI al formato esperado por `model.train()`.
- `main()`
  - Valida `data.yaml`.
  - Resuelve dispositivo con `resolve_device`.
  - Configura `ultralytics.settings.runs_dir` a `entrenamiento ia/runs`.
  - Ejecuta `YOLO(args.model).train(**train_kwargs)`.
  - Imprime ruta real de salida (`ultralytics_save_dir`).

## `tools/eval_predict_yolov8.py`

Objetivo: validar, predecir o hacer ambos procesos con reporte consistente.

Funciones:

- `_resolve_project_dir(project_arg)`
  - Mantiene salidas relativas dentro de `entrenamiento ia`.
- `parse_args()`
  - CLI para `--task val|predict|both` y parámetros de inferencia.
- `build_val_kwargs(args)`
  - Crea kwargs para `model.val()`.
- `build_predict_kwargs(args)`
  - Crea kwargs para `model.predict()`.
- `_normalize_metric_value(value)`
  - Convierte tensores/arrays/escalares a tipos JSON-safe.
- `extract_metrics(metrics_obj)`
  - Extrae métricas clave (`fitness`, `speed`, `box_map`, `box_map50`, etc.).
- `write_metrics_files(metrics, output_dir)`
  - Exporta métricas en JSON y CSV.
- `write_run_report(output_dir, args, val_metrics, prediction_count, ultralytics_save_dir)`
  - Escribe `run_report.json` con trazabilidad de ejecución.
- `_validate_inputs(args)`
  - Verifica rutas necesarias antes de ejecutar Ultralytics.
- `_predict_count(results)`
  - Cuenta elementos procesados en predicción.
- `_extract_save_dir(value)`
  - Recupera `save_dir` real desde objetos devueltos por Ultralytics.
- `main()`
  - Ejecuta el flujo según `--task` y persiste reportes.

## `tools/run_train_eval_predict_yolov8.py`

Objetivo: ejecutar train -> eval/predict en una sola corrida versionada por timestamp UTC.

Funciones:

- `_resolve_project_dir(project_arg)`
  - Normaliza rutas de proyecto (`train`, `eval`, `pipeline`).
- `parse_args()`
  - CLI global del pipeline.
- `utc_stamp()`
  - Genera sello temporal para nombres únicos de corrida.
- `_run_step(command, label)`
  - Ejecuta pasos hijos (`train`, `eval_predict`) y propaga errores.
- `_resolve_trained_model(train_project, train_name, fallback_model)`
  - Busca `best.pt`, luego `last.pt`, y si no existe usa el modelo de entrada.
- `main()`
  - Valida entradas.
  - Ejecuta entrenamiento (opcional con `--skip-train`).
  - Ejecuta evaluación/predicción.
  - Guarda `pipeline_report.json`.

## 4. Qué pasa cuando entrenas una IA aquí

Flujo simplificado:

1. Preparas datos en `dataset/` y confirmas `dataset/data.yaml`.
2. Ejecutas `tools/train_yolov8.py` con modelo base (`yolov8n.pt`, `yolo26n.pt` u otro).
3. El script decide dispositivo (`auto`, `cpu`, `0`, etc.).
4. Ultralytics entrena por épocas y guarda artefactos en `runs/train/<run>/`.
5. Obtienes checkpoints:
   - `weights/best.pt` (mejor resultado).
   - `weights/last.pt` (último estado).
6. El script imprime `ultralytics_save_dir` para ubicar la carpeta exacta.

## 5. Cómo se hace la prueba (predict)

En este proyecto, la prueba operativa suele ser inferencia sobre `dataset/images/test`.

Flujo:

1. Tomas un modelo (`best.pt` recomendado).
2. Ejecutas `tools/eval_predict_yolov8.py --task predict ...`.
3. Se guardan imágenes con cajas en `runs/eval_predict/<run>/`.
4. Se genera `run_report.json` con contexto de la corrida.

## 6. Cómo se valida (val)

Validar = medir rendimiento sobre un split etiquetado (normalmente `val`).

Flujo:

1. Ejecutas `tools/eval_predict_yolov8.py --task val ...`.
2. Ultralytics calcula precisión/recall/mAP.
3. El script exporta:
   - `val_metrics.json`
   - `val_metrics.csv`
   - `run_report.json`

## 7. Cómo usar los resultados del entrenamiento

## 7.1 Inferencia sobre datos nuevos

- Usa `runs/train/<run>/weights/best.pt` como modelo de predicción.
- Corre `predict` y revisa visualmente salidas en `runs/eval_predict/<run>/`.

## 7.2 Comparar experimentos

- Compara `val_metrics.json` y `val_metrics.csv` entre corridas.
- Usa `run_report.json` para reconstruir parámetros usados.
- Si usas pipeline completo, complementa con `runs/pipeline/<run>/pipeline_report.json`.

## 7.3 Continuar entrenamiento (fine-tuning)

- Reanuda desde `last.pt` si necesitas continuidad de una corrida con `--resume`.
- **Mejores prácticas para Finetuning (ej. Portátil 8GB VRAM con modelo preentrenado local `yolo26n.pt`):**
  - **Resolución y Batch:** Usa `--imgsz 800` y `--batch 8` para equilibrar detalle fino (caries) sin colapsar la VRAM.
  - **Protección del modelo base:** Usa `--freeze 10` para congelar las capas iniciales y evitar el olvido catastrófico.
  - **Optimizador y Learning Rate:** Usa `--optimizer AdamW`, `--lr0 0.001`, `--cos-lr` y `--warmup-epochs 3.0` para un ajuste fino y suave.
  - **Evitar sobreajuste:** Desactiva mosaic al final con `--close-mosaic 15`.
  - **Rendimiento térmico:** Limita `--workers 4` para no saturar la CPU en portátiles.

## 7.4 Preparar despliegue

- Conserva como mínimo:
  - checkpoint final (`best.pt`),
  - referencia de clases (`dataset/data.yaml`),
  - métricas (`val_metrics.json`),
  - metadatos de corrida (`run_report.json`).

## 8. Resumen rápido

- Conversión de datos: `tools/csv_to_yolo.py`
- Entrenamiento: `tools/train_yolov8.py`
- Validación/predicción: `tools/eval_predict_yolov8.py`
- Pipeline completo: `tools/run_train_eval_predict_yolov8.py`
- Pruebas: `tests/`
- Artefactos: `runs/`
