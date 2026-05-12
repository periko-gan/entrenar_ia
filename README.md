# Conversor CSV -> YOLO

Este proyecto convierte anotaciones en CSV dentro de `dataset/labels/{train,val,test}/_annotations.csv`
al formato YOLO (`.txt` por imagen) y crea `dataset/data.yaml`.

## Estructura esperada

- `dataset/images/train`, `dataset/images/val`, `dataset/images/test`
- `dataset/labels/train/_annotations.csv`, `dataset/labels/val/_annotations.csv`, `dataset/labels/test/_annotations.csv`

Columnas requeridas en CSV:

`filename,width,height,class,xmin,ymin,xmax,ymax`

## Instalar dependencias

Actualiza `pip` e instala dependencias del proyecto:

PowerShell (Windows):

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Bash (Linux):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt
```

## Instalar PyTorch

CPU (recomendado si no tienes NVIDIA CUDA):

PowerShell (Windows):

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Bash (Linux):

```bash
python3 -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

GPU NVIDIA (instalación limpia recomendada):

PowerShell (Windows):

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --upgrade pip
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Bash (Linux):

```bash
python3 -m pip uninstall -y torch torchvision torchaudio
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Alternativa si `cu130` no está disponible en tu entorno:

PowerShell (Windows):

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Bash (Linux):

```bash
python3 -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verificar detección de CUDA:

PowerShell (Windows):

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Bash (Linux):

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Verificación extendida (nombre de GPU y prueba de kernel CUDA):

PowerShell (Windows):

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('avail', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('name0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); x=torch.randn(512,512,device='cuda:0') if torch.cuda.is_available() else None; print('kernel_ok', x is not None)"
```

Bash (Linux):

```bash
python3 -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('avail', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('name0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); x=torch.randn(512,512,device='cuda:0') if torch.cuda.is_available() else None; print('kernel_ok', x is not None)"
```

Si aparece `CUDA error: no kernel image is available for execution on the device`, normalmente indica un wheel CUDA incompatible con tu GPU. Reinstala con `cu130` (o `cu128`) y verifica de nuevo.

## Ejecutar conversión

PowerShell (Windows):

```powershell
python .\tools\csv_to_yolo.py --dataset-root .\dataset
```

Bash (Linux):

```bash
python3 ./tools/csv_to_yolo.py --dataset-root ./dataset
```

Salida:

- `dataset/labels/{train,val,test}/*.txt`
- `dataset/data.yaml`
- `dataset/conversion_report.json`

Si quieres conservar etiquetas separadas, puedes usar:

PowerShell (Windows):

```powershell
python .\tools\csv_to_yolo.py --dataset-root .\dataset --output-labels-dir labels_yolo
```

Bash (Linux):

```bash
python3 ./tools/csv_to_yolo.py --dataset-root ./dataset --output-labels-dir labels_yolo
```

## Entrenar YOLOv8

Ejemplo recomendado (auto: usa GPU si existe; si no, CPU):

PowerShell (Windows):

```powershell
python .\tools\train_yolov8.py --data .\dataset\data.yaml --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device auto
```

Bash (Linux):

```bash
python3 ./tools/train_yolov8.py --data ./dataset/data.yaml --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device auto
```

Ejemplo base (CPU forzado):

PowerShell (Windows):

```powershell
python .\tools\train_yolov8.py --data .\dataset\data.yaml --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device cpu
```

Bash (Linux):

```bash
python3 ./tools/train_yolov8.py --data ./dataset/data.yaml --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device cpu
```

Ejemplo GPU (si tu equipo la detecta):

PowerShell (Windows):

```powershell
python .\tools\train_yolov8.py --data .\dataset\data.yaml --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

Bash (Linux):

```bash
python3 ./tools/train_yolov8.py --data ./dataset/data.yaml --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

Ejemplo Finetuning Avanzado (Portátil 8GB VRAM, modelo preentrenado local):

PowerShell (Windows):

```powershell
python .\tools\train_yolov8.py --data .\dataset\data.yaml --model yolo26n.pt --device 0 --epochs 150 --imgsz 800 --batch 8 --workers 4 --optimizer AdamW --lr0 0.001 --cos-lr --freeze 10 --close-mosaic 15 --warmup-epochs 3.0 --name "dental_yolo26n_laptop"
```

Bash (Linux):

```bash
python3 ./tools/train_yolov8.py --data ./dataset/data.yaml --model yolo26n.pt --device 0 --epochs 150 --imgsz 800 --batch 8 --workers 4 --optimizer AdamW --lr0 0.001 --cos-lr --freeze 10 --close-mosaic 15 --warmup-epochs 3.0 --name "dental_yolo26n_laptop"
```

Validar configuración sin entrenar:

PowerShell (Windows):

```powershell
python .\tools\train_yolov8.py --data .\dataset\data.yaml --dry-run
```

Bash (Linux):

```bash
python3 ./tools/train_yolov8.py --data ./dataset/data.yaml --dry-run
```

Dónde buscar el entrenamiento:

- Pesos del modelo entrenado: `runs/train/<nombre_run>/weights/best.pt` y `last.pt`
- Ejemplo real: `runs/train/dental_yolov8/weights/best.pt`
- Verificación en consola: al finalizar, `train_yolov8.py` imprime `ultralytics_save_dir` con la ruta final real.

## Evaluar y predecir YOLOv8

Evaluar + predecir en un solo comando:

PowerShell (Windows):

```powershell
python .\tools\eval_predict_yolov8.py --task both --data .\dataset\data.yaml --source .\dataset\images\test --model best.pt --device auto
```

Bash (Linux):

```bash
python3 ./tools/eval_predict_yolov8.py --task both --data ./dataset/data.yaml --source ./dataset/images/test --model best.pt --device auto
```

Solo evaluación:

PowerShell (Windows):

```powershell
python .\tools\eval_predict_yolov8.py --task val --data .\dataset\data.yaml --model best.pt --device auto
```

Bash (Linux):

```bash
python3 ./tools/eval_predict_yolov8.py --task val --data ./dataset/data.yaml --model best.pt --device auto
```

Solo predicción:

PowerShell (Windows):

```powershell
python .\tools\eval_predict_yolov8.py --task predict --data .\dataset\data.yaml --source .\dataset\images\test --model best.pt --device auto
```

Bash (Linux):

```bash
python3 ./tools/eval_predict_yolov8.py --task predict --data ./dataset/data.yaml --source ./dataset/images/test --model best.pt --device auto
```

Validar configuración sin ejecutar Ultralytics:

PowerShell (Windows):

```powershell
python .\tools\eval_predict_yolov8.py --task both --data .\dataset\data.yaml --source .\dataset\images\test --dry-run
```

Bash (Linux):

```bash
python3 ./tools/eval_predict_yolov8.py --task both --data ./dataset/data.yaml --source ./dataset/images/test --dry-run
```

Nota: `eval_predict_yolov8.py` ya normaliza métricas escalares y array-like de Ultralytics al exportar `val_metrics.json/csv`, evitando errores como `TypeError: only 0-dimensional arrays can be converted to Python scalars`.

Salida esperada en `runs/eval_predict/<name>`:

- `run_report.json`
- `val_metrics.json` y `val_metrics.csv` (si se ejecuta `val`)
- imágenes con predicciones guardadas por Ultralytics (si se ejecuta `predict`)

Dónde buscar test y evaluación:

- Test (predicción sobre `dataset/images/test`): imágenes resultantes en `runs/eval_predict/<nombre_run>/`
- Evaluación (`--task val`): métricas en `runs/eval_predict/<nombre_run>/val_metrics.json` y `val_metrics.csv`
- Reporte consolidado (val/predict): `runs/eval_predict/<nombre_run>/run_report.json`
- Verificación en consola: al finalizar, `eval_predict_yolov8.py` imprime `ultralytics_save_dir`.
- `run_report.json` también incluye el campo `ultralytics_save_dir`.

Nota de rutas: en estos scripts, todas las salidas se guardan dentro de `entrenamiento ia/runs`.

## Limpieza rápida (entrenamiento limpio)

Para empezar desde cero, vacía los directorios generados por entrenamiento, test (predicción) y evaluación.

Rutas que se limpian:

- `entrenamiento ia/runs/train`
- `entrenamiento ia/runs/eval_predict`
- `entrenamiento ia/runs/pipeline`

Ejecuta este comando desde `entrenamiento ia`:

PowerShell (Windows):

```powershell
$targets = @(
  (Join-Path (Get-Location) "runs\train"),
  (Join-Path (Get-Location) "runs\eval_predict"),
  (Join-Path (Get-Location) "runs\pipeline")
)
foreach ($path in $targets) {
  if (Test-Path $path) {
    Get-ChildItem -Path $path -Force | Remove-Item -Recurse -Force
  }
}
```

Bash (Linux):

```bash
for path in "./runs/train" "./runs/eval_predict" "./runs/pipeline"; do
  if [ -d "$path" ]; then
    find "$path" -mindepth 1 -delete
  fi
done
```

Este comando no toca `dataset/` ni archivos de código.

## Pipeline único (train -> eval/predict)

Ejecuta todo en secuencia con nombres versionados por timestamp UTC:

PowerShell (Windows):

```powershell
python .\tools\run_train_eval_predict_yolov8.py --data .\dataset\data.yaml --source .\dataset\images\test --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device auto --task both
```

Bash (Linux):

```bash
python3 ./tools/run_train_eval_predict_yolov8.py --data ./dataset/data.yaml --source ./dataset/images/test --model yolo26n.pt --epochs 100 --imgsz 640 --batch 16 --device auto --task both
```

Preset `GPU segura` (prioriza estabilidad, menor riesgo de OOM):

PowerShell (Windows):

```powershell
python .\tools\run_train_eval_predict_yolov8.py --data .\dataset\data.yaml --source .\dataset\images\test --model yolo26n.pt --epochs 120 --imgsz 640 --batch 4 --device 0 --task both
```

Bash (Linux):

```bash
python3 ./tools/run_train_eval_predict_yolov8.py --data ./dataset/data.yaml --source ./dataset/images/test --model yolo26n.pt --epochs 120 --imgsz 640 --batch 4 --device 0 --task both
```

Preset `GPU agresiva` (prioriza calidad/rendimiento, requiere más VRAM):

PowerShell (Windows):

```powershell
python .\tools\run_train_eval_predict_yolov8.py --data .\dataset\data.yaml --source .\dataset\images\test --model yolo26n.pt --epochs 200 --imgsz 896 --batch 8 --device 0 --task both
```

Bash (Linux):

```bash
python3 ./tools/run_train_eval_predict_yolov8.py --data ./dataset/data.yaml --source ./dataset/images/test --model yolo26n.pt --epochs 200 --imgsz 896 --batch 8 --device 0 --task both
```

Si aparece `CUDA out of memory`, baja primero `--batch` (por ejemplo, `8 -> 4 -> 2`) y luego `--imgsz` (`896 -> 768 -> 640`).

Modo validación rápida del pipeline (sin entrenar/evaluar realmente):

PowerShell (Windows):

```powershell
python .\tools\run_train_eval_predict_yolov8.py --data .\dataset\data.yaml --source .\dataset\images\test --dry-run
```

Bash (Linux):

```bash
python3 ./tools/run_train_eval_predict_yolov8.py --data ./dataset/data.yaml --source ./dataset/images/test --dry-run
```

Salida del pipeline:

- `runs/pipeline/<name_prefix>_<timestamp>_pipeline/pipeline_report.json`
- Entrenamiento en `runs/train/<name_prefix>_<timestamp>_train`
- Evaluación/predicción en `runs/eval_predict/<name_prefix>_<timestamp>_evalpredict`

Todas las rutas anteriores son relativas a `entrenamiento ia/`.

## Ejecutar pruebas rápidas

PowerShell (Windows):

```powershell
python -m pytest .\tests\test_csv_to_yolo_smoke.py
python -m pytest .\tests\test_train_yolov8_smoke.py
python -m pytest .\tests\test_eval_predict_yolov8_smoke.py
python -m pytest .\tests\test_eval_predict_yolov8_extract_metrics.py
python -m pytest .\tests\test_run_train_eval_predict_yolov8_smoke.py
```

Bash (Linux):

```bash
python3 -m pytest ./tests/test_csv_to_yolo_smoke.py
python3 -m pytest ./tests/test_train_yolov8_smoke.py
python3 -m pytest ./tests/test_eval_predict_yolov8_smoke.py
python3 -m pytest ./tests/test_eval_predict_yolov8_extract_metrics.py
python3 -m pytest ./tests/test_run_train_eval_predict_yolov8_smoke.py
```

