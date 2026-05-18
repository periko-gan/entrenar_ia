#!/usr/bin/env python3
"""Entrenamiento YOLOv8 sobre un data.yaml existente, con soporte de finetuning."""

# Como leer este archivo:
# 1) `parse_args()` define flags de entrenamiento y finetuning.
# 2) `build_train_kwargs()` traduce flags al formato de Ultralytics.
# 3) `main()` valida rutas, imprime configuración y ejecuta `model.train()`.
#
# Ejemplos de uso:
# python .\tools\train_yolov8.py --data .\dataset\data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --device cpu
# python .\tools\train_yolov8.py --data .\dataset\data.yaml --model yolov8n.pt --epochs 30 --freeze 10 --lr0 0.001
# python .\tools\train_yolov8.py --data .\dataset\data.yaml --model .\runs\train\exp\weights\last.pt --resume

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict

from device_resolver import resolve_device

TOOLS_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_dir(project_arg: str) -> Path:
    # Fuerza rutas relativas a vivir dentro de `entrenamiento ia` para evitar salidas fuera.
    project_path = Path(project_arg)
    if project_path.is_absolute():
        return project_path
    return (TOOLS_ROOT / project_path).resolve()


def parse_args() -> argparse.Namespace:
    # Define todos los parámetros de entrada del entrenamiento.
    # La idea es poder ejecutar el script tanto en pruebas rápidas como en corridas largas.
    parser = argparse.ArgumentParser(
        description="Entrena un detector YOLOv8 con Ultralytics."
    )
    parser.add_argument("--data", default="dataset/data.yaml", help="Ruta al data.yaml")
    # parser.add_argument("--model", default="yolov8n.pt", help="Modelo base o checkpoint")
    parser.add_argument("--model", default="yolo26n.pt", help="Modelo base o checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Numero de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo: auto, cpu, 0, 0, 1, ... (auto usa GPU si existe)",
    )
    parser.add_argument("--project", default="runs/train", help="Carpeta base de resultados")
    parser.add_argument("--name", default="dental_yolov8", help="Nombre del experimento")
    parser.add_argument("--workers", type=int, default=4, help="Workers del dataloader")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")

    # Flags de finetuning
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Numero de capas iniciales a congelar para finetuning",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Learning rate inicial; util para finetuning fino",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=None,
        help="Learning rate final factor",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda entrenamiento desde el ultimo checkpoint",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Checkpoint preentrenado alternativo para cargar pesos antes de entrenar",
    )
    parser.add_argument(
        "--optimizer",
        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        default=None,
        help="Optimizador a usar en entrenamiento",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=None,
        help="Desactiva mosaic en las ultimas N épocas",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        help="Activa scheduler cosenoidal para el learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay del optimizador",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout del head cuando aplique",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=None,
        help="Numero de épocas de warmup",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Ejecuta hyperparameter tuning en lugar de entrenamiento normal",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Numero de iteraciones para el tuning",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Ruta a archivo .yaml con hiperparámetros optimizados (ej. best_hyperparameters.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime configuración, no ejecuta entrenamiento",
    )
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    # Adapta los argumentos CLI al formato esperado por model.train().
    # Separar esta funcion facilita pruebas unitarias y reutilización.
    train_kwargs: Dict[str, Any] = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(_resolve_project_dir(args.project)),
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
    }

    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.resume:
        train_kwargs["resume"] = True
    if args.optimizer is not None:
        train_kwargs["optimizer"] = args.optimizer
    if args.close_mosaic is not None:
        train_kwargs["close_mosaic"] = args.close_mosaic
    if args.cos_lr:
        train_kwargs["cos_lr"] = True
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        train_kwargs["dropout"] = args.dropout
    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs

    return train_kwargs


def _materialize_data_yaml_for_ultralytics(data_path: Path) -> tuple[Path, Path | None]:
    # Ultralytics puede resolver `path:` relativo respecto al CWD; lo convertimos a absoluto.
    lines = data_path.read_text(encoding="utf-8").splitlines()

    path_line_index: int | None = None
    path_value: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("path:"):
            path_line_index = index
            path_value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            break

    if path_line_index is None or path_value is None:
        return data_path, None

    candidate = Path(path_value)
    if candidate.is_absolute():
        return data_path, None

    resolved_root = (data_path.parent / candidate).resolve().as_posix()
    indent = lines[path_line_index][: len(lines[path_line_index]) - len(lines[path_line_index].lstrip())]
    lines[path_line_index] = f"{indent}path: {resolved_root}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(lines) + "\n")
        temp_path = Path(tmp.name)

    return temp_path, temp_path


def main() -> int:
    # 1) Lee argumentos y valida precondiciones mínimas.
    args = parse_args()
    data_path = Path(args.data)

    # El entrenamiento debe fallar rapido si falta el manifiesto del dataset.
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro data.yaml: {data_path}")

    if args.pretrained is not None:
        pretrained_path = Path(args.pretrained)
        if pretrained_path.suffix == ".pt" and not pretrained_path.exists():
            raise FileNotFoundError(f"No se encontro checkpoint pretrained: {pretrained_path}")

    model_path = Path(args.model)
    if model_path.suffix == ".pt" and not model_path.exists() and not args.model.startswith("yolo"):
        raise FileNotFoundError(f"No se encontro modelo/checkpoint: {model_path}")

    temp_data_yaml: Path | None = None
    if not args.dry_run:
        resolved_data_yaml, temp_data_yaml = _materialize_data_yaml_for_ultralytics(data_path)
        args.data = str(resolved_data_yaml)

    # 2) Prepara e imprime la configuración final que se enviara a Ultralytics.
    device_resolution = resolve_device(args.device)
    args.device = device_resolution.resolved
    train_kwargs = build_train_kwargs(args)

    # Permite cargar un checkpoint distinto al de --model cuando se quiera conservar
    # la arquitectura base indicada en --model pero arrancar desde otros pesos.
    model_source = args.pretrained if args.pretrained is not None else args.model

    # Mostrar configuración ayuda a reproducir la corrida exacta después.
    print("Configuración de entrenamiento:")
    print(f" model: {args.model}")
    print(f" model_source: {model_source}")
    print(f" requested_device: {device_resolution.requested}")
    print(f" resolved_device: {device_resolution.resolved}")
    if device_resolution.warning:
        print(f" warning: {device_resolution.warning}")
    for key, value in train_kwargs.items():
        print(f" {key}: {value}")

    # 3) Permite validar configuración sin coste de entrenamiento.
    # Muy util para revisar rutas/flags antes de una corrida larga.
    if args.dry_run:
        print("Dry run activo: no se ejecuto entrenamiento.")
        return 0

    # Import diferido para permitir pruebas sin instalar ultralytics.
    from ultralytics import YOLO, settings
    import yaml

    if args.cfg is not None:
        cfg_path = Path(args.cfg)
        if cfg_path.exists():
            print(f"Cargando hiperparámetros desde: {cfg_path}")
            with open(cfg_path, 'r', encoding='utf-8') as f:
                custom_hyp = yaml.safe_load(f)
                # Actualizar train_kwargs con los hiperparámetros sintonizados
                if isinstance(custom_hyp, dict):
                    # Filtramos keys que podrían interferir (como 'task' o configuraciones de sistema)
                    safe_hyp = {k: v for k, v in custom_hyp.items() if k not in ['task', 'model', 'data', 'project', 'name', 'device', 'epochs', 'batch', 'imgsz']}
                    train_kwargs.update(safe_hyp)
        else:
            print(f"Advertencia: No se encontró el archivo cfg {cfg_path}")

    # Mantiene todos los artefactos de Ultralytics bajo `entrenamiento ia/runs`.
    settings.update({"runs_dir": str((TOOLS_ROOT / "runs").resolve())})

    # 4) Carga pesos base y ejecuta el entrenamiento.
    # Si `model_source` es un .pt, usa pesos preentrenados o checkpoint;
    # si es .yaml, inicializa arquitectura desde definición.
    try:
        model = YOLO(model_source)
        if args.tune:
            print(f"Iniciando Tuning de Hiperparámetros con {args.iterations} iteraciones...")
            train_kwargs["iterations"] = args.iterations
            model.tune(**train_kwargs)
        else:
            model.train(**train_kwargs)
    finally:
        if temp_data_yaml is not None and temp_data_yaml.exists():
            temp_data_yaml.unlink(missing_ok=True)

    # `trainer.save_dir` refleja el directorio final (incluyendo sufijos incrementales).
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is not None:
        print(f"ultralytics_save_dir: {Path(save_dir).resolve()}")

    print("Entrenamiento finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())