#!/usr/bin/env python3
"""Evaluacion y prediccion YOLOv8 con reporte consolidado."""

# Como leer este archivo:
# 1) `parse_args()` define el modo (`val`, `predict` o `both`).
# 2) `build_val_kwargs()` y `build_predict_kwargs()` arman parametros de Ultralytics.
# 3) `extract_metrics()` normaliza metricas para exportarlas.
# 4) `main()` ejecuta tareas y genera `run_report.json`.
#
# Ejemplo minimo de uso:
# python .\tools\eval_predict_yolov8.py --task both --data .\dataset\data.yaml --source .\dataset\images\test --model yolov8n.pt --device cpu

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from device_resolver import resolve_device


TOOLS_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_project_dir(project_arg: str) -> Path:
    # Fuerza rutas relativas a vivir dentro de `entrenamiento ia` para evitar salidas fuera.
    project_path = Path(project_arg)
    if project_path.is_absolute():
        return project_path
    return (TOOLS_ROOT / project_path).resolve()


def parse_args() -> argparse.Namespace:
    # Expone un CLI unico para validar, predecir o ejecutar ambos pasos.
    # Esto evita mantener dos scripts separados para tareas muy parecidas.
    parser = argparse.ArgumentParser(
        description="Evalua y/o genera predicciones con YOLOv8."
    )
    parser.add_argument("--task", choices=("val", "predict", "both"), default="both")
    parser.add_argument("--data", default="dataset/data.yaml", help="Ruta al data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo base o checkpoint")
    parser.add_argument(
        "--source",
        default="dataset/images/test",
        help="Ruta de imagen, carpeta o video para prediccion",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen")
    parser.add_argument("--batch", type=int, default=16, help="Batch size para validacion")
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo: auto, cpu, 0, 0,1, ... (auto usa GPU si existe)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza minima")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU NMS para prediccion")
    parser.add_argument("--project", default="runs/eval_predict", help="Directorio base")
    parser.add_argument("--name", default="dental_eval_predict", help="Nombre del experimento")
    parser.add_argument(
        "--max-pred-images",
        type=int,
        default=100,
        help="Limite de imágenes a procesar en predicción",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime configuración sin ejecutar ultralytics",
    )
    return parser.parse_args()


def build_val_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    # Construye los argumentos de model.val() a partir del CLI.
    return {
        "data": str(Path(args.data).resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(_resolve_project_dir(args.project)),
        "name": args.name,
    }


def build_predict_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    # Construye los argumentos de model.predict() con guardado de resultados activados.
    # save=True asegura visualizar rapidamente resultados sin codigo adicional.
    source_path = Path(args.source).resolve()
    predict_source: str | list[str] = str(source_path)
    if args.max_pred_images > 0 and source_path.is_dir():
        image_paths = [
            str(image_path.resolve())
            for image_path in sorted(source_path.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if image_paths:
            predict_source = image_paths[: args.max_pred_images]

    return {
        "source": predict_source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "project": str(_resolve_project_dir(args.project)),
        "name": args.name,
        "save": True,
        "stream": False,
        "max_det": 300,
    }


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


def _normalize_metric_value(value: Any) -> Any:
    # Convierte valores de Ultralytics (incluyendo arrays/tensores) a tipos JSON-safe.
    if value is None:
        return None

    if isinstance(value, (bool, int, float, str)):
        return value

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return _normalize_metric_value(value.tolist())
        except Exception:
            pass

    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        for k, v in value.items():
            normalized_v = _normalize_metric_value(v)
            if normalized_v is not None:
                normalized[str(k)] = normalized_v
        return normalized

    if isinstance(value, (list, tuple)):
        normalized_list = []
        for item in value:
            normalized_item = _normalize_metric_value(item)
            if normalized_item is not None:
                normalized_list.append(normalized_item)
        return normalized_list

    try:
        return float(value)
    except Exception:
        return str(value)


def extract_metrics(metrics_obj: Any) -> Dict[str, Any]:
    # Extrae metadatos estables para serializarlos en JSON/CSV.
    # Esta capa protege contra cambios menores en la estructura interna de Ultralytics.
    metrics: Dict[str, Any] = {}
    for key in ("fitness", "speed"):
        value = getattr(metrics_obj, key, None)
        normalized_value = _normalize_metric_value(value)
        if normalized_value is not None:
            metrics[key] = normalized_value

    box_obj = getattr(metrics_obj, "box", None)
    if box_obj is not None:
        for source_name in ("map", "map50", "map75", "maps"):
            value = getattr(box_obj, source_name, None)
            normalized_value = _normalize_metric_value(value)
            if normalized_value is not None:
                metrics[f"box_{source_name}"] = normalized_value

    # Fallback para objetos no estandar.
    # Si la API devuelve un diccionario de resultados, se conserva lo simple (int/float/str).
    results_dict = getattr(metrics_obj, "results_dict", None)
    if isinstance(results_dict, dict):
        for k, v in results_dict.items():
            normalized_value = _normalize_metric_value(v)
            if normalized_value is not None:
                metrics[str(k)] = normalized_value

    return metrics


def write_metrics_files(metrics: Dict[str, Any], output_dir: Path) -> None:
    # Escribe métricas en formatos legibles para humanos (CSV) y máquinas (JSON).
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "val_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_path = output_dir / "val_metrics.csv"
    # El CSV en formato key/value facilita abrir métricas en Excel o Google Sheets.
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics.keys()):
            writer.writerow([key, metrics[key]])


def write_run_report(
    output_dir: Path,
    args: argparse.Namespace,
    val_metrics: Dict[str, Any] | None,
    prediction_count: int | None,
    ultralytics_save_dir: str | None,
) -> Path:
    # Registra un resumen de ejecución para trazabilidad del experimento.
    # Incluye configuración y rutas para reconstruir más adelante.
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "task": args.task,
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "source": str(Path(args.source).resolve()),
        "output_dir": str(output_dir.resolve()),
        "ultralytics_save_dir": ultralytics_save_dir,
        "val_metrics": val_metrics,
        "prediction_items": prediction_count,
    }

    report_path = output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _validate_inputs(args: argparse.Namespace) -> None:
    # Verífica rutas criticas antes de cargar YOLO para fallar rapido.
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro data.yaml: {data_path}")

    # Solo exigimos source cuando se va a ejecutar inferencia.
    if args.task in ("predict", "both"):
        source_path = Path(args.source)
        if not source_path.exists():
            raise FileNotFoundError(f"No se encontro source para prediccion: {source_path}")


def _predict_count(results: Iterable[Any]) -> int:
    # Cuenta resultados iterables sin asumir estructura interna de Ultralytics.
    count = 0
    for _ in results:
        count += 1
    return count


def _extract_save_dir(value: Any) -> str | None:
    save_dir = getattr(value, "save_dir", None)
    if save_dir is None:
        return None
    return str(Path(save_dir).resolve())


def main() -> int:
    # 1) Prepara y valida configuración de entrada.
    args = parse_args()
    _validate_inputs(args)

    temp_data_yaml: Path | None = None
    if not args.dry_run and args.task in ("val", "both"):
        resolved_data_yaml, temp_data_yaml = _materialize_data_yaml_for_ultralytics(Path(args.data))
        args.data = str(resolved_data_yaml)

    device_resolution = resolve_device(args.device)
    args.device = device_resolution.resolved

    # 2) Compone parámetros de val/predict y muestra configuración final.
    val_kwargs = build_val_kwargs(args)
    predict_kwargs = build_predict_kwargs(args)
    # Carpeta canónica de salida para reportes y artefactos.
    output_dir = (_resolve_project_dir(args.project) / args.name).resolve()

    print("Configuracion eval/predict:")
    print(f"  task: {args.task}")
    print(f"  model: {args.model}")
    print(f"  requested_device: {device_resolution.requested}")
    print(f"  resolved_device: {device_resolution.resolved}")
    if device_resolution.warning:
        print(f"  warning: {device_resolution.warning}")
    print(f"  data: {val_kwargs['data']}")
    print(f"  source: {predict_kwargs['source']}")
    print(f"  output_dir: {output_dir}")

    # 3) Permite revisar configuracion sin ejecutar inferencia real.
    # Es equivalente a un "lint" operacional de la corrida.
    if args.dry_run:
        print("Dry run activo: no se ejecuto evaluacion ni prediccion.")
        return 0

    # Import diferido para permitir pruebas smoke sin ultralytics.
    from ultralytics import YOLO, settings

    # Mantiene todos los artefactos de Ultralytics bajo `entrenamiento ia/runs`.
    settings.update({"runs_dir": str((TOOLS_ROOT / "runs").resolve())})

    # 4) Carga modelo y ejecuta tareas solicitadas.
    val_metrics: Dict[str, Any] | None = None
    prediction_count: int | None = None
    ultralytics_save_dir: str | None = None
    try:
        model = YOLO(args.model)

        if args.task in ("val", "both"):
            # Guarda métricas de validación para análisis posterior.
            metrics_obj = model.val(**val_kwargs)
            ultralytics_save_dir = ultralytics_save_dir or _extract_save_dir(metrics_obj)
            val_metrics = extract_metrics(metrics_obj)
            write_metrics_files(val_metrics, output_dir)

        if args.task in ("predict", "both"):
            # Ejecuta predicción y reporta cuantas muestras fueron procesadas.
            results = model.predict(**predict_kwargs)
            prediction_count = _predict_count(results)
            if ultralytics_save_dir is None:
                ultralytics_save_dir = _extract_save_dir(results)
                if ultralytics_save_dir is None and isinstance(results, list) and results:
                    ultralytics_save_dir = _extract_save_dir(results[0])
    finally:
        if temp_data_yaml is not None and temp_data_yaml.exists():
            temp_data_yaml.unlink(missing_ok=True)

    # 5) Escribe reporte global de ejecución.
    report_path = write_run_report(
        output_dir=output_dir,
        args=args,
        val_metrics=val_metrics,
        prediction_count=prediction_count,
        ultralytics_save_dir=ultralytics_save_dir,
    )

    if ultralytics_save_dir:
        print(f"ultralytics_save_dir: {ultralytics_save_dir}")

    # Mensaje final con la ruta principal de salida.
    print(f"Proceso completado. Reporte: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

