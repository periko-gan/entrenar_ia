#!/usr/bin/env python3
"""Convierte anotaciones CSV por split al formato YOLOv8."""

# Como leer este archivo:
# 1) `parse_args()` define entradas del CLI.
# 2) `collect_classes()` y `convert_split()` realizan la conversion principal.
# 3) `write_data_yaml()` y `write_report()` guardan artefactos finales.
# 4) `main()` orquesta todo el flujo de extremo a extremo.
#
# Ejemplo minimo de uso:
# python .\tools\csv_to_yolo.py --dataset-root .\dataset

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Columnas obligatorias para poder reconstruir una bounding box en formato YOLO.
REQUIRED_COLUMNS = {
    "filename",
    "width",
    "height",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
}
# Extensiones de imagen admitidas durante la validación de archivos.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Splits esperados en estructura estilo YOLO.
SPLITS = ("train", "val", "test")


@dataclass
class Box:
    # Representa una caja ya normalizada al formato YOLO (valores entre 0 y 1).
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class SplitSummary:
    # Acumula métricas de calidad del split para auditar conversion y detectar problemas.
    csv_rows: int = 0
    valid_boxes: int = 0
    invalid_rows: int = 0
    missing_images: int = 0
    images_in_csv: int = 0
    images_on_disk: int = 0
    empty_label_files: int = 0


def parse_args() -> argparse.Namespace:
    # Define las rutas y nombres configurables del proceso de conversion.
    parser = argparse.ArgumentParser(
        description="Convierte CSV con bounding boxes a etiquetas YOLO por split."
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Raiz del dataset con carpetas images/ y labels/ (default: dataset).",
    )
    parser.add_argument(
        "--csv-name",
        default="_annotations.csv",
        help="Nombre del CSV dentro de labels/<split>/ (default: _annotations.csv).",
    )
    parser.add_argument(
        "--output-labels-dir",
        default="labels",
        help="Carpeta de salida bajo dataset-root para labels YOLO (default: labels).",
    )
    parser.add_argument(
        "--data-yaml",
        default="data.yaml",
        help="Nombre del data.yaml de salida dentro de dataset-root (default: data.yaml).",
    )
    return parser.parse_args()


def collect_classes(dataset_root: Path, csv_name: str) -> List[str]:
    # Recorre todos los splits para construir un vocabulario de clases consistente.
    classes: Set[str] = set()
    for split in SPLITS:
        csv_path = dataset_root / "labels" / split / csv_name
        if not csv_path.exists():
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Validación temprana: fallamos antes de convertir, si el esquema es incompleto.
            if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
                missing = REQUIRED_COLUMNS.difference(set(reader.fieldnames or []))
                raise ValueError(f"CSV inválido en {csv_path}: faltan columnas {sorted(missing)}")
            for row in reader:
                value = (row.get("class") or "").strip()
                if value:
                    classes.add(value)
    if not classes:
        raise ValueError("No se encontraron clases en los CSV.")
    return sorted(classes)


def list_images(images_dir: Path) -> Set[str]:
    # Lista únicamente archivos de imagen válidos para evitar falsos positivos.
    if not images_dir.exists():
        return set()
    return {
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def row_to_box(row: Dict[str, str], class_to_id: Dict[str, int]) -> Tuple[Box | None, str | None]:
    # Convierte una fila CSV en caja YOLO normalizada y valida coordenadas.
    class_name = (row.get("class") or "").strip()
    if class_name not in class_to_id:
        return None, f"Clase desconocida: {class_name!r}"

    # Parseo robusto: acepta enteros/decimales en texto y convierte a números reales.
    try:
        img_w = int(float(row["width"]))
        img_h = int(float(row["height"]))
        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])
    except (KeyError, ValueError) as exc:
        return None, f"Valores no numericos: {exc}"

    # Evita divisiones inválidas y datos físicamente imposibles.
    if img_w <= 0 or img_h <= 0:
        return None, "Dimensiones no validas"

    # Ajuste defensivo de coordenadas al tamaño de imagen.
    xmin = max(0.0, min(xmin, float(img_w)))
    xmax = max(0.0, min(xmax, float(img_w)))
    ymin = max(0.0, min(ymin, float(img_h)))
    ymax = max(0.0, min(ymax, float(img_h)))

    # Una caja sin área no sirve para entrenamiento.
    if xmax <= xmin or ymax <= ymin:
        return None, "Caja degenerada tras normalización"

    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h

    return Box(class_to_id[class_name], x_center, y_center, width, height), None


def write_label_file(path: Path, boxes: Iterable[Box]) -> None:
    # Genera el .txt YOLO por imagen, con una línea por caja.
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    # YOLO espera una línea por caja: class_id x_center y_center width height.
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def convert_split(
    dataset_root: Path,
    split: str,
    csv_name: str,
    class_to_id: Dict[str, int],
    output_labels_root: Path,
) -> SplitSummary:
    # Procesa un split completo: válida filas, agrupa cajas y escribe etiquetas.
    summary = SplitSummary()
    csv_path = dataset_root / "labels" / split / csv_name
    images_dir = dataset_root / "images" / split
    output_dir = output_labels_root / split

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Directorio de imágenes no encontrado: {images_dir}")

    # Referencia de verdad para validar que cada fila apunte a una imagen existente.
    image_files = list_images(images_dir)
    summary.images_on_disk = len(image_files)

    grouped_boxes: Dict[str, List[Box]] = defaultdict(list)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            missing = REQUIRED_COLUMNS.difference(set(reader.fieldnames or []))
            raise ValueError(f"CSV invalido en {csv_path}: faltan columnas {sorted(missing)}")

        for row in reader:
            # Acumula estadísticas de calidad para el reporte final.
            summary.csv_rows += 1
            filename = (row.get("filename") or "").strip()
            if not filename:
                summary.invalid_rows += 1
                continue
            # Si la imagen no existe en disco, se descarta la fila y se reporta.
            if filename not in image_files:
                summary.missing_images += 1
                continue

            # Reusa la validación centralizada de coordenadas y normalización.
            box, error = row_to_box(row, class_to_id)
            if error:
                summary.invalid_rows += 1
                continue

            grouped_boxes[filename].append(box)
            summary.valid_boxes += 1

    summary.images_in_csv = len(grouped_boxes)

    # Escribe un .txt por imagen para que YOLO soporte también imágenes sin cajas.
    # Si una imagen no tiene objetos, su archivo .txt queda vacío a propósito.
    for image_name in sorted(image_files):
        label_path = output_dir / f"{Path(image_name).stem}.txt"
        boxes = grouped_boxes.get(image_name, [])
        write_label_file(label_path, boxes)
        if not boxes:
            summary.empty_label_files += 1

    return summary


def write_data_yaml(dataset_root: Path, data_yaml_name: str, class_names: List[str]) -> Path:
    # Escribe el manifiesto de dataset que consumirà Ultralytics.
    data_yaml_path = dataset_root / data_yaml_name
    # Usar ruta relativa evita que el data.yaml quede atado a una maquina concreta.
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    # Se respeta el orden class_id -> nombre para mantener consistencia de etiquetas.
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    data_yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml_path


def write_report(dataset_root: Path, report: Dict[str, object]) -> Path:
    # Persistencia de resumen de conversion para auditoria/reproducibilidad.
    report_path = dataset_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    # 1) Inicializa rutas y mapeo clase->id.
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()

    # Las clases se detectan desde CSV para evitar listas hardcodeadas.
    class_names = collect_classes(dataset_root, args.csv_name)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    # Directorio final de etiquetas YOLO (por defecto: dataset/labels).
    output_labels_root = dataset_root / args.output_labels_dir
    output_labels_root.mkdir(parents=True, exist_ok=True)

    # 2) Convierte cada split al formato YOLO.
    split_summaries: Dict[str, SplitSummary] = {}
    for split in SPLITS:
        split_summaries[split] = convert_split(
            dataset_root=dataset_root,
            split=split,
            csv_name=args.csv_name,
            class_to_id=class_to_id,
            output_labels_root=output_labels_root,
        )

    # 3) Emite archivos de salida (data.yaml + reporte JSON).
    data_yaml_path = write_data_yaml(dataset_root, args.data_yaml, class_names)

    # 4) Construye un reporte legible para trazabilidad y depuración.
    report: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "class_to_id": class_to_id,
        "labels_output": str(output_labels_root),
        "data_yaml": str(data_yaml_path),
        "splits": {
            split: {
                "csv_rows": summary.csv_rows,
                "valid_boxes": summary.valid_boxes,
                "invalid_rows": summary.invalid_rows,
                "missing_images": summary.missing_images,
                "images_in_csv": summary.images_in_csv,
                "images_on_disk": summary.images_on_disk,
                "empty_label_files": summary.empty_label_files,
            }
            for split, summary in split_summaries.items()
        },
    }
    report_path = write_report(dataset_root, report)

    print("Conversion completada.")
    print(f"Clases detectadas: {class_to_id}")
    print(f"Etiquetas YOLO en: {output_labels_root}")
    print(f"Archivo data.yaml: {data_yaml_path}")
    print(f"Reporte: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

