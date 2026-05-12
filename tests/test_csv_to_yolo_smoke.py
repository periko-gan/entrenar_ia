import csv
import json
import subprocess
import sys
from pathlib import Path

# Como leer este test:
# 1) ARRANGE: crea un mini-dataset temporal (imagenes + CSV por split).
# 2) ACT: ejecuta `csv_to_yolo.py` como subprocess.
# 3) ASSERT: verifica etiquetas YOLO, `data.yaml` y `conversion_report.json`.


def _touch(path: Path) -> None:
    # Helper de test: crea una "imagen" dummy (solo para que exista el archivo).
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _write_csv(path: Path, rows):
    # Helper de test: escribe un CSV compatible con el conversor.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
        writer.writerows(rows)


def test_smoke_conversion(tmp_path: Path):
    # ARRANGE: construimos un dataset minimo temporal con 3 splits.
    dataset_root = tmp_path / "dataset"

    for split in ("train", "val", "test"):
        _touch(dataset_root / "images" / split / f"{split}_img.jpg")

    # Creamos anotaciones de ejemplo (una por split) para probar el flujo completo.
    _write_csv(
        dataset_root / "labels" / "train" / "_annotations.csv",
        [["train_img.jpg", 100, 100, "Fillings", 10, 10, 50, 60]],
    )
    _write_csv(
        dataset_root / "labels" / "val" / "_annotations.csv",
        [["val_img.jpg", 200, 100, "Implant", 20, 10, 120, 60]],
    )
    _write_csv(
        dataset_root / "labels" / "test" / "_annotations.csv",
        [["test_img.jpg", 100, 200, "Fillings", 0, 0, 50, 100]],
    )

    # ACT: ejecutamos el conversor como lo haria un usuario desde terminal.
    script_path = Path(__file__).resolve().parents[1] / "tools" / "csv_to_yolo.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dataset-root",
            str(dataset_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    # ASSERT 1: se crea etiqueta YOLO en la ruta esperada y con formato valido.
    train_label = dataset_root / "labels" / "train" / "train_img.txt"
    assert train_label.exists()
    first_line = train_label.read_text(encoding="utf-8").strip()
    assert first_line.startswith("0 ")

    # ASSERT 2: se genera data.yaml con rutas coherentes para Ultralytics.
    data_yaml = dataset_root / "data.yaml"
    assert data_yaml.exists()
    data_yaml_text = data_yaml.read_text(encoding="utf-8")
    assert "path: ." in data_yaml_text
    assert "train: images/train" in data_yaml_text
    assert "val: images/val" in data_yaml_text

    # ASSERT 3: el reporte JSON registra al menos una caja valida en train.
    report = json.loads((dataset_root / "conversion_report.json").read_text(encoding="utf-8"))
    assert report["splits"]["train"]["valid_boxes"] == 1

