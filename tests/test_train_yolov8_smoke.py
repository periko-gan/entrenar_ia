import subprocess
import sys
from pathlib import Path

# Como leer este test:
# 1) ARRANGE: crea un `data.yaml` minimo temporal.
# 2) ACT: ejecuta `train_yolov8.py --dry-run`.
# 3) ASSERT: confirma salida exitosa y mensaje de dry-run.


def test_train_script_dry_run(tmp_path: Path):
    # ARRANGE: creamos un data.yaml minimo para validar solo el flujo CLI.
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: lesion\n",
        encoding="utf-8",
    )

    # ACT: ejecutamos en dry-run para evitar entrenamiento real dentro del test.
    script_path = Path(__file__).resolve().parents[1] / "tools" / "train_yolov8.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data",
            str(data_yaml),
            "--device",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # ASSERT: el script termina bien y confirma modo dry-run en stdout.
    assert result.returncode == 0, result.stderr
    assert "Dry run activo" in result.stdout
    assert "requested_device: 0" in result.stdout
    assert "resolved_device:" in result.stdout

