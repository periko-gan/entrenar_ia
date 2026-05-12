import subprocess
import sys
from pathlib import Path

# Como leer este test:
# 1) ARRANGE: prepara `data.yaml` y un directorio `source` temporal.
# 2) ACT: ejecuta `eval_predict_yolov8.py --task both --dry-run`.
# 3) ASSERT: valida retorno correcto y mensaje esperado en stdout.


def test_eval_predict_script_dry_run(tmp_path: Path):
    # ARRANGE: creamos data.yaml y source para cubrir validaciones de rutas.
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

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    # ACT: ejecutamos el script en dry-run para no requerir inferencia real.
    script_path = Path(__file__).resolve().parents[1] / "tools" / "eval_predict_yolov8.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--task",
            "both",
            "--data",
            str(data_yaml),
            "--source",
            str(source_dir),
            "--device",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # ASSERT: la ejecucion finaliza correctamente y reporta dry-run.
    assert result.returncode == 0, result.stderr
    assert "Dry run activo" in result.stdout
    assert "requested_device: 0" in result.stdout
    assert "resolved_device:" in result.stdout

