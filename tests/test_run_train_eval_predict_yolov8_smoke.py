import json
import subprocess
import sys
from pathlib import Path

# Como leer este test:
# 1) ARRANGE: crea entradas y carpetas temporales para pipeline.
# 2) ACT: ejecuta `run_train_eval_predict_yolov8.py --dry-run`.
# 3) ASSERT: verifica invocacion de pasos y `pipeline_report.json`.


def test_pipeline_script_dry_run(tmp_path: Path):
    # ARRANGE: creamos un dataset minimo en un directorio temporal.
    # El test no necesita imagenes reales porque se ejecuta en modo --dry-run.
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

    # Directorios de salida separados para verificar que el pipeline respeta rutas custom.
    train_project = tmp_path / "runs" / "train"
    eval_project = tmp_path / "runs" / "eval_predict"
    pipeline_project = tmp_path / "runs" / "pipeline"

    # Ruta absoluta al script orquestador que vamos a probar.
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_train_eval_predict_yolov8.py"

    # ACT: ejecutamos el pipeline en seco.
    # Importante: usamos subprocess para simular ejecucion real desde terminal.
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data",
            str(data_yaml),
            "--source",
            str(source_dir),
            "--project-train",
            str(train_project),
            "--project-eval",
            str(eval_project),
            "--project-pipeline",
            str(pipeline_project),
            "--device",
            "0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # ASSERT: el proceso termina correctamente y muestra que invoca los 2 pasos.
    assert result.returncode == 0, result.stderr
    assert "[train] Ejecutando:" in result.stdout
    assert "[eval_predict] Ejecutando:" in result.stdout
    assert "Pipeline completado." in result.stdout
    assert "requested_device: 0" in result.stdout

    # ASSERT: el pipeline debe crear un unico reporte JSON en la carpeta de pipeline.
    # Si hubiera 0 o >1 reportes, indicaria que el nombrado/versionado no es consistente.
    report_files = list(pipeline_project.glob("*/pipeline_report.json"))
    assert len(report_files) == 1

    # ASSERT: validamos campos clave para confirmar que el modo dry-run y la tarea por
    # defecto ('both') se propagan correctamente al reporte.
    report_data = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report_data["dry_run"] == "True"
    assert report_data["task"] == "both"

