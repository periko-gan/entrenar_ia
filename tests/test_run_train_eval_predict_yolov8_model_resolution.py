import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_train_eval_predict_yolov8 import _resolve_trained_model


def test_resolve_trained_model_raises_when_weights_missing(tmp_path: Path):
    train_project = tmp_path / "runs" / "train"

    with pytest.raises(FileNotFoundError):
        _resolve_trained_model(train_project, "missing_run", "yolov8n.pt")


def test_resolve_trained_model_prefers_best(tmp_path: Path):
    weights_dir = tmp_path / "runs" / "train" / "run_1" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best = weights_dir / "best.pt"
    best.write_bytes(b"fake")

    resolved = _resolve_trained_model(tmp_path / "runs" / "train", "run_1", "yolov8n.pt")

    assert resolved == str(best.resolve())
