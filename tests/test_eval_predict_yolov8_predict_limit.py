import sys
from argparse import Namespace
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval_predict_yolov8 import build_predict_kwargs


def test_build_predict_kwargs_limits_directory_sources(tmp_path: Path):
    source_dir = tmp_path / "images"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "a.jpg").write_bytes(b"x")
    (source_dir / "b.png").write_bytes(b"x")
    (source_dir / "c.jpeg").write_bytes(b"x")
    (source_dir / "ignore.txt").write_text("x", encoding="utf-8")

    args = Namespace(
        source=str(source_dir),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device="cpu",
        project="runs/eval_predict",
        name="demo",
        max_pred_images=2,
    )

    kwargs = build_predict_kwargs(args)

    assert isinstance(kwargs["source"], list)
    assert len(kwargs["source"]) == 2
    assert kwargs["source"][0].endswith("a.jpg")
    assert kwargs["source"][1].endswith("b.png")
