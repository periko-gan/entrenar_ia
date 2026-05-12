import sys
from pathlib import Path

# Import directo del script para probar la normalizacion sin ejecutar Ultralytics.
TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval_predict_yolov8 import extract_metrics


class FakeScalarArray:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBox:
    map = FakeScalarArray(0.12)
    map50 = FakeScalarArray(0.34)
    map75 = FakeArray([0.56, 0.78])
    maps = FakeArray([0.1, 0.2, 0.3])


class FakeMetrics:
    fitness = FakeScalarArray(0.9)
    speed = {
        "preprocess": FakeScalarArray(0.8),
        "inference": FakeScalarArray(3.0),
    }
    box = FakeBox()
    results_dict = {
        "metrics/precision(B)": FakeScalarArray(0.11),
        "metrics/recall(B)": FakeScalarArray(0.22),
        "metrics/mAP50(B)": FakeArray([0.33, 0.44]),
    }


def test_extract_metrics_handles_array_like_values():
    metrics = extract_metrics(FakeMetrics())

    assert metrics["fitness"] == 0.9
    assert metrics["speed"]["preprocess"] == 0.8
    assert metrics["box_map"] == 0.12
    assert metrics["box_map50"] == 0.34
    assert metrics["box_map75"] == [0.56, 0.78]
    assert metrics["box_maps"] == [0.1, 0.2, 0.3]
    assert metrics["metrics/precision(B)"] == 0.11
    assert metrics["metrics/mAP50(B)"] == [0.33, 0.44]

