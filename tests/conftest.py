"""Shared fixtures.

Most tests use a stub model, so they run anywhere (including CI, where the
gitignored weights are absent). Tests that need the real weights use the
`real_models` fixture, which skips when the files are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import MODEL_FILES, load_models  # noqa: E402


class _Tensor:
    """Just enough of a torch tensor: the code under test only calls .tolist()."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _Boxes:
    xyxy = _Tensor([[10.04, 20.06, 110.0, 220.0]])
    conf = _Tensor([0.91234])
    cls = _Tensor([4.0])


class _Result:
    boxes = _Boxes()


class StubModel:
    """Stands in for ultralytics.YOLO and always finds one white queen."""

    names = {4: "white-queen"}

    def predict(self, image, **kwargs):
        return [_Result()]


class _Keypoints:
    def __init__(self, xy):
        self.xy = _Tensor(xy)


class _CornerBoxes:
    def __init__(self, confidences):
        self.conf = _Tensor(confidences)


class _CornerResult:
    def __init__(self, confidences, xy):
        self.boxes = _CornerBoxes(confidences)
        self.keypoints = _Keypoints(xy)


class StubCornerModel:
    """Stands in for the corner detector: reports one board with the given corners, or no
    board at all when `corners` is None. Corners are in screen order (TL, TR, BR, BL)."""

    def __init__(self, corners=None):
        self.corners = corners

    def predict(self, image, **kwargs):
        if self.corners is None:
            return [_CornerResult([], [])]
        return [_CornerResult([0.93], [self.corners])]


EXPECTED_STUB_DETECTION = {
    "class_name": "white-queen",
    "confidence": 0.9123,
    "bbox": [10.0, 20.1, 110.0, 220.0],
}


@pytest.fixture()
def stub_models() -> dict:
    return {name: StubModel() for name in MODEL_FILES}


@pytest.fixture(scope="session")
def real_models():
    missing = [f for f in MODEL_FILES.values() if not (ROOT / "models" / f).is_file()]
    if missing:
        pytest.skip(f"weights not present: {', '.join(missing)}")
    return load_models(ROOT / "models")


@pytest.fixture()
def tiny_image() -> Image.Image:
    """A generated board-like image, so no dataset images live in the repo."""
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    for row in range(4):
        for col in range(4):
            if (row + col) % 2:
                draw.rectangle(
                    [col * 60 + 40, row * 50 + 20, col * 60 + 99, row * 50 + 69],
                    fill="black",
                )
    return img
