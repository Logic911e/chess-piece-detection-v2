"""Unit tests for src/predict.py."""

from __future__ import annotations

import pytest

from predict import MODEL_FILES, load_models, predict


def test_both_models_load_with_same_classes(models):
    assert set(models) == set(MODEL_FILES)
    assert models["synthetic"].names == models["finetuned"].names
    assert len(models["synthetic"].names) == 12


def test_missing_weights_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models(tmp_path)


@pytest.mark.parametrize("name", ["synthetic", "finetuned"])
def test_result_shape(models, tiny_image, name):
    out = predict(models, name, tiny_image, conf=0.05)
    assert out["model"] == name
    assert (out["image_width"], out["image_height"]) == tiny_image.size
    assert out["inference_ms"] > 0
    for det in out["detections"]:
        assert det["class_name"] in models[name].names.values()
        assert 0.0 <= det["confidence"] <= 1.0
        x1, y1, x2, y2 = det["bbox"]
        assert x1 <= x2 and y1 <= y2


def test_conf_of_one_returns_nothing(models, tiny_image):
    # Scores are strictly below 1.0 in practice, so nothing survives.
    assert predict(models, "synthetic", tiny_image, conf=1.0)["detections"] == []


def test_unknown_model_raises(models, tiny_image):
    with pytest.raises(ValueError, match="Unknown model"):
        predict(models, "resnet", tiny_image)


def test_detections_are_converted_to_plain_dicts(tiny_image):
    """The fixture image yields no real detections, so a stub model with known
    boxes checks the conversion (class lookup, rounding, plain floats)."""

    class Boxes:
        xyxy = _Tensor([[10.04, 20.06, 110.0, 220.0]])
        conf = _Tensor([0.91234])
        cls = _Tensor([4.0])

    class Result:
        boxes = Boxes()

    class StubModel:
        names = {4: "white-queen"}

        def predict(self, image, **kwargs):
            return [Result()]

    out = predict({"stub": StubModel()}, "stub", tiny_image, conf=0.3)
    assert out["detections"] == [
        {"class_name": "white-queen", "confidence": 0.9123, "bbox": [10.0, 20.1, 110.0, 220.0]}
    ]


class _Tensor:
    """Just enough of a torch tensor: the code under test only calls .tolist()."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values
