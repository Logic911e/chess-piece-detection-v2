"""Unit tests for src/predict.py."""

from __future__ import annotations

import pytest
from conftest import EXPECTED_STUB_DETECTION

from predict import MODEL_FILES, load_models, predict


def test_missing_weights_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_models(tmp_path)


def test_detections_are_converted_to_plain_dicts(stub_models, tiny_image):
    out = predict(stub_models, "synthetic", tiny_image, conf=0.3)
    assert out["model"] == "synthetic"
    assert (out["image_width"], out["image_height"]) == tiny_image.size
    assert out["detections"] == [EXPECTED_STUB_DETECTION]
    assert out["inference_ms"] >= 0


def test_unknown_model_raises(stub_models, tiny_image):
    with pytest.raises(ValueError, match="Unknown model"):
        predict(stub_models, "resnet", tiny_image)


# The tests below load the real weights and skip when they are absent.


def test_both_models_load_with_same_classes(real_models):
    assert set(real_models) == set(MODEL_FILES)
    assert real_models["synthetic"].names == real_models["finetuned"].names
    assert len(real_models["synthetic"].names) == 12


@pytest.mark.parametrize("name", ["synthetic", "finetuned"])
def test_real_result_shape(real_models, tiny_image, name):
    out = predict(real_models, name, tiny_image, conf=0.05)
    assert out["model"] == name
    assert out["inference_ms"] > 0
    for det in out["detections"]:
        assert det["class_name"] in real_models[name].names.values()
        assert 0.0 <= det["confidence"] <= 1.0


def test_conf_of_one_returns_nothing(real_models, tiny_image):
    assert predict(real_models, "synthetic", tiny_image, conf=1.0)["detections"] == []
