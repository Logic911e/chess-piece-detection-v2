"""API tests. TestClient runs the real app, including the startup hook.

The main `client` swaps model loading for stubs, so these run without the
gitignored weights. `real_client` at the bottom uses the real weights.
"""

from __future__ import annotations

import io

import pytest
from conftest import EXPECTED_STUB_DETECTION, StubModel
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            app_module,
            "load_models",
            lambda _dir: {"synthetic": StubModel(), "finetuned": StubModel()},
        )
        with TestClient(app_module.app) as c:  # `with` triggers the startup hook
            yield c


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def post(client, data: bytes, content_type="image/png", **params):
    return client.post("/predict", params=params, files={"file": ("x.png", data, content_type)})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models": ["synthetic", "finetuned"]}


@pytest.mark.parametrize("model", ["synthetic", "finetuned"])
def test_predict_ok(client, tiny_image, model):
    r = post(client, png_bytes(tiny_image), model=model, conf=0.1)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == model
    assert (body["image_width"], body["image_height"]) == tiny_image.size
    assert body["detections"] == [EXPECTED_STUB_DETECTION]


def test_predict_defaults(client, tiny_image):
    r = post(client, png_bytes(tiny_image))
    assert r.status_code == 200
    assert r.json()["model"] == "finetuned"


def test_wrong_content_type(client):
    assert post(client, b"hello", content_type="text/plain").status_code == 415


def test_oversize_upload(client):
    assert post(client, b"0" * (app_module.MAX_UPLOAD_BYTES + 1)).status_code == 413


def test_corrupt_image(client):
    r = post(client, b"not really a png")
    assert r.status_code == 400
    assert "traceback" not in r.text.lower()


@pytest.mark.parametrize("params", [{"model": "resnet"}, {"conf": 1.5}, {"conf": -0.1}])
def test_bad_query_params(client, tiny_image, params):
    assert post(client, png_bytes(tiny_image), **params).status_code == 422


def test_missing_file(client):
    assert client.post("/predict").status_code == 422


@pytest.fixture()
def real_client(real_models):  # skips when the weights are absent
    with TestClient(app_module.app) as c:
        yield c


@pytest.mark.parametrize("model", ["synthetic", "finetuned"])
def test_real_weights_end_to_end(real_client, tiny_image, model):
    r = post(real_client, png_bytes(tiny_image), model=model, conf=0.1)
    assert r.status_code == 200
    assert r.json()["model"] == model
