"""API tests. TestClient runs the real app, including the startup hook.

The main `client` swaps model loading for stubs, so these run without the
gitignored weights. `real_client` at the bottom uses the real weights.
"""

from __future__ import annotations

import io
import json

import pytest
from conftest import EXPECTED_STUB_DETECTION, StubModel
from fastapi.testclient import TestClient
from PIL import Image

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


# --- /position -------------------------------------------------------------

# Corners of a square board filling most of the 320x240 test image. The stub
# model's one detection (a white queen) stands on b2 for these corners.
BOARD_CORNERS = {
    "bottom_left": [20, 230],
    "bottom_right": [260, 230],
    "top_right": [260, 10],
    "top_left": [20, 10],
}


def post_position(client, data: bytes, corners=BOARD_CORNERS, content_type="image/png", **params):
    form = {"corners": corners if isinstance(corners, str) else json.dumps(corners)}
    return client.post(
        "/position",
        params=params,
        data=form,
        files={"file": ("x.png", data, content_type)},
    )


def test_position_ok(client, tiny_image):
    r = post_position(client, png_bytes(tiny_image), conf=0.1)
    assert r.status_code == 200
    body = r.json()
    assert body["squares"] == {"b2": "white-queen"}
    assert body["fen_placement"] == "8/8/8/8/8/8/1Q6/8"
    assert body["pieces"] == 1
    assert body["ignored_detections"] == 0
    assert (body["image_width"], body["image_height"]) == tiny_image.size


def test_position_ignores_detections_outside_the_corners(client, tiny_image):
    # Same board, but moved so the queen's base falls outside it.
    away = {k: [x + 200, y] for k, (x, y) in BOARD_CORNERS.items()}
    r = post_position(client, png_bytes(tiny_image), corners=away, conf=0.1)
    assert r.status_code == 200
    body = r.json()
    assert body["squares"] == {}
    assert body["fen_placement"] == "8/8/8/8/8/8/8/8"
    assert body["ignored_detections"] == 1


@pytest.mark.parametrize(
    "corners",
    [
        "not json",
        json.dumps({"bottom_left": [0, 0]}),
        json.dumps(
            {
                **BOARD_CORNERS,
                "top_left": BOARD_CORNERS["top_right"],
                "top_right": BOARD_CORNERS["top_left"],
            }
        ),
    ],
)
def test_position_rejects_bad_corners(client, tiny_image, corners):
    r = post_position(client, png_bytes(tiny_image), corners=corners)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)


def test_position_requires_corners(client, tiny_image):
    r = client.post("/position", files={"file": ("x.png", png_bytes(tiny_image), "image/png")})
    assert r.status_code == 422


def test_position_rejects_wrong_file_type(client):
    r = post_position(client, b"hello", content_type="text/plain")
    assert r.status_code == 415


def test_position_rejects_bad_anchor(client, tiny_image):
    r = post_position(client, png_bytes(tiny_image), anchor_y=0.1)
    assert r.status_code == 422


def test_exif_rotation_is_applied(client, tiny_image):
    """A phone photo tagged 'rotate 90 degrees' must be read as the browser shows it."""
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 270 to display
    buf = io.BytesIO()
    tiny_image.save(buf, format="JPEG", exif=exif)
    r = post(client, buf.getvalue(), content_type="image/jpeg")
    assert r.status_code == 200
    width, height = tiny_image.size
    assert (r.json()["image_width"], r.json()["image_height"]) == (height, width)
