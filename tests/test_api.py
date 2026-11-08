"""API tests. TestClient runs the real app, including the startup hook.

The main `client` swaps model loading for stubs, so these run without the
gitignored weights. `real_client` at the bottom uses the real weights.
"""

from __future__ import annotations

import io
import json

import pytest
from conftest import EXPECTED_STUB_DETECTION, StubCornerModel, StubModel
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
        mp.setattr(app_module, "load_corner_model", lambda _dir: None)
        with TestClient(app_module.app) as c:  # `with` triggers the startup hook
            yield c


# Where a corner detector would report the BOARD_CORNERS board: top-left, top-right,
# bottom-right, bottom-left.
SCREEN_CORNERS = [[20, 10], [260, 10], [260, 230], [20, 230]]


def client_with_corner_model(corner_model):
    """A client whose corner detector is the given stub."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            app_module,
            "load_models",
            lambda _dir: {"synthetic": StubModel(), "finetuned": StubModel()},
        )
        mp.setattr(app_module, "load_corner_model", lambda _dir: corner_model)
        with TestClient(app_module.app) as c:
            yield c


@pytest.fixture()
def auto_client():
    yield from client_with_corner_model(StubCornerModel(SCREEN_CORNERS))


@pytest.fixture()
def blind_client():
    yield from client_with_corner_model(StubCornerModel(None))


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def post(client, data: bytes, content_type="image/png", **params):
    return client.post("/predict", params=params, files={"file": ("x.png", data, content_type)})


def test_health(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_engine", lambda: None)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "models": ["synthetic", "finetuned"],
        "auto_corners": False,
        "engine": False,
    }


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


def test_position_without_corners_needs_the_corner_model(client, tiny_image):
    r = client.post("/position", files={"file": ("x.png", png_bytes(tiny_image), "image/png")})
    assert r.status_code == 503
    assert "Send the corners" in r.json()["detail"]


def test_position_reports_the_corners_it_was_given(client, tiny_image):
    body = post_position(client, png_bytes(tiny_image), conf=0.1).json()
    assert body["corners_source"] == "given"
    assert body["corners"] == {k: [float(x), float(y)] for k, (x, y) in BOARD_CORNERS.items()}


def test_health_says_when_an_engine_is_available(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_engine", lambda: "/fake/engine")
    assert client.get("/health").json()["engine"] is True


def test_health_says_when_corners_can_be_found(auto_client):
    assert auto_client.get("/health").json()["auto_corners"] is True


def test_position_finds_the_corners_itself(auto_client, tiny_image):
    r = auto_client.post(
        "/position",
        params={"conf": 0.1},
        files={"file": ("x.png", png_bytes(tiny_image), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["corners_source"] == "auto"
    assert set(body["corners"]) == {"bottom_left", "bottom_right", "top_right", "top_left"}
    assert body["pieces"] == 1  # the stub's queen lands on the board


def test_position_with_no_board_found_asks_for_corners(blind_client, tiny_image):
    r = blind_client.post(
        "/position", files={"file": ("x.png", png_bytes(tiny_image), "image/png")}
    )
    assert r.status_code == 422
    assert "Mark the corners by hand" in r.json()["detail"]


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


# --- warnings, /analyse and the served pages --------------------------------


def test_position_warns_when_the_placement_is_impossible(client, tiny_image):
    # The stub finds one white queen: no kings, which a real board always has.
    body = post_position(client, png_bytes(tiny_image), conf=0.1).json()
    assert "White has no king." in body["warnings"]
    assert "Black has no king." in body["warnings"]


CANNED_ANALYSIS = {
    "engine": "Stub engine",
    "fen": "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
    "score_kind": "cp",
    "score_value": 500,
    "score_text": "+5.00",
    "best_move_uci": "h1h8",
    "best_move_san": "Rh8+",
    "line": "1. Rh8+",
    "depth": 12,
}


def post_analyse(client, fen: str, **extra):
    return client.post("/analyse", json={"fen": fen, **extra})


def test_analyse_ok(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_engine", lambda: "/fake/engine")
    monkeypatch.setattr(app_module, "analyse", lambda path, fen, seconds: CANNED_ANALYSIS)
    r = post_analyse(client, "4k3/8/8/8/8/8/8/4K2R w - - 0 1", seconds=0.5)
    assert r.status_code == 200
    assert r.json() == CANNED_ANALYSIS


def test_analyse_without_an_engine_is_503(client, monkeypatch):
    monkeypatch.setattr(app_module, "find_engine", lambda: None)
    r = post_analyse(client, "4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    assert r.status_code == 503
    assert "engine" in r.json()["detail"]


@pytest.mark.parametrize(
    "fen, expected",
    [
        ("not a fen", "not a valid FEN"),
        ("8/8/8/8/8/8/8/8 w - - 0 1", "cannot be analysed"),
        ("4k3/8/8/8/8/8/8/4RK2 w - - 0 1", "cannot be analysed"),
    ],
)
def test_analyse_rejects_bad_positions(client, monkeypatch, fen, expected):
    monkeypatch.setattr(app_module, "find_engine", lambda: "/fake/engine")
    r = post_analyse(client, fen)
    assert r.status_code == 422
    assert expected in r.json()["detail"]


def test_analyse_rejects_out_of_range_time(client):
    assert post_analyse(client, "4k3/8/8/8/8/8/8/4K2R w - - 0 1", seconds=60).status_code == 422


def test_pages_are_served_by_the_api(client):
    r = client.get("/app/position.html")
    assert r.status_code == 200
    assert "Board to position" in r.text
    redirect = client.get("/", follow_redirects=False)
    assert redirect.status_code in (301, 302, 307)
    assert redirect.headers["location"] == "/app/"


def test_too_many_pixels_is_413(client, tiny_image, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_PIXELS", 1000)  # the image is 320 x 240
    r = post(client, png_bytes(tiny_image))
    assert r.status_code == 413
    assert "pixels" in r.json()["detail"]


def test_image_at_the_pixel_limit_is_accepted(client, tiny_image, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_PIXELS", 320 * 240)
    assert post(client, png_bytes(tiny_image)).status_code == 200


def test_analyse_refuses_when_every_engine_slot_is_taken(client, monkeypatch):
    import threading

    monkeypatch.setattr(app_module, "find_engine", lambda: "/fake/engine")
    monkeypatch.setattr(app_module, "ENGINE_SLOTS", threading.BoundedSemaphore(1))
    app_module.ENGINE_SLOTS.acquire()  # someone else is using the only slot
    r = post_analyse(client, "4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    assert r.status_code == 429


def test_analyse_frees_its_slot_after_a_failure(client, monkeypatch):
    import threading

    def boom(path, fen, seconds):
        raise OSError("engine missing")

    monkeypatch.setattr(app_module, "find_engine", lambda: "/fake/engine")
    monkeypatch.setattr(app_module, "ENGINE_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(app_module, "analyse", boom)
    assert post_analyse(client, "4k3/8/8/8/8/8/8/4K2R w - - 0 1").status_code == 502
    assert app_module.ENGINE_SLOTS.acquire(blocking=False)  # the slot came back


def test_position_with_given_corners_has_no_confidence_fields(client, tiny_image):
    body = post_position(client, png_bytes(tiny_image), conf=0.1).json()
    assert body["alignment_error"] is None
    assert body["low_confidence"] is None


def test_position_with_found_corners_reports_alignment(auto_client, tiny_image):
    r = auto_client.post(
        "/position",
        params={"conf": 0.1},
        files={"file": ("x.png", png_bytes(tiny_image), "image/png")},
    )
    body = r.json()
    assert 0.0 <= body["alignment_error"] <= 0.5
    assert isinstance(body["low_confidence"], bool)


def test_low_confidence_follows_the_threshold(auto_client, tiny_image, monkeypatch):
    files = {"file": ("x.png", png_bytes(tiny_image), "image/png")}
    monkeypatch.setattr(app_module, "LOW_CONFIDENCE_ALIGNMENT", -1.0)
    assert auto_client.post("/position", params={"conf": 0.1}, files=files).json()["low_confidence"]
    monkeypatch.setattr(app_module, "LOW_CONFIDENCE_ALIGNMENT", 1.0)
    assert not auto_client.post("/position", params={"conf": 0.1}, files=files).json()[
        "low_confidence"
    ]
