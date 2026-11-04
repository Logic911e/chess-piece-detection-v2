"""Tests for position.py. Pure geometry, so no weights or images are needed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from position import (
    board_homography,
    detections_to_position,
    fen_placement,
    image_to_board,
    square_name,
)

# A board seen at an angle: a1 bottom left, h8 top right, with perspective.
CORNERS = {
    "bottom_left": [200.0, 900.0],
    "bottom_right": [1000.0, 880.0],
    "top_right": [850.0, 300.0],
    "top_left": [350.0, 310.0],
}


def square_centre_in_image(file_index: int, rank_index: int):
    """Where the centre of a square appears in the image."""
    board_to_image = cv2.getPerspectiveTransform(
        np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32),
        np.array(
            [CORNERS[k] for k in ("bottom_left", "bottom_right", "top_right", "top_left")],
            dtype=np.float32,
        ),
    )
    pt = np.array([[[file_index + 0.5, rank_index + 0.5]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(pt, board_to_image)[0, 0]
    return float(x), float(y)


def det(name: str, point: tuple[float, float], conf: float = 0.9) -> dict:
    """A detection whose box has its bottom edge at `point`."""
    x, y = point
    return {"class_name": name, "confidence": conf, "bbox": [x - 10, y - 40, x + 10, y]}


def test_corners_map_to_board_corners():
    h = board_homography(CORNERS)
    assert image_to_board(h, tuple(CORNERS["bottom_left"])) == pytest.approx((0, 0), abs=1e-3)
    assert image_to_board(h, tuple(CORNERS["bottom_right"])) == pytest.approx((8, 0), abs=1e-3)
    assert image_to_board(h, tuple(CORNERS["top_right"])) == pytest.approx((8, 8), abs=1e-3)
    assert image_to_board(h, tuple(CORNERS["top_left"])) == pytest.approx((0, 8), abs=1e-3)


def test_missing_corner_is_an_error():
    with pytest.raises(ValueError, match="top_left"):
        board_homography({k: v for k, v in CORNERS.items() if k != "top_left"})


def test_square_names():
    assert square_name(0.1, 0.1) == "a1"
    assert square_name(7.9, 7.9) == "h8"
    assert square_name(4.5, 3.5) == "e4"
    assert square_name(-0.1, 2) is None
    assert square_name(8.0, 2) is None


def test_every_square_is_recovered():
    expected = {}
    detections = []
    for f, file in enumerate("abcdefgh"):
        for r in range(8):
            expected[f"{file}{r + 1}"] = "white-pawn"
            detections.append(det("white-pawn", square_centre_in_image(f, r)))
    assert detections_to_position(detections, CORNERS, anchor_y=1.0) == expected


def test_orientation_is_fixed_by_corner_names():
    detections = [
        det("white-rook", square_centre_in_image(0, 0)),
        det("black-king", square_centre_in_image(4, 7)),
    ]
    position = detections_to_position(detections, CORNERS, anchor_y=1.0)
    assert position == {"a1": "white-rook", "e8": "black-king"}


def test_off_board_detection_is_dropped():
    assert detections_to_position([det("white-pawn", (50.0, 50.0))], CORNERS) == {}


def test_more_confident_detection_wins_a_shared_square():
    spot = square_centre_in_image(3, 3)
    detections = [det("white-queen", spot, 0.6), det("black-queen", spot, 0.8)]
    assert detections_to_position(detections, CORNERS, anchor_y=1.0) == {"d4": "black-queen"}


def test_min_conf_filters():
    spot = square_centre_in_image(3, 3)
    assert detections_to_position([det("white-queen", spot, 0.3)], CORNERS, min_conf=0.5) == {}


def test_fen_placement():
    start = {}
    back = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
    for i, file in enumerate("abcdefgh"):
        start[f"{file}1"] = f"white-{back[i]}"
        start[f"{file}2"] = "white-pawn"
        start[f"{file}7"] = "black-pawn"
        start[f"{file}8"] = f"black-{back[i]}"
    assert fen_placement(start) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert fen_placement({}) == "8/8/8/8/8/8/8/8"
    assert fen_placement({"e4": "white-pawn", "e5": "black-pawn"}) == "8/8/8/4p3/4P3/8/8/8"
