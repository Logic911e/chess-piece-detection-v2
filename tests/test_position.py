"""Tests for position.py. Pure geometry, so no weights or images are needed."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from position import (
    CORNER_ORDER,
    alignment_error,
    board_homography,
    detections_to_position,
    fen_placement,
    image_to_board,
    locate_board,
    locate_board_multistart,
    orient_corners,
    parse_corners,
    refine_corners,
    square_contrast,
    square_name,
    start_points,
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


# --- parse_corners ---------------------------------------------------------

GOOD_JSON = json.dumps(CORNERS)


def test_parse_corners_accepts_a_good_board():
    assert parse_corners(GOOD_JSON) == CORNERS


def test_parse_corners_accepts_either_winding_direction():
    # Clockwise on screen (the normal case) and mirrored (a flipped photo).
    mirrored = {
        "bottom_left": CORNERS["bottom_right"],
        "bottom_right": CORNERS["bottom_left"],
        "top_right": CORNERS["top_left"],
        "top_left": CORNERS["top_right"],
    }
    assert parse_corners(json.dumps(mirrored)) == mirrored


@pytest.mark.parametrize(
    "raw, message",
    [
        ("not json", "JSON"),
        ("[1, 2]", "exactly these keys"),
        (json.dumps({"bottom_left": [0, 0]}), "exactly these keys"),
        (json.dumps({**CORNERS, "top_left": [1]}), "two finite numbers"),
        (json.dumps({**CORNERS, "top_left": ["a", "b"]}), "two finite numbers"),
        (json.dumps({**CORNERS, "top_left": [True, False]}), "two finite numbers"),
        (json.dumps({**CORNERS, "extra": [1, 2]}), "exactly these keys"),
    ],
)
def test_parse_corners_rejects_malformed_input(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_corners(raw)


def test_parse_corners_rejects_non_finite_numbers():
    # json.loads accepts the non-standard NaN token, so it must be caught here.
    raw = GOOD_JSON.replace("200.0", "NaN", 1)
    with pytest.raises(ValueError, match="two finite numbers"):
        parse_corners(raw)


def test_parse_corners_rejects_a_twisted_shape():
    twisted = {**CORNERS, "top_left": CORNERS["top_right"], "top_right": CORNERS["top_left"]}
    with pytest.raises(ValueError, match="do not form a board shape"):
        parse_corners(json.dumps(twisted))


def test_parse_corners_rejects_a_flat_shape():
    flat = {k: [float(i * 100), 50.0] for i, k in enumerate(CORNERS)}
    with pytest.raises(ValueError, match="do not form a board shape"):
        parse_corners(json.dumps(flat))


def opening_detections() -> list[dict]:
    """Both sides' pieces on their starting ranks, each box ending at its square centre."""
    out = []
    for file_index in range(8):
        for rank_index, colour in ((0, "white"), (1, "white"), (6, "black"), (7, "black")):
            out.append(det(f"{colour}-pawn", square_centre_in_image(file_index, rank_index)))
    return out


def checkerboard_image() -> np.ndarray:
    """A grayscale photo of the board in CORNERS: a1 dark, 100 px per square before warping."""
    top_down = np.zeros((800, 800), dtype=np.uint8)
    for file_index in range(8):
        for rank_index in range(8):
            shade = 40 if (file_index + rank_index) % 2 == 0 else 220
            top_down[
                rank_index * 100 : (rank_index + 1) * 100, file_index * 100 : (file_index + 1) * 100
            ] = shade
    to_image = cv2.getPerspectiveTransform(
        np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32),
        np.array([CORNERS[name] for name in CORNER_ORDER], dtype=np.float32),
    )
    return cv2.warpPerspective(top_down, to_image, (1100, 1000), flags=cv2.INTER_NEAREST)


def screen_order(shift: int) -> list[list[float]]:
    """CORNERS as top-left, top-right, bottom-right, bottom-left, cyclically shifted, as a
    detector that cannot tell which corner is which might report them."""
    ring = [
        CORNERS["top_left"],
        CORNERS["top_right"],
        CORNERS["bottom_right"],
        CORNERS["bottom_left"],
    ]
    return ring[shift:] + ring[:shift]


def max_corner_error(found: dict, expected: dict = CORNERS) -> float:
    return max(float(np.hypot(*np.subtract(found[n], expected[n]))) for n in CORNER_ORDER)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_pieces_name_the_corners_whatever_the_starting_corner(shift):
    corners = orient_corners(screen_order(shift), opening_detections(), anchor_y=1.0)
    assert max_corner_error(corners) < 1e-3


def test_checkerboard_is_light_minus_dark_when_a1_is_dark():
    gray = checkerboard_image()
    assert square_contrast(gray, CORNERS, set()) > 100
    turned = {name: CORNERS[CORNER_ORDER[(i + 1) % 4]] for i, name in enumerate(CORNER_ORDER)}
    assert square_contrast(gray, turned, set()) < -100


def test_checkerboard_separates_quarter_turns_when_pieces_cannot():
    # One lone white pawn on rank 1 gives the pieces no way to tell the sides apart.
    lone = [det("white-pawn", square_centre_in_image(3, 0))]
    gray = checkerboard_image()
    for shift in range(4):
        found = orient_corners(screen_order(shift), lone, anchor_y=1.0, gray=gray)
        # Right up to a half turn, which only the pieces can settle.
        half_turn = {n: CORNERS[CORNER_ORDER[(i + 2) % 4]] for i, n in enumerate(CORNER_ORDER)}
        assert min(max_corner_error(found), max_corner_error(found, half_turn)) < 1e-3


def test_refine_pulls_misplaced_corners_back():
    nudged = {
        "bottom_left": [CORNERS["bottom_left"][0] + 22, CORNERS["bottom_left"][1] - 18],
        "bottom_right": [CORNERS["bottom_right"][0] - 20, CORNERS["bottom_right"][1] + 15],
        "top_right": [CORNERS["top_right"][0] + 15, CORNERS["top_right"][1] + 20],
        "top_left": [CORNERS["top_left"][0] - 18, CORNERS["top_left"][1] - 16],
    }
    refined = refine_corners(nudged, opening_detections(), anchor_y=1.0, prior_weight=1.0)
    assert max_corner_error(refined) < 0.4 * max_corner_error(nudged)


def test_refine_leaves_corners_alone_with_too_few_pieces():
    few = [det("white-pawn", square_centre_in_image(0, 0))]
    assert refine_corners(CORNERS, few, anchor_y=1.0) == CORNERS


def test_locate_board_names_and_refines_detected_corners():
    ring = screen_order(0)
    detected = [
        [x + dx, y + dy]
        for (x, y), (dx, dy) in zip(ring, [(14, 12), (-12, 16), (15, -13), (-16, -14)], strict=True)
    ]
    found = locate_board(
        detected, opening_detections(), checkerboard_image(), anchor_y=1.0, prior_weight=1.0
    )
    assert max_corner_error(found) < 12


def test_start_points_begin_with_the_detected_corners():
    ring = screen_order(0)
    starts = start_points(ring)
    assert len(starts) == 11
    assert np.allclose(starts[0], ring)


def test_alignment_error_is_lower_for_corners_on_the_board():
    dets = opening_detections()
    on_board = alignment_error(dets, CORNERS, anchor_y=1.0)
    nudged = {k: [x + 40, y] for k, (x, y) in CORNERS.items()}
    assert on_board < alignment_error(dets, nudged, anchor_y=1.0)


def test_locate_board_multistart_names_and_refines_detected_corners():
    ring = screen_order(0)
    detected = [
        [x + dx, y + dy]
        for (x, y), (dx, dy) in zip(ring, [(14, 12), (-12, 16), (15, -13), (-16, -14)], strict=True)
    ]
    found = locate_board_multistart(
        detected, opening_detections(), checkerboard_image(), anchor_y=1.0, prior_weight=1.0
    )
    assert max_corner_error(found) < 12
