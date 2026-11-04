"""Turn piece detections and the four board corners into a chess position.

The corners are the corners of the 8x8 playing area, named by the square they
sit on: a1 is `bottom_left`, h1 `bottom_right`, h8 `top_right`, a8 `top_left`
(the naming ChessReD uses). A perspective transform (homography) maps image
pixels to board coordinates, where x runs over files a to h and y over ranks
1 to 8, one unit per square.

Like predict.py, this module has no FastAPI or config imports.
"""

from __future__ import annotations

import cv2
import numpy as np

FILES = "abcdefgh"

# Corner names in the order of the board points they map to: a1, h1, h8, a8.
CORNER_ORDER = ("bottom_left", "bottom_right", "top_right", "top_left")
_BOARD_POINTS = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)

FEN_LETTER = {
    "pawn": "p",
    "knight": "n",
    "bishop": "b",
    "rook": "r",
    "queen": "q",
    "king": "k",
}


def board_homography(corners: dict[str, list[float]]) -> np.ndarray:
    """3x3 matrix taking image pixels to board coordinates."""
    missing = [name for name in CORNER_ORDER if name not in corners]
    if missing:
        raise ValueError(f"Missing corners: {missing}")
    image_points = np.array([corners[name] for name in CORNER_ORDER], dtype=np.float32)
    return cv2.getPerspectiveTransform(image_points, _BOARD_POINTS)


def image_to_board(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Map one (x, y) image point to board coordinates."""
    mapped = cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), homography)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def square_name(x: float, y: float) -> str | None:
    """Board coordinates to a name like 'e4', or None if off the board."""
    if not (0 <= x < 8 and 0 <= y < 8):
        return None
    return f"{FILES[int(x)]}{int(y) + 1}"


def anchor_point(bbox: list[float], anchor_y: float) -> tuple[float, float]:
    """The point of a box taken to be where the piece stands.

    x is the box centre. anchor_y runs from 0 (top edge) to 1 (bottom edge).
    The base of a piece is near the bottom of its box, but a tall piece also
    leans over the square behind it, so the centre of the box would land on
    the wrong square."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, y1 + anchor_y * (y2 - y1)


def detections_to_position(
    detections: list[dict],
    corners: dict[str, list[float]],
    anchor_y: float = 0.9,
    min_conf: float = 0.0,
) -> dict[str, str]:
    """Map detections to {square: class_name}.

    A square holds one piece, so if two detections land on the same square the
    more confident one wins. Detections that fall off the board are dropped."""
    homography = board_homography(corners)
    best: dict[str, tuple[float, str]] = {}
    for det in detections:
        if det["confidence"] < min_conf:
            continue
        x, y = image_to_board(homography, anchor_point(det["bbox"], anchor_y))
        square = square_name(x, y)
        if square is None:
            continue
        if square not in best or det["confidence"] > best[square][0]:
            best[square] = (det["confidence"], det["class_name"])
    return {square: name for square, (_, name) in best.items()}


def fen_placement(position: dict[str, str]) -> str:
    """The piece-placement field of a FEN string (ranks 8 to 1).

    A full FEN also needs side to move, castling rights and move counters,
    none of which a photo can tell us, so only this field is produced."""
    ranks = []
    for rank in range(8, 0, -1):
        row = ""
        empty = 0
        for file in FILES:
            name = position.get(f"{file}{rank}")
            if name is None:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            colour, piece = name.split("-")
            letter = FEN_LETTER[piece]
            row += letter.upper() if colour == "white" else letter
        if empty:
            row += str(empty)
        ranks.append(row)
    return "/".join(ranks)
