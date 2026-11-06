"""Turn piece detections and the four board corners into a chess position.

The corners are the corners of the 8x8 playing area, named by the square they
sit on: a1 is `bottom_left`, h1 `bottom_right`, h8 `top_right`, a8 `top_left`
(the naming ChessReD uses). A perspective transform (homography) maps image
pixels to board coordinates, where x runs over files a to h and y over ranks
1 to 8, one unit per square.

Like predict.py, this module has no FastAPI or config imports.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

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


def parse_corners(raw: str) -> dict[str, list[float]]:
    """Read corners sent as JSON text, e.g. {"bottom_left": [x, y], ...}.

    Raises ValueError with a message meant for the user if the corners are
    malformed or cannot be the corners of a board seen in a photo."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("Corners must be JSON text") from None
    if not isinstance(data, dict) or set(data) != set(CORNER_ORDER):
        raise ValueError(f"Corners must have exactly these keys: {', '.join(CORNER_ORDER)}")

    corners: dict[str, list[float]] = {}
    for name in CORNER_ORDER:
        point = data[name]
        ok = (
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(v, int | float) and not isinstance(v, bool) for v in point)
            and all(math.isfinite(v) for v in point)
        )
        if not ok:
            raise ValueError(f"Corner '{name}' must be two finite numbers [x, y]")
        corners[name] = [float(point[0]), float(point[1])]

    # Walking the corners in order must always turn the same way, otherwise the
    # shape is twisted (two corners swapped) or flat, and no board looks like that.
    pts = [corners[name] for name in CORNER_ORDER]
    turns = []
    for i in range(4):
        (ax, ay), (bx, by), (cx, cy) = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        turns.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        raise ValueError(
            "The four corners do not form a board shape. Check that each corner is "
            "on the right square (a1, h1, h8, a8)."
        )
    return corners


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


def square_contrast(gray: np.ndarray, corners: dict[str, list[float]], skip: set[str]) -> float:
    """Mean brightness of the light squares minus the dark ones, for squares not in skip.

    a1 is a dark square, so this is clearly positive when the corners are named
    correctly and clearly negative when the board is turned a quarter turn."""
    inverse = np.linalg.inv(board_homography(corners))
    squares = [
        (file, rank)
        for file in range(8)
        for rank in range(8)
        if f"{FILES[file]}{rank + 1}" not in skip
    ]
    centres = np.array([[[file + 0.5, rank + 0.5]] for file, rank in squares], dtype=np.float32)
    pixels = cv2.perspectiveTransform(centres, inverse.astype(np.float32))[:, 0]
    light: list[float] = []
    dark: list[float] = []
    for (file, rank), (x, y) in zip(squares, pixels, strict=True):
        col, row = int(round(x)), int(round(y))
        patch = gray[max(0, row - 4) : row + 5, max(0, col - 4) : col + 5]
        if patch.size:
            (light if (file + rank) % 2 else dark).append(float(patch.mean()))
    if not light or not dark:
        return 0.0
    return float(np.mean(light) - np.mean(dark))


def orient_corners(
    screen_corners: list[list[float]],
    detections: list[dict],
    anchor_y: float = 0.9,
    min_conf: float = 0.0,
    gray: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Name four corners given in screen order (top-left, top-right, bottom-right,
    bottom-left) as a1, h1, h8, a8.

    Two cues decide, because an empty board looks the same from every side.
    The checkerboard (when the grayscale photo is given) separates quarter turns:
    a1 is dark, so only two of the four namings put dark squares in the right
    places. The pieces then separate the two that remain, which differ by a half
    turn: the naming that puts most white pieces on ranks 1 to 4 and black
    pieces on ranks 5 to 8 wins. Walking a1, h1, h8, a8 turns the opposite way
    to top-left, top-right, bottom-right, bottom-left, so the ring is reversed first."""
    top_left, top_right, bottom_right, bottom_left = screen_corners
    ring = [bottom_left, bottom_right, top_right, top_left]
    candidates = [
        dict(zip(CORNER_ORDER, ring[shift:] + ring[:shift], strict=True)) for shift in range(4)
    ]
    positions = [detections_to_position(detections, c, anchor_y, min_conf) for c in candidates]

    allowed = list(range(4))
    if gray is not None:
        contrast = [
            square_contrast(gray, c, set(pos)) for c, pos in zip(candidates, positions, strict=True)
        ]
        # The two namings with the checkerboard the right way round.
        allowed = sorted(range(4), key=lambda i: contrast[i], reverse=True)[:2]

    def pieces_score(position: dict[str, str]) -> int:
        return sum(
            1 if (int(square[1]) <= 4) == name.startswith("white") else -1
            for square, name in position.items()
        )

    return candidates[max(allowed, key=lambda i: pieces_score(positions[i]))]


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 matrix taking four src points to four dst points, in float64.

    cv2.getPerspectiveTransform works in float32, which is too coarse for the small
    steps the optimiser takes on corners a few thousand pixels from the origin."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    h = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    return np.append(h, 1.0).reshape(3, 3)


def _to_board(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def refine_corners(
    corners: dict[str, list[float]],
    detections: list[dict],
    anchor_y: float = 0.9,
    min_conf: float = 0.5,
    prior_weight: float = 10.0,
    min_pieces: int = 6,
    rounds: int = 3,
) -> dict[str, list[float]]:
    """Nudge the corners so each piece sits nearer the middle of its square.

    Each round gives every piece the square it lands on now, then moves the four
    corners to shrink the distance between each piece's base and that square's
    centre. This only helps when the starting corners are within about half a
    square: shifted by a whole square the pieces fit the grid just as well, so
    the starting guess decides which square each piece belongs to.

    prior_weight pulls the corners back towards where they started, so a few badly
    placed pieces cannot drag the board away. Returns the corners unchanged if
    there are too few pieces to constrain them."""
    anchors = np.array(
        [anchor_point(d["bbox"], anchor_y) for d in detections if d["confidence"] >= min_conf],
        dtype=np.float64,
    )
    if len(anchors) < min_pieces:
        return corners

    start = np.array([corners[name] for name in CORNER_ORDER], dtype=np.float64)
    side = float(np.mean([np.linalg.norm(start[i] - start[(i + 1) % 4]) for i in range(4)]))
    board = _BOARD_POINTS.astype(np.float64)
    current = start.copy()

    for _ in range(rounds):
        now = _to_board(_homography(current, board), anchors)
        near = (now > -0.5).all(axis=1) & (now < 8.5).all(axis=1)
        if near.sum() < min_pieces:
            break
        points = anchors[near]
        centres = np.clip(np.floor(now[near]) + 0.5, 0.5, 7.5)
        base = current

        def residuals(
            offset: np.ndarray,
            base: np.ndarray = base,
            points: np.ndarray = points,
            centres: np.ndarray = centres,
        ) -> np.ndarray:
            moved = base + offset.reshape(4, 2) * side
            fit = (_to_board(_homography(moved, board), points) - centres).ravel()
            drift = prior_weight * ((moved - start) / side).ravel()
            return np.concatenate([fit, drift])

        try:
            result = least_squares(
                residuals, np.zeros(8), loss="soft_l1", f_scale=0.25, diff_step=1e-6
            )
        except np.linalg.LinAlgError:
            break
        current = base + result.x.reshape(4, 2) * side

    return {name: [float(x), float(y)] for name, (x, y) in zip(CORNER_ORDER, current, strict=True)}


def locate_board(
    screen_corners: list[list[float]],
    detections: list[dict],
    gray: np.ndarray,
    anchor_y: float = 0.9,
    min_conf: float = 0.5,
    prior_weight: float = 10.0,
) -> dict[str, list[float]]:
    """Four detected corners in screen order to named, refined corners.

    Refining first matters: the checkerboard test that names a1 only works when the
    corners are within about a third of a square, and refinement needs no names,
    because a square grid looks the same after a quarter turn."""
    top_left, top_right, bottom_right, bottom_left = screen_corners
    unnamed = dict(zip(CORNER_ORDER, [bottom_left, bottom_right, top_right, top_left], strict=True))
    refined = refine_corners(unnamed, detections, anchor_y, min_conf, prior_weight)
    a1, h1, h8, a8 = (refined[name] for name in CORNER_ORDER)
    return orient_corners([a8, h8, h1, a1], detections, anchor_y, min_conf, gray)
