"""Second stage of corner finding: cut a square crop around the board, so a model can look at
the board at a higher resolution than the whole photo allows."""

from __future__ import annotations

MARGIN = 0.15  # crop margin around the board's bounding box, as a fraction of its larger side


def crop_square(
    corners: list[list[float]],
    margin: float = MARGIN,
    shift: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """(left, top, side) of a square that holds the corners plus a margin.

    `shift` (in units of the crop side) and `scale` move and resize it; training uses them to
    imitate the first stage's errors."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * margin) * scale
    cx = (min(xs) + max(xs)) / 2 + shift[0] * side
    cy = (min(ys) + max(ys)) / 2 + shift[1] * side
    return cx - side / 2, cy - side / 2, side


def to_crop(
    points: list[list[float]], box: tuple[float, float, float], size: int
) -> list[list[float]]:
    """Photo pixels to crop pixels, for a crop resized to size x size."""
    left, top, side = box
    f = size / side
    return [[(x - left) * f, (y - top) * f] for x, y in points]


def from_crop(
    points: list[list[float]], box: tuple[float, float, float], size: int
) -> list[list[float]]:
    """Crop pixels back to photo pixels."""
    left, top, side = box
    f = side / size
    return [[x * f + left, y * f + top] for x, y in points]
