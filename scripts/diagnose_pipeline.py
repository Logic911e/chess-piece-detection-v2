"""Diagnostics for the automatic photo-to-position pipeline, from the cached detections.

Answers four questions, on the validation and test splits, for the corner model that is in use:
  1. How does the exact rate differ between games? (each split is only three games)
  2. When the position is wrong, is it a turn of the board, a shift, or wrong pieces?
  3. Which way round is the board in the photos? (does ChessReD test orientation at all?)
  4. How far do correctly placed pieces sit from their square centres, with the annotated
     corners? That spread includes noise in the annotations themselves.

Needs the caches written by eval_pipeline.py. Writes reports/pipeline_diagnostics.csv.

    python scripts/diagnose_pipeline.py --data ../data/raw/chessred \\
        --weights build/runs/corners-n-nomosaic/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pipeline import ANCHOR_Y, CONF, build, corner_error_pct  # noqa: E402
from eval_position import load_ground_truth  # noqa: E402
from prepare_corner_data import screen_order  # noqa: E402

from position import (  # noqa: E402
    CORNER_ORDER,  # noqa: E402
    anchor_point,
    board_homography,
    detections_to_position,
    image_to_board,
)

PLACED_WITHIN = 6.0  # worst corner within this % of the board side counts as on the board
PRIOR = 3.0  # the refinement strength chosen on validation
CORNER_SCREEN = ["top-left", "top-right", "bottom-right", "bottom-left"]


def move(position: dict[str, str], fn) -> dict[str, str]:
    """The position with every square sent through fn(file, rank) -> (file, rank) or None."""
    out = {}
    for sq, piece in position.items():
        target = fn("abcdefgh".index(sq[0]), int(sq[1]) - 1)
        if target is not None and 0 <= target[0] < 8 and 0 <= target[1] < 8:
            out["abcdefgh"[target[0]] + str(target[1] + 1)] = piece
    return out


TURNS = {
    "half turn": lambda f, r: (7 - f, 7 - r),
    "quarter turn": lambda f, r: (7 - r, f),
    "quarter turn the other way": lambda f, r: (r, 7 - f),
}


def shifter(df: int, dr: int):
    return lambda f, r: (f + df, r + dr)


SHIFTS = {
    f"shift by one square ({df:+d},{dr:+d})": shifter(df, dr)
    for df in (-1, 0, 1)
    for dr in (-1, 0, 1)
    if (df, dr) != (0, 0)
}


def classify(pred: dict[str, str], truth: dict[str, str]) -> str:
    if pred == truth:
        return "exact"
    for label, fn in {**TURNS, **SHIFTS}.items():
        if move(pred, fn) == truth:
            return label.split(" (")[0]  # all eight shifts count together
    wrong = len({*pred, *truth} - {sq for sq in pred if truth.get(sq) == pred[sq]})
    return "1 to 2 squares wrong" if wrong <= 2 else "3 or more squares wrong"


def placement(found: dict, truth_corners: dict) -> str:
    """Whether the found corners sit on the board, and whether they carry the right names.
    The error is taken against the best of the four ways of naming the annotated corners, so
    a well-placed but wrongly named board is not mistaken for a badly placed one."""
    errors = []
    for turn in range(4):
        names = CORNER_ORDER[turn:] + CORNER_ORDER[:turn]
        renamed = {n: truth_corners[m] for n, m in zip(CORNER_ORDER, names, strict=True)}
        errors.append(corner_error_pct(found, renamed))
    best = min(range(4), key=errors.__getitem__)
    if errors[best] >= PLACED_WITHIN:
        return "corners misplaced"
    return "placed, wrong name" if best else "placed, right name"


def a1_screen_position(truth_corners: dict) -> str:
    """Where a1 sits in the photo, as one of the four screen corners."""
    ring = screen_order(truth_corners)
    return CORNER_SCREEN[ring.index(truth_corners["bottom_left"])]


def offsets(items, detections) -> list[tuple[float, float]]:
    """(dx, dy) in squares from the centre of the right square, for correctly placed pieces
    using the annotated corners."""
    out = []
    for image_id, _, corners, truth in items:
        homography = board_homography(corners)
        for det in detections[image_id]:
            if det["confidence"] < CONF:
                continue
            x, y = image_to_board(homography, anchor_point(det["bbox"], ANCHOR_Y))
            square = "abcdefgh"[min(7, max(0, int(x)))] + str(min(7, max(0, int(y))) + 1)
            if 0 <= x < 8 and 0 <= y < 8 and truth.get(square) == det["class_name"]:
                out.append((x - int(x) - 0.5, y - int(y) - 0.5))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/pipeline_diagnostics.csv")
    args = ap.parse_args()

    tag = args.weights.parent.parent.name
    ann = json.loads((args.data / "annotations.json").read_text())
    game = {i["id"]: i["game_id"] for i in ann["images"]}
    rows: list[dict] = []

    def add(split, section, key, value, n=""):
        rows.append({"split": split, "section": section, "key": key, "value": value, "images": n})
        print(f"{split:5s} {section:22s} {key:42s} {value}  {n}")

    for split in ("val", "test"):
        items = list(load_ground_truth(args.data, split))
        detections = {
            int(k): v
            for k, v in json.loads(
                (ROOT / f"build/position_eval/finetuned_{split}.json").read_text()
            ).items()
        }
        screen = {
            int(k): v
            for k, v in json.loads(
                (ROOT / f"build/corner_eval/{tag}_{split}.json").read_text()
            ).items()
        }
        located = build(items, detections, screen, PRIOR, True)

        failures: Counter[str] = Counter()
        placed: Counter[str] = Counter()
        wrong_with_annotated: Counter[str] = Counter()
        per_game: dict[int, list[tuple[bool, bool]]] = {}
        for image_id, _, truth_corners, truth in items:
            given = detections_to_position(detections[image_id], truth_corners, ANCHOR_Y, CONF)
            found = located.get(image_id)
            pred = (
                detections_to_position(detections[image_id], found, ANCHOR_Y, CONF) if found else {}
            )
            kind = classify(pred, truth) if found else "no board found"
            failures[kind] += 1
            if found and kind != "exact":
                placed[placement(found, truth_corners)] += 1
            if kind != "exact":
                ok = "right" if given == truth else "also wrong"
                wrong_with_annotated[f"{ok} with annotated corners"] += 1
            per_game.setdefault(game[image_id], []).append((given == truth, kind == "exact"))

        for g, results in sorted(per_game.items()):
            n = len(results)
            add(
                split,
                "exact by game",
                f"game {g}, annotated corners",
                f"{sum(a for a, _ in results) / n:.4f}",
                n,
            )
            add(
                split,
                "exact by game",
                f"game {g}, corners found",
                f"{sum(b for _, b in results) / n:.4f}",
                n,
            )
        for kind, count in failures.most_common():
            add(split, "what the result is", kind, f"{count / len(items):.4f}", count)

        for key, count in wrong_with_annotated.items():
            add(
                split,
                "of the wrong ones",
                key,
                f"{count / sum(wrong_with_annotated.values()):.4f}",
                count,
            )
        for key, count in placed.most_common():
            add(split, "wrong ones: corners", key, f"{count / sum(placed.values()):.4f}", count)

        where = Counter(a1_screen_position(c) for _, _, c, _ in items)
        for pos in CORNER_SCREEN:
            add(
                split, "where a1 is in the photo", pos, f"{where[pos] / len(items):.4f}", where[pos]
            )
        pieces = [len(t) for *_, t in items]
        add(
            split,
            "pieces on the board",
            "median, min",
            f"{statistics.median(pieces)}, {min(pieces)}",
        )

        off = offsets(items, detections)
        for axis, idx in (("across the file", 0), ("along the rank", 1)):
            vals = sorted(o[idx] for o in off)
            q = lambda p, vals=vals: vals[int(p * (len(vals) - 1))]  # noqa: E731
            add(
                split,
                "anchor offset in squares",
                axis,
                f"median {statistics.median(vals):+.3f}, 10th {q(0.1):+.3f}, 90th {q(0.9):+.3f}",
                len(vals),
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
