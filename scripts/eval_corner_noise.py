"""How much does the position score drop when the board corners are imprecise?

Takes the detections cached by eval_position.py (so it needs no model), shifts
each of the four annotated corners by random noise, and scores the test split.
Noise is a standard deviation as a percentage of the board's mean side length
in the image, applied independently in x and y to each corner. A fixed list of
seeds is averaged. The anchor and confidence settings are the ones chosen on
the validation split without noise; they are not re-tuned here.

    python scripts/eval_corner_noise.py --data ../data/raw/chessred
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_position import load_ground_truth, score  # noqa: E402

from position import CORNER_ORDER  # noqa: E402

NOISE_PERCENT = [0, 0.5, 1, 2, 3, 5]
SEEDS = [0, 1, 2, 3, 4]


def board_size(corners: dict[str, list[float]]) -> float:
    """Mean length of the four sides of the board quadrilateral, in pixels."""
    points = [corners[name] for name in CORNER_ORDER]
    sides = [math.dist(points[i], points[(i + 1) % 4]) for i in range(4)]
    return sum(sides) / 4


def noisy(corners: dict[str, list[float]], percent: float, rng) -> dict[str, list[float]]:
    sigma = board_size(corners) * percent / 100
    return {
        name: [float(v + rng.normal(0, sigma)) for v in point] for name, point in corners.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="folder with annotations.json")
    parser.add_argument("--model", default="finetuned")
    parser.add_argument("--anchor", type=float, default=0.9)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--cache", type=Path, default=ROOT / "build" / "position_eval")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "position_corner_noise.csv")
    args = parser.parse_args()

    cache_file = args.cache / f"{args.model}_test.json"
    if not cache_file.is_file():
        sys.exit(f"{cache_file} not found. Run scripts/eval_position.py first.")
    detections = {int(k): v for k, v in json.loads(cache_file.read_text()).items()}
    items = list(load_ground_truth(args.data, "test"))

    rows = []
    for percent in NOISE_PERCENT:
        runs = []
        for seed in SEEDS if percent else SEEDS[:1]:
            rng = np.random.default_rng(seed)
            noisy_items = [
                (image_id, path, noisy(corners, percent, rng), truth)
                for image_id, path, corners, truth in items
            ]
            runs.append(score(noisy_items, detections, args.anchor, args.conf))

        def mean(key: str, runs=runs) -> float:
            return round(statistics.mean(r[key] for r in runs), 4)

        row = {
            "model": args.model,
            "noise_percent_of_board": percent,
            "seeds": len(runs),
            "piece_f1": mean("piece_f1"),
            "exact_position": mean("exact_position"),
            "within_one_error": mean("within_one_error"),
            "median_errors": mean("median_errors"),
        }
        rows.append(row)
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
