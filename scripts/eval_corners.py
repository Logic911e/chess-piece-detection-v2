"""Measure how close the corner detector gets to the annotated board corners.

Error is the distance from each predicted corner to the annotated one, as a percentage
of the board's side (the mean edge length of the annotated quadrilateral). That is the
unit used in docs/POSITION.md, so the numbers compare with the corner-noise table.
An image with no detection counts as a failure.

    python scripts/eval_corners.py --weights build/runs/corners-n/weights/best.pt --split test
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 6.0]


def read_label(path: Path, size: int) -> list[tuple[float, float]]:
    v = [float(x) for x in path.read_text().split()[5:]]
    return [(v[i] * size, v[i + 1] * size) for i in range(0, 12, 3)]


def board_side(points: list[tuple[float, float]]) -> float:
    return statistics.mean(math.dist(points[i], points[(i + 1) % 4]) for i in range(4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "build/corners")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/corner_eval.csv")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    worst_per_image: list[float] = []  # largest corner error in the image, % of board side
    mean_per_image: list[float] = []
    missed = 0
    for n, img in enumerate(images, 1):
        result = model.predict(str(img), imgsz=args.imgsz, verbose=False, device="cpu")[0]
        if result.keypoints is None or len(result.boxes) == 0:
            missed += 1
            continue
        best = int(result.boxes.conf.argmax())
        pred = result.keypoints.xy[best].tolist()
        size = result.orig_shape[0]
        truth = read_label(args.data / "labels" / args.split / f"{img.stem}.txt", size)
        side = board_side(truth)
        errs = [100 * math.dist(p, t) / side for p, t in zip(pred, truth, strict=True)]
        worst_per_image.append(max(errs))
        mean_per_image.append(statistics.mean(errs))
        if n % 100 == 0:
            print(f"{n}/{len(images)}", flush=True)

    total = len(images)
    found = len(worst_per_image)
    ordered = sorted(worst_per_image)
    row = {
        "weights": args.weights.parent.parent.name,
        "split": args.split,
        "images": total,
        "detected": found,
        "mean_error_pct": round(statistics.mean(mean_per_image), 3),
        "median_worst_corner_pct": round(statistics.median(ordered), 3),
        "p90_worst_corner_pct": round(ordered[int(0.9 * (found - 1))], 3),
    }
    for t in THRESHOLDS:
        # Share of ALL images whose worst corner is within t% of the board side.
        row[f"all_corners_within_{t}pct"] = round(sum(e <= t for e in ordered) / total, 4)
    print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
