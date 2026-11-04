"""Measure how often a photo becomes the right chess position.

For each ChessReD2K image, run a detector, map the detections to squares with
the annotated board corners (src/position.py), and compare with the annotated
position. Two settings of detections_to_position are tuned on the validation
split (anchor_y and the confidence cut-off), then the best pair is scored once
on the test split.

Uses the ground-truth corners, so this measures the square assignment given
perfect corners. Corners from a user's taps or a corner detector will be worse.

    python scripts/eval_position.py --data ../data/raw/chessred
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from position import detections_to_position  # noqa: E402
from predict import load_models, predict  # noqa: E402

ANCHORS = [0.6, 0.7, 0.8, 0.9, 1.0]
CONFS = [0.25, 0.4, 0.5, 0.6, 0.7]
CACHE_CONF = min(CONFS)


def load_ground_truth(data_dir: Path, split: str):
    """Yield (image_id, image_path, corners, true_position) for a split."""
    ann = json.loads((data_dir / "annotations.json").read_text())
    names = {c["id"]: c["name"] for c in ann["categories"]}
    images = {i["id"]: i for i in ann["images"]}
    corners = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
    pieces: dict[int, dict[str, str]] = {}
    for p in ann["annotations"]["pieces"]:
        pieces.setdefault(p["image_id"], {})[p["chessboard_position"]] = names[p["category_id"]]
    for image_id in ann["splits"]["chessred2k"][split]["image_ids"]:
        path = data_dir / images[image_id]["path"]
        yield image_id, path, corners[image_id], pieces.get(image_id, {})


def detect_all(models, model_name: str, items: list, cache_file: Path) -> dict[int, list[dict]]:
    """Detections per image at the lowest confidence in the grid, cached on disk."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict]] = {}
    if cache_file.is_file():
        cache = json.loads(cache_file.read_text())
    for n, (image_id, path, _, _) in enumerate(items, 1):
        if str(image_id) in cache:
            continue
        with Image.open(path) as img:
            result = predict(models, model_name, img.convert("RGB"), conf=CACHE_CONF)
        cache[str(image_id)] = result["detections"]
        if n % 25 == 0:
            print(f"  {model_name}: {n}/{len(items)} images", flush=True)
            cache_file.write_text(json.dumps(cache))
    cache_file.write_text(json.dumps(cache))
    return {int(k): v for k, v in cache.items()}


def score(items: list, detections: dict[int, list[dict]], anchor_y: float, conf: float) -> dict:
    """Piece-level and whole-position metrics over a list of images."""
    tp = n_pred = n_true = 0
    exact = within_one = 0
    errors = []
    for image_id, _, corners, truth in items:
        pred = detections_to_position(detections[image_id], corners, anchor_y, conf)
        right = sum(1 for sq, name in pred.items() if truth.get(sq) == name)
        tp += right
        n_pred += len(pred)
        n_true += len(truth)
        wrong = len({*pred, *truth} - {sq for sq in pred if truth.get(sq) == pred[sq]})
        errors.append(wrong)
        exact += wrong == 0
        within_one += wrong <= 1
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(items)
    return {
        "anchor_y": anchor_y,
        "conf": conf,
        "images": n,
        "piece_precision": round(precision, 4),
        "piece_recall": round(recall, 4),
        "piece_f1": round(f1, 4),
        "exact_position": round(exact / n, 4),
        "within_one_error": round(within_one / n, 4),
        "median_errors": statistics.median(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="folder with annotations.json")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--cache", type=Path, default=ROOT / "build" / "position_eval")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "position_eval.csv")
    parser.add_argument("--limit", type=int, help="only the first N images per split (smoke test)")
    args = parser.parse_args()

    models = load_models(args.models)
    splits = {s: list(load_ground_truth(args.data, s)) for s in ("val", "test")}
    if args.limit:
        splits = {s: items[: args.limit] for s, items in splits.items()}

    rows = []
    for model_name in models:
        dets = {
            s: detect_all(models, model_name, items, args.cache / f"{model_name}_{s}.json")
            for s, items in splits.items()
        }
        grid = [
            {"model": model_name, "split": "val", **score(splits["val"], dets["val"], a, c)}
            for a, c in itertools.product(ANCHORS, CONFS)
        ]
        best = max(grid, key=lambda r: r["piece_f1"])
        test = {
            "model": model_name,
            "split": "test",
            **score(splits["test"], dets["test"], best["anchor_y"], best["conf"]),
        }
        rows += grid + [test]
        print(
            f"{model_name}: chosen on val anchor_y={best['anchor_y']} conf={best['conf']} "
            f"(val F1 {best['piece_f1']}) -> test F1 {test['piece_f1']}, "
            f"exact {test['exact_position']}, within one error {test['within_one_error']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
