"""Measure the whole photo-to-position pipeline with no hand-placed corners.

Per image: the corner detector finds four corners (screen order), the pieces decide which
is a1 (orient_corners), and optionally the corners are nudged towards the piece grid
(refine_corners). The position is then scored against the annotation, using the cached
piece detections from eval_position.py.

The refinement strength is tuned on the validation split. Every variant is also scored on the
test split and written to the CSV, so the test split is seen for all of them. Piece settings
stay at the values tuned there earlier (anchor 0.9, conf 0.5).

    python scripts/eval_pipeline.py --data ../data/raw/chessred \\
        --weights build/runs/corners-n-nomosaic/weights/best.pt

Add --stage2 with the weights of the crop model to run the second stage on the first stage's
corners before everything else.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_position import load_ground_truth, score  # noqa: E402
from prepare_corner_data import screen_order  # noqa: E402

from crops import crop_square, from_crop  # noqa: E402
from position import (  # noqa: E402
    CORNER_ORDER,
    detections_to_position,
    locate_board,
    locate_board_multistart,
    orient_corners,
    refine_corners,
)

ANCHOR_Y = 0.9
CONF = 0.5
PRIOR_WEIGHTS = [1.0, 3.0, 10.0, 30.0, 100.0]
CROP_SIZE = 640
CORNER_IMAGES = ROOT / "build/corners/images"
# A square far outside any photo: every piece lands off the board, so the image scores as a miss.
FAR_AWAY = [[-2000.0, -2000.0], [-1000.0, -2000.0], [-1000.0, -1000.0], [-2000.0, -1000.0]]


def detect_corners(model: YOLO, split: str, items: list, cache_file: Path) -> dict[int, list]:
    """Screen-order corners (original-image pixels) per image id, cached on disk."""
    cache: dict[str, list] = {}
    if cache_file.is_file():
        cache = json.loads(cache_file.read_text())
    for image_id, path, _, _ in items:
        if str(image_id) in cache:
            continue
        small = CORNER_IMAGES / split / f"{path.stem}.jpg"
        result = model.predict(str(small), imgsz=640, verbose=False, device="cpu")[0]
        if result.keypoints is None or len(result.boxes) == 0:
            cache[str(image_id)] = []
            continue
        best = int(result.boxes.conf.argmax())
        factor = _original_side(path) / result.orig_shape[0]
        cache[str(image_id)] = [
            [x * factor, y * factor] for x, y in result.keypoints.xy[best].tolist()
        ]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return {int(k): v for k, v in cache.items()}


def second_stage(model: YOLO, items: list, screen: dict[int, list], cache_file: Path) -> dict:
    """Move each first-stage corner set using a crop of the full-size photo around it."""
    cache: dict[str, list] = {}
    if cache_file.is_file():
        cache = json.loads(cache_file.read_text())
    for image_id, path, _, _ in items:
        if str(image_id) in cache:
            continue
        pts = screen.get(image_id)
        if not pts:
            cache[str(image_id)] = []
            continue
        left, top, side = crop_square(pts)
        with Image.open(path) as img:
            crop = img.convert("RGB").crop(
                (round(left), round(top), round(left) + round(side), round(top) + round(side))
            )
        box = (round(left), round(top), crop.width)
        crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
        result = model.predict(crop, imgsz=CROP_SIZE, verbose=False, device="cpu")[0]
        if result.keypoints is None or len(result.boxes) == 0:
            cache[str(image_id)] = pts  # keep the first stage's answer
            continue
        best = int(result.boxes.conf.argmax())
        cache[str(image_id)] = from_crop(result.keypoints.xy[best].tolist(), box, CROP_SIZE)
    cache_file.write_text(json.dumps(cache))
    return {int(k): v for k, v in cache.items()}


def raw_error_pct(items: list, screen: dict[int, list]) -> float:
    """Mean corner error of the detector's own output, % of board side, missed boards excluded."""
    errs = []
    for image_id, _, truth, _ in items:
        pts = screen.get(image_id)
        if not pts:
            continue
        gt = screen_order(truth)
        side = sum(math.dist(gt[i], gt[(i + 1) % 4]) for i in range(4)) / 4
        errs += [100 * math.dist(p, t) / side for p, t in zip(pts, gt, strict=True)]
    return round(sum(errs) / len(errs), 3)


def _original_side(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as img:
        return img.width  # the photos are square


def with_corners(items: list, corners_by_id: dict[int, dict | None]) -> list:
    """Items with the corners replaced. A photo with no detected board gets FAR_AWAY corners,
    so it is scored as a failure."""
    out = []
    for image_id, path, _truth_corners, truth in items:
        corners = corners_by_id.get(image_id)
        if corners is None:
            corners = dict(zip(CORNER_ORDER, FAR_AWAY, strict=True))
        out.append((image_id, path, corners, truth))
    return out


def corner_error_pct(corners: dict, truth: dict) -> float:
    pts = [truth[n] for n in CORNER_ORDER]
    side = sum(math.dist(pts[i], pts[(i + 1) % 4]) for i in range(4)) / 4
    return max(100 * math.dist(corners[n], truth[n]) / side for n in CORNER_ORDER)


def build(
    items, detections, screen, refine_weight: float | None, checkerboard: bool, multistart=False
) -> dict[int, dict | None]:
    result: dict[int, dict | None] = {}
    for image_id, path, _, _ in items:
        pts = screen.get(image_id)
        if not pts:
            result[image_id] = None
            continue
        dets = detections[image_id]
        try:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if checkerboard else None
            if multistart:
                corners = locate_board_multistart(pts, dets, gray, ANCHOR_Y, CONF, refine_weight)
            elif refine_weight is not None and checkerboard:
                corners = locate_board(pts, dets, gray, ANCHOR_Y, CONF, refine_weight)
            else:
                corners = orient_corners(pts, dets, ANCHOR_Y, CONF, gray)
                if refine_weight is not None:
                    corners = refine_corners(
                        corners, dets, ANCHOR_Y, CONF, prior_weight=refine_weight
                    )
        except Exception:  # a twisted quadrilateral: count it as not found
            corners = None
        result[image_id] = corners
    return result


def half_turn(corners: dict) -> dict:
    """The same corners with a1 and h8 swapped, and h1 and a8 swapped."""
    return {
        "bottom_left": corners["top_right"],
        "top_right": corners["bottom_left"],
        "bottom_right": corners["top_left"],
        "top_left": corners["bottom_right"],
    }


def exact_with_flip(items, detections, corners_by_id) -> float:
    """Share of photos whose position is right as found, or right after the user flips the
    board half a turn (the page offers that button)."""
    right = 0
    for image_id, _, _, truth in items:
        corners = corners_by_id.get(image_id)
        if corners is None:
            continue
        for candidate in (corners, half_turn(corners)):
            if detections_to_position(detections[image_id], candidate, ANCHOR_Y, CONF) == truth:
                right += 1
                break
    return right / len(items)


def evaluate(items, detections, corners_by_id) -> dict:
    metrics = score(with_corners(items, corners_by_id), detections, ANCHOR_Y, CONF)
    errors = [
        corner_error_pct(corners_by_id[i], c)
        for i, _, c, _ in items
        if corners_by_id.get(i) is not None
    ]
    metrics["worst_corner_median_pct"] = round(sorted(errors)[len(errors) // 2], 3)
    metrics["exact_with_flip_option"] = round(exact_with_flip(items, detections, corners_by_id), 4)
    metrics["all_corners_within_1pct"] = round(sum(e <= 1 for e in errors) / len(items), 4)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--stage2", type=Path, help="weights of the crop model (second stage)")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/pipeline_eval.csv")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    tag = tag0 = args.weights.parent.parent.name
    model2 = YOLO(str(args.stage2)) if args.stage2 else None
    if model2:
        tag += "+" + args.stage2.parent.parent.name
    split_data = {}
    for split in ("val", "test"):
        items = list(load_ground_truth(args.data, split))
        detections = {
            int(k): v
            for k, v in json.loads(
                (ROOT / f"build/position_eval/finetuned_{split}.json").read_text()
            ).items()
        }
        cache = ROOT / f"build/corner_eval/{tag}_{split}.json"

        screen = detect_corners(
            model, split, items, ROOT / f"build/corner_eval/{tag0}_{split}.json"
        )
        if model2:
            screen = second_stage(model2, items, screen, cache)
        split_data[split] = (items, detections, screen)

    def variants(split: str) -> dict[str, dict]:
        items, detections, screen = split_data[split]
        gt = {i: c for i, _, c, _ in items}
        out = {"annotated corners": evaluate(items, detections, gt)}
        for label, checkerboard in (("pieces", False), ("pieces + checkerboard", True)):
            out[f"detected, orientation from {label}"] = evaluate(
                items, detections, build(items, detections, screen, None, checkerboard)
            )
        out["detected, orientation from pieces + refine (prior 10)"] = evaluate(
            items, detections, build(items, detections, screen, 10.0, False)
        )
        for w in PRIOR_WEIGHTS:
            name = f"detected, refine, then orientation from pieces + checkerboard (prior {w:g})"
            out[name] = evaluate(items, detections, build(items, detections, screen, w, True))
        name = "detected, best of 11 starts by piece alignment, same method (prior 3)"
        out[name] = evaluate(items, detections, build(items, detections, screen, 3.0, True, True))
        for m in out.values():
            m["detector_mean_error_pct"] = raw_error_pct(items, screen)
        return out

    val = variants("val")
    for name, m in val.items():
        print(
            f"val  {name:55s} exact {m['exact_position']:.4f}  within1 {m['within_one_error']:.4f}"
        )
    best = max(
        (n for n in val if "refine, then orientation" in n), key=lambda n: val[n]["exact_position"]
    )
    print("best refine setting on val:", best)

    test = variants("test")
    rows = []
    for split, table in (("val", val), ("test", test)):
        for name, m in table.items():
            rows.append({"weights": tag, "split": split, "variant": name, **m})
    for name in test:
        m = test[name]
        print(
            f"test {name:55s} exact {m['exact_position']:.4f}  within1 {m['within_one_error']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():  # keep the rows of other weights, replace this run's own
        with args.out.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["weights"] != tag] + rows
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
