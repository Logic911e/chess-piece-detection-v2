"""Does the alignment error of the located board tell right positions from wrong ones?

alignment_error is the mean distance, in squares, from each piece's base to the centre of its
square. It needs no labels, so the service can report it. This script scores it as a detector of
wrong positions on the validation and test splits (AUROC), and shows what a threshold chosen on
validation does on both. Run eval_pipeline.py first so the caches exist.

    python scripts/eval_confidence.py --data ../data/raw/chessred \\
        --weights build/runs/corners-n-nomosaic/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pipeline import ANCHOR_Y, CONF  # noqa: E402
from eval_position import load_ground_truth  # noqa: E402

from position import (  # noqa: E402
    alignment_error,
    detections_to_position,
    locate_board_multistart,
)

PRIOR = 3.0


def auroc(scores_wrong: list[float], scores_right: list[float]) -> float:
    """Chance that a wrong position has a higher score than a right one (ties count half)."""
    wins = sum((w > r) + 0.5 * (w == r) for w in scores_wrong for r in scores_right)
    return wins / (len(scores_wrong) * len(scores_right))


def share_right(results: list[tuple[float, bool]]) -> float:
    return sum(ok for _, ok in results) / max(1, len(results))


def collect(data: Path, tag: str, split: str) -> list[tuple[float, bool]]:
    items = list(load_ground_truth(data, split))
    detections = {
        int(k): v
        for k, v in json.loads(
            (ROOT / f"build/position_eval/finetuned_{split}.json").read_text()
        ).items()
    }
    screen = {
        int(k): v
        for k, v in json.loads((ROOT / f"build/corner_eval/{tag}_{split}.json").read_text()).items()
    }
    out = []
    for image_id, path, _, truth in items:
        pts = screen.get(image_id)
        if not pts:
            out.append((1.0, False))
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        try:
            corners = locate_board_multistart(
                pts, detections[image_id], gray, ANCHOR_Y, CONF, PRIOR
            )
        except Exception:
            out.append((1.0, False))
            continue
        score = alignment_error(detections[image_id], corners, ANCHOR_Y, CONF)
        right = detections_to_position(detections[image_id], corners, ANCHOR_Y, CONF) == truth
        out.append((score, right))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/confidence_eval.csv")
    args = ap.parse_args()
    tag = args.weights.parent.parent.name

    data = {s: collect(args.data, tag, s) for s in ("val", "test")}
    rows = []
    for split, results in data.items():
        wrong = [s for s, ok in results if not ok]
        right = [s for s, ok in results if ok]
        a = auroc(wrong, right)
        rows.append(
            {
                "weights": tag,
                "split": split,
                "measure": "AUROC (wrong vs right)",
                "value": round(a, 4),
            }
        )
        print(split, "AUROC", round(a, 4), f"({len(wrong)} wrong, {len(right)} right)")

    # Thresholds from the validation scores: the value that flags this share of val photos.
    val_scores = sorted(s for s, _ in data["val"])
    for share in (0.1, 0.2, 0.3, 0.4):
        threshold = val_scores[int((1 - share) * (len(val_scores) - 1))]
        for split, results in data.items():
            flagged = [(s, ok) for s, ok in results if s > threshold]
            kept = [(s, ok) for s, ok in results if s <= threshold]
            row = {
                "weights": tag,
                "split": split,
                "measure": f"threshold {threshold:.4f} (flags top {share:.0%} on val)",
                "value": (
                    f"flagged {len(flagged) / len(results):.3f}; "
                    f"right among flagged {share_right(flagged):.3f}; "
                    f"right among kept {share_right(kept):.3f}"
                ),
            }
            rows.append(row)
            print(split, row["measure"], row["value"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
