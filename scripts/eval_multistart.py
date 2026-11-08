"""How much the multi-start step can gain, and how much of it the label-free choice gets.

For each split, locates the board from each of the 11 starting points on its own (the first is
the detector's answer), then reports the exact-position rate of each start, of the choice
locate_board_multistart makes without labels, and of an oracle that picks the best start with
the correct answer. The oracle is a ceiling, not something the service can do.

    python scripts/eval_multistart.py --data ../data/raw/chessred \\
        --weights build/runs/corners-n-nomosaic/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pipeline import ANCHOR_Y, CONF, build  # noqa: E402
from eval_position import load_ground_truth  # noqa: E402

from position import detections_to_position, start_points  # noqa: E402

PRIOR = 3.0


def exact_by_image(items, detections, located) -> dict[int, bool]:
    return {
        image_id: located.get(image_id) is not None
        and detections_to_position(detections[image_id], located[image_id], ANCHOR_Y, CONF) == truth
        for image_id, _, _, truth in items
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/multistart_eval.csv")
    args = ap.parse_args()

    tag = args.weights.parent.parent.name
    rows = []
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
        n = len(items)
        per_start = []
        for k in range(len(start_points([[0, 0], [1, 0], [1, 1], [0, 1]]))):
            moved = {i: start_points(p)[k] if p else p for i, p in screen.items()}
            located = build(items, detections, moved, PRIOR, True)
            per_start.append(exact_by_image(items, detections, located))
        chosen = exact_by_image(
            items, detections, build(items, detections, screen, PRIOR, True, True)
        )
        for k, ok in enumerate(per_start):
            rows.append(
                {
                    "weights": tag,
                    "split": split,
                    "what": f"start {k} alone",
                    "exact": round(sum(ok.values()) / n, 4),
                }
            )
        oracle = sum(any(ok[i] for ok in per_start) for i in chosen) / n
        rows.append(
            {
                "weights": tag,
                "split": split,
                "what": "chosen without labels",
                "exact": round(sum(chosen.values()) / n, 4),
            }
        )
        rows.append(
            {
                "weights": tag,
                "split": split,
                "what": "oracle (best start, uses labels)",
                "exact": round(oracle, 4),
            }
        )
        for row in rows[-2:]:
            print(split, row["what"], row["exact"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
