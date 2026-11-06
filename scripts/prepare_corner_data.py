"""Convert ChessReD's board-corner annotations to a YOLO pose dataset.

One object per image (the board) with four keypoints. The keypoints are in screen
order (top-left, top-right, bottom-right, bottom-left of the quadrilateral as it
appears in the photo), not a1/h1/h8/a8. Which corner is a1 cannot be told from an
empty board and depends on where White sits, so that is decided later from the
detected pieces.

Uses the same splits as the piece detector (chessred2k), so the test photos stay
unseen by both models. Photos are shrunk to 1024 px: the originals are about 3000 px
and decoding them every epoch would dominate training time.

    python scripts/prepare_corner_data.py --data ../data/raw/chessred --out build/corners
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

SIZE = 1024
BOX_PAD = 0.02  # box margin around the corners, as a fraction of the image side


def screen_order(corners: dict[str, list[float]]) -> list[list[float]]:
    """Corners as [TL, TR, BR, BL] in image coordinates."""
    pts = list(corners.values())
    cx = sum(p[0] for p in pts) / 4
    cy = sum(p[1] for p in pts) / 4
    # Clockwise on screen (y points down) is increasing atan2 angle.
    import math

    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(4), key=lambda i: pts[i][0] + pts[i][1])
    return pts[start:] + pts[:start]


def label_line(points: list[list[float]], width: int, height: int) -> str:
    xs = [p[0] / width for p in points]
    ys = [p[1] / height for p in points]
    x0, x1 = max(0.0, min(xs) - BOX_PAD), min(1.0, max(xs) + BOX_PAD)
    y0, y1 = max(0.0, min(ys) - BOX_PAD), min(1.0, max(ys) + BOX_PAD)
    parts = [0, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    for x, y in zip(xs, ys, strict=True):
        parts += [min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), 2]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("build/corners"))
    args = ap.parse_args()

    ann = json.loads((args.data / "annotations.json").read_text())
    images = {i["id"]: i for i in ann["images"]}
    corners = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}

    for split in ("train", "val", "test"):
        ids = ann["splits"]["chessred2k"][split]["image_ids"]
        img_dir = args.out / "images" / split
        lbl_dir = args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for n, image_id in enumerate(ids, 1):
            info = images[image_id]
            stem = Path(info["file_name"]).stem
            dst = img_dir / f"{stem}.jpg"
            if not dst.exists():
                with Image.open(args.data / info["path"]) as img:
                    img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(dst, quality=92)
            pts = screen_order(corners[image_id])
            (lbl_dir / f"{stem}.txt").write_text(
                label_line(pts, info["width"], info["height"]) + "\n"
            )
            if n % 200 == 0:
                print(f"{split}: {n}/{len(ids)}", flush=True)

    yaml = (
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: board\n"
    )
    (args.out / "corners.yaml").write_text(yaml)
    print("wrote", args.out / "corners.yaml")


if __name__ == "__main__":
    main()
