"""Build the second-stage corner dataset: crops around the board, cut from the full-size photos.

Each training crop is placed from the annotated corners and then shifted and rescaled at
random, to imitate the first stage's errors (about 4% of the board side on average, 10% for
the worst tenth). Validation crops get one fixed random draw. Test photos are not cut here:
eval_pipeline.py crops them from the first stage's own output, which is what the service does.

    python scripts/prepare_crop_data.py --data ../data/raw/chessred --out build/crops
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_corner_data import label_line, screen_order  # noqa: E402

from crops import crop_square, to_crop  # noqa: E402

SIZE = 640
COPIES = {"train": 3, "val": 1}  # jittered crops per photo
SHIFT = 0.06  # centre shift, as a fraction of the crop side
SCALE = (0.9, 1.15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("build/crops"))
    args = ap.parse_args()

    ann = json.loads((args.data / "annotations.json").read_text())
    images = {i["id"]: i for i in ann["images"]}
    corners = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
    rng = random.Random(42)

    for split, copies in COPIES.items():
        img_dir = args.out / "images" / split
        lbl_dir = args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        ids = ann["splits"]["chessred2k"][split]["image_ids"]
        for n, image_id in enumerate(ids, 1):
            info = images[image_id]
            pts = screen_order(corners[image_id])
            with Image.open(args.data / info["path"]) as img:
                img = img.convert("RGB")
                for k in range(copies):
                    shift = (rng.uniform(-SHIFT, SHIFT), rng.uniform(-SHIFT, SHIFT))
                    box = crop_square(pts, shift=shift, scale=rng.uniform(*SCALE))
                    left, top, side = box
                    crop = img.crop(
                        (round(left), round(top), round(left + side), round(top + side))
                    )
                    # Round the box to whole pixels so the labels match the cut exactly.
                    box = (round(left), round(top), crop.width)
                    crop = crop.resize((SIZE, SIZE), Image.LANCZOS)
                    stem = f"{Path(info['file_name']).stem}_{k}"
                    crop.save(img_dir / f"{stem}.jpg", quality=92)
                    local = to_crop(pts, box, SIZE)
                    (lbl_dir / f"{stem}.txt").write_text(label_line(local, SIZE, SIZE) + "\n")
            if n % 200 == 0:
                print(f"{split}: {n}/{len(ids)}", flush=True)

    yaml = (
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\n"
        "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: board\n"
    )
    (args.out / "crops.yaml").write_text(yaml)
    print("wrote", args.out / "crops.yaml")


if __name__ == "__main__":
    main()
