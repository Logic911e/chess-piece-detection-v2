# From photo to position: first evaluation

Everything here comes from `reports/position_eval.csv`, written by
`scripts/eval_position.py`. The method is in `src/position.py`.

## Method

1. Run a detector on the photo.
2. Map image pixels to board coordinates with a homography built from the four
   board corners. The corners are named by the square they sit on (a1 is
   `bottom_left`, a8 `top_left`, h1 `bottom_right`, h8 `top_right`), which is how
   ChessReD annotates them, so the board comes out the right way round.
3. Take the point on each box where the piece stands (centre of the box, a set
   fraction `anchor_y` of the way down) and find its square.
4. If two detections land on one square, keep the more confident one.

Two settings were tuned on the validation split: `anchor_y` (0.6 to 1.0) and the
confidence cut-off (0.25 to 0.7). The best pair by F1 on validation was then
scored once on the test split.

**This uses the annotated corners**, so it measures the square assignment when
the corners are exact. Corners from user taps or a corner detector will be worse,
and that has not been measured.

## Results on the 306 test photographs

| Model | Settings chosen on val | Piece precision | Piece recall | Piece F1 | Exact position | At most one wrong square |
|---|---|---|---|---|---|---|
| synthetic | anchor 0.9, conf 0.4 | 0.2756 | 0.2526 | 0.2636 | 0.0% | 0.0% |
| finetuned | anchor 0.9, conf 0.5 | 0.9994 | 0.9975 | 0.9984 | 96.73% | 99.35% |

A piece counts as right when the square and the class both match the annotation.
"Exact position" means every square in the image is right. The synthetic model's
median image has 16 wrong squares.

Sensitivity on the validation split (fine-tuned model): F1 is 0.8725 at
`anchor_y` 0.6, 0.9704 at 0.7, 0.9971 at 0.8, 0.9973 at 0.9 and 0.9954 at 1.0.
The confidence cut-off barely matters between 0.25 and 0.6 (F1 0.9970 to 0.9973).
The anchor matters because a tall piece's box covers the squares behind it, so
a point too high on the box lands on the wrong square.

## How far to trust this

- **Three games.** The 306 test photographs come from three games, with many
  near-identical frames of each position. The 96.73% exact rate is across
  photographs and likely overstates how it would do on new games, boards and
  cameras. This is the main caveat.
- **Perfect corners.** See Method.
- **Piece placement only.** A photo cannot give whose turn it is, castling rights
  or move counters, so only the piece-placement part of a FEN is produced.
- **Tuning and test share a pipeline but not images.** The settings came from
  the validation split and the test split was scored once.
- **Not tested:** other lighting, other chess sets, hidden pieces on crowded
  boards beyond what is in these photographs.

## Reproduce

```bash
python scripts/eval_position.py --data /path/to/chessred   # folder with annotations.json
```

Detections are cached under `build/position_eval/` so reruns are fast.
