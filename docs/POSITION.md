# From photo to position: first evaluation

Everything here comes from `reports/position_eval.csv`, `reports/corner_eval.csv` and
`reports/pipeline_eval.csv`, written by `scripts/eval_position.py`, `scripts/eval_corners.py`
and `scripts/eval_pipeline.py`. The method is in `src/position.py`.

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
the corners are exact. Corners from user taps or a corner detector are worse. Taps on
a phone gave a position that was often off by a row, column or diagonal when tried by hand
(not measured). The corner detector is measured in "Finding the corners automatically" below.

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

## Sensitivity to imprecise corners

`scripts/eval_corner_noise.py` moves each of the four annotated corners by random
noise and re-scores the test split with the same detections and the settings
chosen above (not re-tuned). Noise is a standard deviation as a percentage of the
board's mean side length, applied in x and y to each corner, averaged over 5 seeds
(`reports/position_corner_noise.csv`). In these photographs the median board side
is 1,415 pixels, so 1% is about 14 pixels and one square is about 177 pixels.

| Corner noise (% of board side) | 0 | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| Piece F1 | 0.9984 | 0.9984 | 0.9983 | 0.9833 | 0.9216 | 0.7321 |
| Exact position | 96.73% | 96.73% | 96.54% | 85.16% | 59.54% | 19.21% |
| At most one wrong square | 99.35% | 99.35% | 99.15% | 89.87% | 65.75% | 22.74% |

The score holds up to about 1% of the board side and then falls quickly. By 3% only
about 6 photographs in 10 come out exactly right.

What this means for a tap-to-mark app, as reasoning and not a measurement: on a phone
the board is shown perhaps 300 to 400 pixels wide, so 1% is only 3 to 4 screen pixels,
smaller than a fingertip's usual error. Plain taps are therefore likely to land in the
2 to 3% range or worse. A zoomed corner picker, draggable handles with the 8x8 grid
drawn over the photo so the user can check the fit, or automatic corner detection is
needed. Real tap errors have not been measured, and noise here is random while human
errors may be biased in one direction.

## Finding the corners automatically

Marking four corners by hand is too imprecise (see the noise table), so a second model
finds them. It runs in three steps, and the first step is the only one that needs training.

1. **Corner detector.** A YOLOv8n-pose model with four keypoints, trained on the 1,442
   training photographs from ChessReD2K that have annotated corners, with the same
   train, validation and test split as the piece detector. The keypoints are stored in
   screen order (top-left, top-right, bottom-right, bottom-left), because an empty board
   does not show which side White is on.
2. **Refinement.** Move the four corners so each detected piece's base sits nearer the
   middle of its square (`refine_corners`). A pull towards the starting corners
   (`prior_weight`) stops a few misread pieces from dragging the board away. This only
   helps when the corners start within about half a square of the truth, since a grid
   shifted by a whole square fits the pieces equally well.
   **Several starts** (`locate_board_multistart`): because that basin is narrow, the whole
   procedure is run from 11 starting points (the detected corners, eight copies shifted by 4% of
   the board side in x and/or y, and two rescalings by 0.94 and 1.06), and the result where the
   pieces sit closest to the middle of their squares (`alignment_error`, no labels used) is kept.
   If the correct answer were used to pick among the 11, the position would be right in 83.03%
   of the validation photos and 92.81% of the test photos, so the label-free choice (80.00%,
   91.18%) recovers most of that ceiling. It adds about 60 ms per photo (median of 21 test
   photos, measured on this laptop).
3. **Naming a1** (`orient_corners`). The checkerboard pattern separates quarter turns,
   because a1 is a dark square (light minus dark brightness was positive at the annotated
   corners in all 160 photographs sampled from val and test). The pieces then separate the
   two remaining half-turn options: more white pieces on ranks 1 to 4 and black pieces on
   ranks 5 to 8 wins. Refinement comes first because the checkerboard test only works
   when the corners are within about a third of a square.

Corner error is measured as a percentage of the board's side, the same unit as the noise
table. On the 306 test photographs:

| Corner model | Mean corner error | Median of the worst corner | All corners within 3% | All corners within 6% |
|---|---|---|---|---|
| Trained with mosaic augmentation | 19.77% | 15.75% | 0.0% | 0.65% |
| Trained without mosaic augmentation | 4.22% | 5.00% | 10.46% | 64.38% |

Turning mosaic augmentation off was the only change between these two runs (both single
runs, 40 epochs requested, the first stopped early at epoch 20). The likely reason is that
mosaic shrinks and crops the board and can push corners out of the image, but that was not
tested separately. The second model is used below.

Whole-position accuracy with detected corners, same piece settings as before (anchor 0.9,
conf 0.5). The refinement strength was chosen on val (candidates 1, 3, 10, 30 and 100; 3
won) and the test split was scored with that setting:

| Corners | Exact position, val | Exact position, test |
|---|---|---|
| Annotated (ceiling) | 94.24% | 96.73% |
| Detected, a1 named from the pieces only | 50.30% | 67.65% |
| Detected, a1 named from pieces and checkerboard | 56.36% | 70.59% |
| Detected, refined, then named from pieces and checkerboard | 72.42% | 86.60% |
| Same, counting a photo as right if flipping the board half a turn fixes it | 81.82% | 87.25% |
| Detected, best of 11 nearby starts, then refined and named (used in the service) | 80.00% | 91.18% |

How to read this:

- **The multi-start row was developed after looking at both splits.** The start offsets (4%,
  0.94 and 1.06) and the selection score were tried while the validation and test results were
  both on screen, and I did not keep a split untouched for it. The score has no free
  weight to tune (I also tried adding a legality penalty and kept the simpler version), but the
  91.18% is the least clean number in this document. The 80.00% on validation is the better
  guide, and a fresh set is the only real check.

- **The gap to the ceiling is real.** With detected corners the test result is 86.60%
  against 96.73% with annotated ones, and the median worst corner after refinement is
  still 3.11% of the board side on test, with no photo having all four corners within 1%.
- **Validation is worse than test (80.00% against 91.18%), and the gap is mostly which games
  they hold.** Each split is only three games, and photos of one game are near duplicates, so
  this is a comparison of six games, not of 636 photos (`scripts/diagnose_pipeline.py`,
  `reports/pipeline_diagnostics.csv`; "one start" is the refine-first method before the
  multi-start step below, "11 starts" is the method in the service):

  | Game | Camera | Split | Annotated corners | Corners found, one start | Corners found, 11 starts |
  |---|---|---|---|---|---|
  | 6 | Huawei P40 pro | val | 92.50% | 74.17% | 81.67% |
  | 47 | iPhone 12 | val | 94.50% | 69.72% | 77.98% |
  | 58 | Galaxy S8 | val | 96.04% | 73.27% | 80.20% |
  | 0 | Huawei P40 pro | test | 94.17% | 74.76% | 84.47% |
  | 33 | iPhone 12 | test | 98.06% | 91.26% | 94.17% |
  | 76 | Galaxy S8 | test | 98.00% | 94.00% | 95.00% |

  Games 33 and 76 are easier than the other four. The camera does not explain it (each phone
  has one easy and one hard game among these six); what makes a game hard is not known.
- **What goes wrong.** Of the photos that come out wrong with 11 starts, 36.4% on validation
  (66 photos) and 44.4% on test (27 photos) still have a corner 6% or more of the board side
  from the board. On validation 40.9% have the corners on the board but a1 named wrongly, nearly
  all half turns (32 photos, 9.70% of all validation photos; 5 photos, 1.63%, on test), and on
  test 18.5% do. The rest were placed and named correctly and still wrong. 71% (val) and 63%
  (test) of the wrong photos come out right with the annotated corners. With one start, misplaced
  corners were 52% of the validation failures and 73% of the test failures, so the extra starts
  mainly removed those. An earlier version of this page said most validation failures were
  half-turn mistakes. That came from a check that was never saved, and the saved numbers do not
  support it.
- **Which way round the board sits is well covered.** a1 is in each of the four screen
  corners in about a quarter of the photos on both splits, so the orientation step is not
  being flattered by photos that all have White at the bottom.
- **The test split is not a clean one-shot for this stage.** The checkerboard step and
  the refine-first order were added after the pieces-only results on both splits had been
  seen, and after a failure analysis on validation. The refinement strength itself was
  chosen on validation only. The test split was also scored for the larger and two-stage corner
  models, so it has been looked at several times.
- **Single runs.** One training run of the corner model per setting, with MPS training
  that is not bit-reproducible.
- **Same three-game limit as before.** The corner model learned from ChessReD's boards and
  cameras. Photos of other boards and phones have not been tested, so how well it works on
  a new phone is unknown.

### Two attempts to find the corners more precisely that did not help

Both were scored the same way as the nano model (tuned on validation, then scored on test).

| Corner finder | Mean corner error, test (% of board side) | Whole position exact, val | Whole position exact, test |
|---|---|---|---|
| Nano pose model (used) | 4.22 | 72.42% | 86.60% |
| Small pose model (yolov8s-pose) | 5.26 | 73.33% | 85.62% |
| Nano model, then a second nano model on a crop around the board | 5.22 | 70.91% | 85.95% |

The second stage cuts a square crop around the first stage's corners from the full-size photo,
resizes it to 640 pixels and finds the corners again; it was trained on crops placed from the
annotated corners with random shifts and rescaling, three per training photo. It made the raw
corner error slightly worse, and the whole-position rates are within a point or two of the
nano model in either direction on 330 and 306 photos, which is not a difference these sets can
resolve. I did not find out why. Untested guesses: the annotated corners are themselves
imprecise at this level, or the crop model needs more varied jitter than I gave it.
`scripts/prepare_crop_data.py` and `--stage2` in `eval_pipeline.py` reproduce it. The service
uses the single nano model.

### In the service

`models/corners.pt` (not committed) switches this on. With it, `POST /position` without a
`corners` field runs the piece detector, the corner detector, `locate_board_multistart` (refine
from several starts, then name a1) and returns the corners it used next to the squares, and
`GET /health` reports `auto_corners: true`. `web/position.html` calls it as soon as a photo is
chosen, places the four handles from the answer for you to check, and has a "Flip board" button for
the half-turn mistakes described above. Without the weights, or if no board is found, the server
says so and the page falls back to hand-placed corners. As a check on the wiring, 51 test
photographs (every sixth) sent through the real API with no corners gave 47 exact positions
(92.2%), in line with the table (a small sample; the median request took 0.5 s on this laptop,
with the server already running and both models loaded, so it is not a cold-start figure).

## How far to trust this

- **Three games.** The 306 test photographs come from three games, with many
  near-identical frames of each position. The 96.73% exact rate is across
  photographs and likely overstates how it would do on new games, boards and
  cameras. This is the main caveat.
- **Perfect corners in the main table.** See Method, the sensitivity table, and the
  automatic-corner results above.
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

Detections are cached under `build/position_eval/` so reruns are fast. Then
`python scripts/eval_corner_noise.py --data /path/to/chessred` runs the corner sensitivity test.

The corner detector:

```bash
python scripts/prepare_corner_data.py --data /path/to/chessred --out build/corners
yolo pose train data=$PWD/build/corners/corners.yaml model=yolov8n-pose.pt imgsz=640 \
    epochs=40 patience=40 mosaic=0 batch=16 device=mps seed=42 \
    project=$PWD/build/runs name=corners-n-nomosaic
python scripts/eval_corners.py --weights build/runs/corners-n-nomosaic/weights/best.pt
python scripts/eval_pipeline.py --data /path/to/chessred \
    --weights build/runs/corners-n-nomosaic/weights/best.pt
```

Pass an absolute `project=` path: a relative one is placed under Ultralytics' own runs
folder, which can be outside this repository. The weights are not committed.
