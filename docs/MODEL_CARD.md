# Model card: chess piece detectors (synthetic and fine-tuned)

Two YOLOv8s object detectors that find chess pieces in a photograph and label
each one with one of 12 classes (6 piece types, 2 colours). They exist to show
the **domain gap**: a model trained only on rendered images, and the same model
after fine-tuning on real photographs.

Every number here comes from `RESULTS.md`, `reports/eval_test.csv`,
`reports/finetune_sweep.csv`, or the metadata stored inside the checkpoints.
Where a number was never recorded it says TBD.

## Models

| | `synthetic` | `finetuned` |
|---|---|---|
| Architecture | YOLOv8s | YOLOv8s |
| Trained on | 4,400 rendered images | The synthetic model, then 1,442 real photographs |
| Weights file | `models/synthetic.pt` | `models/finetuned.pt` |
| Classes | 12 (same list for both) | 12 |
| Role in the project | The baseline that shows the gap | The adapted model |

Classes, in index order: white-pawn, white-knight, white-bishop, white-rook,
white-queen, white-king, then the same six for black.

## Intended use

- A portfolio and teaching demo of deploying a computer-vision model as a
  tested, containerised service, and of showing a domain gap on real data.
- Detecting chess pieces in smartphone photographs of a board, in conditions
  similar to the ChessReD photographs (see below).

## Out of scope

- **Commercial use.** The fine-tuned weights are derived from a non-commercial
  dataset (see Licence notes).
- **Recovering the game state automatically.** The models detect and label pieces.
  They do not find the board. Given the four corners, `src/position.py` maps pieces
  to squares; see `docs/POSITION.md` for how well that works.
- **Other chess sets, boards or camera setups.** Not tested. The synthetic model
  in particular does not transfer (see Metrics).
- Any safety-relevant or high-stakes decision.

## Training data

| Source | What | Used for |
|---|---|---|
| chesscog rendered dataset (Wölflein and Arandjelović), [OSF](https://doi.org/10.17605/OSF.IO/XF3KA) | 4,888 Blender renders with exact per-piece boxes. Split 4,400 train / 146 val / 342 test | Synthetic model |
| ChessReD2K, a box-annotated subset of ChessReD (Masouris and van Gemert, VISAPP 2024), [4TU.ResearchData](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2) | 2,078 real smartphone photographs, 3 handsets. Split 1,442 train / 330 val / 306 test | Fine-tuning and all real-photo evaluation |

Notes that affect how to read the results:

- The renders' boxes are **amodal**: computed from all mesh vertices, including
  parts hidden behind other pieces. Hand-annotated datasets differ.
- In the real test split each handset photographed exactly one game, so camera
  and game are the same variable (`RESULTS.md`, F8).
- The class balance is intrinsic to chess (pawn to queen ratio 8.6 to 1) and was
  not rebalanced.

## Training setup

Read from `train_args` stored in each checkpoint. There is no `args.yaml` or
`results.csv` for these runs.

| | `synthetic` | `finetuned` |
|---|---|---|
| Starting weights | `yolov8s.pt` | The synthetic model |
| Epochs requested | 50 | 30 |
| Epochs completed | **8** (Colab ended the session; the saved checkpoint is epoch index 7) | Not recorded in the stripped checkpoint |
| Batch size | 16 | 8 |
| Image size | 640 | 640 |
| Initial learning rate | 0.01 | 0.001 |
| Early-stopping patience | 20 | 10 |
| Seed | 42 | 42 |
| Hardware | Colab T4 GPU | Apple MPS (not bit-reproducible) |
| Training images | 4,400 | 1,442 |

The 8-of-50 shortfall means the synthetic model is under-trained. `RESULTS.md`
estimates continuing to 50 epochs would add roughly 0.02 mAP50-95, so the
reported domain-gap drop may be slightly overstated.

## Metrics

All figures are mAP50-95 unless stated. Test sets: 342 rendered images, and 306
real photographs. Sources: `reports/eval_test.csv`, `reports/finetune_sweep.csv`,
`RESULTS.md` (Experiments 2 and 3).

| Model | Test set | mAP50-95 | mAP50 | Precision | Recall |
|---|---|---|---|---|---|
| synthetic | Rendered (342) | 0.9366 | 0.9950 | 0.9975 | 0.9963 |
| synthetic | Real photographs (306), zero-shot | **0.1895** | 0.2687 | 0.2441 | 0.4036 |
| finetuned (N=1,442) | Real photographs (306) | **0.9029** | 0.9949 | 0.9957 | 0.9936 |

The synthetic model loses 0.7471 mAP50-95 (79.8%) on real photographs.

Fine-tuning data efficiency (mean of 3 seeds up to N=50, single runs above):

| Real images used | 0 | 2 | 5 | 10 | 25 | 50 | 100 | 1,442 |
|---|---|---|---|---|---|---|---|---|
| mAP50-95 | 0.1895 | 0.4330 | 0.5673 | 0.6227 | 0.7405 | 0.8262 | 0.8711 | 0.9029 |

The seed-to-seed spread is large for small N (0.1085 at N=2) and small by N=50
(0.0137), so differences smaller than that spread are not meaningful.

Checkpoint metrics (`val_*` in MLflow) are measured on each run's own
validation split, which is synthetic for one model and real for the other.
**They are not comparable to each other.**

## Known failure modes

Synthetic model on real photographs (`RESULTS.md`, F5 to F7):
- **Rooks transfer, nothing else does.** Per-class mAP50-95 on real photos is
  0.602 (white rook) and 0.557 (black rook), against 0.034 for the black bishop
  and 0.045 for the white knight. Rooks average 0.579, every other class 0.112.
  The likely reason is geometric: rook silhouettes look alike across chess sets,
  while knights, bishops and queens differ between sets. This is an explanation
  from the analysis, not a controlled experiment.
- **It finds pieces but names them wrongly.** Recall (0.4036) is well above
  precision (0.2441).
- **White pieces do better than black.** The average gap grows from 0.0110 on
  renders to 0.0617 on real photos.

Both models:
- **Knights are the hardest class on renders** (0.9141 and 0.9188, the two lowest
  of 12), because a knight is the one piece that looks different as it rotates.
- **Shallower camera angles score lower on renders**: 0.9302 at 45 degrees to
  0.9425 at 60 degrees, consistent with pieces hiding behind each other.
- **Box tightness on real photos.** After fine-tuning, mAP50 matches the
  synthetic model on its home data, but mAP50-95 is still 0.034 lower. The
  remaining gap is strict box placement.

Not tested: lighting changes on real photos, other chess sets, other angles
beyond ChessReD, and the model on the author's own photographs.

## Licence notes

These need a decision before anything is made public.

- **Ultralytics YOLOv8 is AGPL-3.0.** The fine-tuned checkpoint's metadata records this. Running the
  models as a network service has source-offer obligations under the AGPL. A
  commercial Ultralytics licence is the usual alternative.
- **ChessReD is CC BY-NC-SA 4.0** (non-commercial, share-alike). The fine-tuned
  weights are derived from it, so they inherit the non-commercial restriction
  and should not be redistributed outside those terms. The project does not
  publish the weights.
- **The chesscog rendered dataset is CC BY 4.0.** This was read from the OSF
  project record (node `xf3ka`) on 2026-11-04. It allows reuse, including
  commercial use, with attribution. The chesscog code is MIT licensed. The
  synthetic model is trained only on this data, but it starts from Ultralytics'
  `yolov8s.pt`, so the AGPL point above still applies to it.
- **Whether model weights count as an adaptation of the training data is not
  settled in law**, and this card is not legal advice. The cautious reading is
  that the fine-tuned weights must follow CC BY-NC-SA terms: non-commercial,
  with attribution, and shared under the same licence if shared at all.
- **python-chess and Stockfish are GPL-3.0.** The service imports python-chess
  (GPL-3.0 or later) for legality checks and calls a separately installed Stockfish
  binary (GPL-3.0) for analysis. Stockfish is not part of this repository or the
  Docker image. GPL and AGPL code can be combined, but check the combined terms
  before distributing the service.
- This is a non-commercial portfolio project.

## Citation and credit

- Rendered dataset: Wölflein and Arandjelović, chesscog (OSF DOI above).
- Real photographs: Masouris and van Gemert, ChessReD, VISAPP 2024 (DOI above).
- BibTeX entries: [`CITATIONS.bib`](CITATIONS.bib).
- Detector: Ultralytics YOLOv8.
