# Portfolio copy (drafts, not posted)

Nothing here has been posted. Before using any of it:

- Replace the bracketed placeholders. `[LIVE DEMO LINK]` only exists once you
  have deployed, so delete the sentences that mention it if you have not.
- Every number below comes from `RESULTS.md`, `docs/POSITION.md` or the report CSVs. Keep it
  that way if you edit.
- Change anything that does not sound like you. These are starting points.
- Re-check the numbers against `reports/*.csv` first. They have changed during the project.

## LinkedIn post

I took a course project and turned it into a service that reads a chess position from a photo.

The course project asked what happens when a chess piece detector trained only on
computer-generated images meets real photos. A YOLOv8 model trained on renders scored
0.9366 mAP50-95 on rendered test images and 0.1895 on 306 real smartphone photos.
Fine-tuning on 1,442 real photos brought it to 0.9029.

What I built afterwards: given one photo, find the board, work out which corner is a1, put
every piece on its square and write the FEN. The first version needed the four board corners
marked by hand, and trying it on my phone showed how a small corner error moves pieces onto
the wrong squares. So I trained a second model to find the corners, moved them to fit the
detected pieces, and used the checkerboard pattern and the piece colours to name a1.

On the ChessReD photos it gets the whole position exactly right in 85.4% of the 636
validation and test photos pooled (91.2% on test, 80.0% on validation). Those photos come
from six games, results range from 78% to 95% between games, and I tuned the method while
looking at both splits, so a new board or camera may do worse. I have not measured it on my
own phone yet.

Several things I tried did not help: a larger corner model, a second stage that finds the
corners again on a crop, and extra image augmentation. They are written up in the repo with
their numbers.

It runs as a FastAPI service in Docker with 99 tests, linting and continuous integration, and
a web page where you upload a photo and check the corners before reading the position.

[LIVE DEMO LINK, if deployed]
Code: [GITHUB LINK]

#computervision #yolo #machinelearning #docker #fastapi

## CV project entry

**Chess position from a photo: detector, corner finding and a tested service** | Python, YOLOv8, OpenCV, SciPy, FastAPI, Docker | [GITHUB LINK]

- Trained a YOLOv8s piece detector on rendered images (0.9366 mAP50-95) and measured the domain gap on 306 real photographs (0.1895), then fine-tuned on real photos to reach 0.9029.
- Built the photo-to-FEN pipeline: a pose model finds the board corners, a least-squares step fits them to the detected pieces from 11 starting points, and the checkerboard and piece colours name a1. Whole-position exact match with found corners: 85.4% pooled over 636 validation and test photos from six games (78% to 95% by game), against 95.4% with annotated corners.
- Recorded the experiments that failed (larger corner model, crop-based second stage, geometric augmentation), and corrected my own early claims after an adversarial review of the repo.
- Packaged it as a FastAPI service in a CPU-only Docker container with upload limits, 99 pytest tests, ruff, mypy, GitHub Actions CI and MLflow run tracking, plus a web page for checking the corners.

## Short version for a recruiter message

I built a service that reads a chess position from a photo: a YOLOv8 piece detector, a
corner-finding model and a fitting step, behind a tested Docker API. It gets the whole position
right in about 85% of ChessReD validation and test photos, from six games, and I have written up
what did and did not work. The code and results are here: [GITHUB LINK].

## Notes for you

- The 95.4% with annotated corners is 607 of 636 photos (311 of 330 on validation, 296 of 306 on
  test). Check it against `reports/pipeline_eval.csv` before using it.
- The fine-tuning figures were produced with checkpoints chosen on the 330 validation photos,
  and the full-data figure is a single run. `RESULTS.md` and the model card say so. If asked
  about "two photos", say that.
- The 8 of 50 epochs shortfall in the base training is not in the post. If you are asked about
  the 79.8% drop, say it may be slightly overstated for that reason. The model card says this.
- The multi-start step was designed while both the validation and test results were visible, and
  no split was kept untouched. Say that if asked about the test number.
- Claude Code wrote much of the service code, tests and docs. You need to be able to explain
  every part in an interview, and you should say so if asked how it was built.
- Add what you were responsible for if the project is discussed in an interview. The copy does
  not claim a team, a user base or a production deployment, and you should not add one unless it
  is true.
- Add the live link, and any Cloud Run mention, only after you have deployed and tested the
  service yourself.
- Three questions you should be ready for: what the early-stopping set was for the two-photo
  result, what accuracy to expect on a new phone and board and how the service signals when it is
  wrong (its confidence signal is weak: `docs/POSITION.md`), and why the pooled
  figure leads when the test split alone looks better.
