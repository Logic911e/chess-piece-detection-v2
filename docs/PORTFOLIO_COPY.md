# Portfolio copy (drafts, not posted)

Nothing here has been posted. Before using any of it:

- Replace the bracketed placeholders. `[LIVE DEMO LINK]` only exists once you
  have deployed, so delete the sentences that mention it if you have not.
- Every number below comes from `RESULTS.md` or the report CSVs. Keep it that way
  if you edit.
- Change anything that does not sound like you. These are starting points.

## LinkedIn post

I took a course project and turned it into something that runs as a service.

The project started with a question: what happens when a chess piece detector
trained only on computer-generated images meets real photos? I trained a YOLOv8
model on rendered boards. On rendered test images it scored 0.9366 mAP50-95. On
306 real smartphone photos it scored 0.1895.

Then I fine-tuned it on real photos. Two photos took the score to 0.4330 on
average over three random draws. With 50 photos it reached 0.8262, and with all
1,442 it reached 0.9029. The spread between draws was large at two photos and
small by fifty, so a handful of photos gives an unreliable result and fifty gives
a dependable one. (Each run's best checkpoint was chosen on the 330 validation photos, so
the small counts also used those labels for selection.)

The second half of the work was making it run as a service. The model now sits
behind a FastAPI service in a Docker container, with tests, linting and a
continuous integration workflow, and the training runs are logged in MLflow. I
wrote a small web page that sends a photo to both models and draws their boxes
side by side, so the gap is easy to see.

[LIVE DEMO LINK, if deployed]
Code: [GITHUB LINK]

#computervision #yolo #machinelearning #docker #fastapi

## CV project entry

**Chess piece detection: from notebook to tested service** | Python, YOLOv8, FastAPI, Docker, MLflow | [GITHUB LINK]

- Trained a YOLOv8s detector on rendered chess images (0.9366 mAP50-95) and measured the domain gap on 306 real photographs (0.1895), then fine-tuned on real data to reach 0.9029.
- Ran a data-efficiency study with 3 seeds up to 50 photos: 2 photos lifted the score to 0.4330 and 50 to 0.8262, and the spread between seeds fell from 0.1085 to 0.0137.
- Packaged both models as a FastAPI service in a CPU-only Docker container with upload validation, 20 pytest tests, ruff, GitHub Actions CI and MLflow run tracking, plus a static comparison page.

## Short version for a recruiter message

I built a chess piece detector, measured how badly it breaks on real photos when
trained only on rendered ones, and wrapped it in a tested Docker service. The
code and a write-up of the results are here: [GITHUB LINK].

## Notes for you

- The post says "I fine-tuned it on real photos" and the entry says "reach
  0.9029". Both are single-run figures for the full 1,442 photos, and `RESULTS.md`
  says so. Keep that in mind if someone asks.
- The 8 of 50 epochs shortfall in the base training is not in the post. If you
  are asked about the 79.8% drop, say it may be slightly overstated for that
  reason. The model card says this.
- Claude Code wrote much of the service code, tests and docs. You need to be able to
  explain every part in an interview, and you should say so if asked how it was built.
- Add what you were responsible for if the project is discussed in an interview.
  The copy does not claim a team, a user base or a production deployment, and
  you should not add one unless it is true.
- Add the live link, and the Cloud Run mention, only after you have deployed and
  tested the service yourself.
