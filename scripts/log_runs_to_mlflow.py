"""Log the training runs to a local MLflow store, so the experiments can be browsed.

Every value is read from a file, nothing is typed in:
  * parameters and validation metrics: the training checkpoints (train_args,
    train_metrics are saved inside Ultralytics .pt files)
  * test metrics on the real photographs: reports/eval_test.csv and
    reports/finetune_sweep.csv

There is no results.csv or args.yaml for these runs (they were not kept), so
the checkpoints are the source for settings.

Usage:
    python scripts/log_runs_to_mlflow.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import csv
from pathlib import Path

import mlflow
import torch

ROOT = Path(__file__).resolve().parent.parent
DB_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

# The sweep run that produced models/finetuned.pt (all 1,442 real images).
FINETUNED_RUN = "ft-n1442-20260908-002129"

TRAIN_ARGS = (
    "model",
    "epochs",
    "batch",
    "imgsz",
    "lr0",
    "freeze",
    "seed",
    "patience",
    "optimizer",
    "device",
)
# Checkpoint key -> MLflow metric name (MLflow rejects parentheses in names).
VAL_METRICS = {
    "metrics/mAP50-95(B)": "val_mAP50-95",
    "metrics/mAP50(B)": "val_mAP50",
    "metrics/precision(B)": "val_precision",
    "metrics/recall(B)": "val_recall",
}
TEST_METRICS = ("mAP50-95", "mAP50", "precision", "recall")


def load_checkpoint(path: Path) -> dict:
    # weights_only=False because Ultralytics checkpoints contain pickled model
    # objects. Only do this for files you trained yourself.
    return torch.load(path, map_location="cpu", weights_only=False)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def log_checkpoint(ckpt: dict, extra_params: dict) -> None:
    args = ckpt.get("train_args") or {}
    mlflow.log_params({k: args.get(k) for k in TRAIN_ARGS if args.get(k) is not None})
    mlflow.log_params(extra_params)
    # val_* is measured on whichever data the run trained with, so the two runs'
    # val numbers are NOT comparable. Say so in the run itself.
    data_yaml = str(args.get("data", ""))
    mlflow.set_tag(
        "val_set", "real photographs (ChessReD)" if "chessred" in data_yaml else "synthetic renders"
    )
    metrics = ckpt.get("train_metrics") or {}
    mlflow.log_metrics({name: metrics[key] for key, name in VAL_METRICS.items() if key in metrics})


def log_test_metrics(row: dict) -> None:
    mlflow.log_metrics({f"test_real_{m}": float(row[m]) for m in TEST_METRICS})


def already_logged(experiment: str, run_name: str) -> bool:
    runs = mlflow.search_runs(
        experiment_names=[experiment], filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    return not runs.empty


def main() -> None:
    mlflow.set_tracking_uri(DB_URI)
    sweep = read_rows(ROOT / "reports" / "finetune_sweep.csv")
    overall = next(
        r for r in read_rows(ROOT / "reports" / "eval_test.csv") if r["stratum"] == "overall"
    )

    mlflow.set_experiment("chess-detection-models")

    # Run 1: trained on synthetic renders only. Its test score on real
    # photographs is the zero-shot domain-gap number.
    synth = load_checkpoint(ROOT / "models" / "synthetic_best.pt")
    if not already_logged("chess-detection-models", "yolov8s-synthetic"):
        with mlflow.start_run(run_name="yolov8s-synthetic"):
            log_checkpoint(
                synth,
                {
                    "epochs_requested": synth["train_args"]["epochs"],
                    # The saved epoch index starts at 0; Colab ended the session early.
                    "epochs_completed": synth["epoch"] + 1,
                    "trained_on": "synthetic renders",
                },
            )
            log_test_metrics(overall)
            mlflow.set_tag("test_set", f"{overall['n_images']} real photographs (zero-shot)")

    # Run 2: the same model fine-tuned on all real training images.
    fine = load_checkpoint(ROOT / "models" / "finetuned.pt")
    row = next(r for r in sweep if r["run"] == FINETUNED_RUN)
    if not already_logged("chess-detection-models", "yolov8s-finetuned-n1442"):
        with mlflow.start_run(run_name="yolov8s-finetuned-n1442"):
            log_checkpoint(
                fine,
                {
                    "trained_on": "synthetic, then 1,442 real photographs",
                    "sweep_run": FINETUNED_RUN,
                },
            )
            log_test_metrics(row)
            mlflow.set_tag("test_set", "real photographs (same test split as the zero-shot run)")

    # The data-efficiency sweep: one MLflow run per row of the CSV.
    mlflow.set_experiment("finetune-sweep")
    for r in sweep:
        name = r["run"] or "zero-shot"
        if already_logged("finetune-sweep", name):
            continue
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {k: r[k] for k in ("n_images", "seed", "epochs", "games_covered") if r[k] != ""}
            )
            log_test_metrics(r)

    print(f"Logged to {DB_URI}")
    print("View with (from the repo root): mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
