#!/usr/bin/env bash
# Assemble the folder that gets uploaded to a Hugging Face Docker Space.
# The Space is a separate repo that needs a Dockerfile, the code, the two
# weight files and a README with a YAML header. Output: build/space/
set -euo pipefail

cd "$(dirname "$0")/.."
out=build/space

for f in models/synthetic.pt models/finetuned.pt; do
    [ -f "$f" ] || { echo "Missing $f" >&2; exit 1; }
done

rm -rf "$out"
mkdir -p "$out/models"
cp Dockerfile requirements-api.txt "$out/"
cp -R src "$out/src"
find "$out/src" -name __pycache__ -type d -prune -exec rm -rf {} +
cp models/synthetic.pt models/finetuned.pt "$out/models/"

cat > "$out/README.md" <<'README'
---
title: Chess Piece Detection
sdk: docker
app_port: 8080
pinned: false
---

FastAPI service with two YOLOv8s chess piece detectors: one trained on
rendered images only, and the same model fine-tuned on real photographs.

Endpoints: `GET /health`, `POST /predict?model=synthetic|finetuned&conf=0.25`.
Interactive docs are at `/docs`. Non-commercial portfolio project.
README

echo "Space folder ready:"
du -sh "$out"
find "$out" -type f ! -path '*/src/*' | sort
