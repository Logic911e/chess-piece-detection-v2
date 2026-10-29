"""Model loading and inference for the API.

Deliberately free of FastAPI imports and of src/config.py (which creates
directories on import), so it can be tested and reused on its own.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

# The two models the service exposes. Keys are what clients pass as ?model=.
MODEL_FILES = {
    "synthetic": "synthetic.pt",
    "finetuned": "finetuned.pt",
}


def load_models(models_dir: Path) -> dict:
    """Load every model once, at startup. Loading per request would add
    seconds to each call."""
    # Imported here so importing this module stays cheap (ultralytics pulls
    # in torch).
    from ultralytics import YOLO

    models = {}
    for name, filename in MODEL_FILES.items():
        path = Path(models_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing weights for '{name}': {path}")
        models[name] = YOLO(str(path))
    return models


def predict(models: dict, name: str, image: Image.Image, conf: float = 0.25) -> dict:
    """Run one model on one image and return plain Python types."""
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(models)}")

    model = models[name]
    start = time.perf_counter()
    result = model.predict(image, conf=conf, device="cpu", verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = [
        {
            "class_name": model.names[int(cls)],
            "confidence": round(float(score), 4),
            "bbox": [round(float(v), 1) for v in box],
        }
        for box, score, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
            strict=True,
        )
    ]
    return {
        "model": name,
        "image_width": image.width,
        "image_height": image.height,
        "detections": detections,
        "inference_ms": round(elapsed_ms, 1),
    }
