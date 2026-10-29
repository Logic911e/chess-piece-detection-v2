"""FastAPI service: GET /health and POST /predict.

Run locally:  uvicorn app:app --app-dir src --port 8080
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from predict import load_models, predict
from schemas import HealthResponse, PredictResponse

log = logging.getLogger("chess-api")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}
# Only the demo page's origin may call the API from a browser. Comma-separated.
_cors_raw = os.environ.get("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request. On Cloud Run this is the cold start.
    app.state.models = load_models(MODELS_DIR)
    yield


app = FastAPI(title="Chess piece detection", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # Log the detail server-side; the client only ever sees a generic message.
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    return {"status": "ok", "models": list(request.app.state.models)}


# A plain `def` endpoint runs in a worker thread, so CPU-heavy inference
# does not block the event loop.
@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG or PNG, up to 5 MB")],
    model: Annotated[Literal["synthetic", "finetuned"], Query()] = "finetuned",
    conf: Annotated[float, Query(ge=0.0, le=1.0, description="Minimum confidence")] = 0.25,
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPEG or PNG image")

    # Read one byte past the limit so an oversize file is detected without
    # loading all of it.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:  # corrupt data, truncated file, decompression bomb, ...
        raise HTTPException(400, "Could not read the file as an image") from None

    return predict(request.app.state.models, model, image, conf)
