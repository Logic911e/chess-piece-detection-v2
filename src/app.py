"""FastAPI service: GET /health, POST /predict, POST /position and POST /analyse.

The pages in web/ are served at /app/ so a page and its API share one address.

Run locally:  uvicorn app:app --app-dir src --port 8080
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import chess.engine
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from analysis import analyse, fen_problems, find_engine, placement_problems
from position import detections_to_position, fen_placement, parse_corners
from predict import load_models, predict
from schemas import (
    AnalyseRequest,
    AnalyseResponse,
    HealthResponse,
    PositionResponse,
    PredictResponse,
)

log = logging.getLogger("chess-api")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
WEB_DIR = Path(os.environ.get("WEB_DIR", ROOT / "web"))
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


def read_upload(file: UploadFile) -> Image.Image:
    """Validate an uploaded image and decode it. Shared by the image endpoints."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPEG or PNG image")

    # Read one byte past the limit so an oversize file is detected without
    # loading all of it.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    try:
        image = Image.open(io.BytesIO(data))
        # Phones store a rotated photo as upright pixels plus an orientation tag.
        # Browsers apply the tag when they show the photo, so apply it here too,
        # otherwise boxes and corners would not line up with what the user saw.
        return ImageOps.exif_transpose(image).convert("RGB")
    except Exception:  # corrupt data, truncated file, decompression bomb, ...
        raise HTTPException(400, "Could not read the file as an image") from None


# A plain `def` endpoint runs in a worker thread, so CPU-heavy inference
# does not block the event loop.
@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG or PNG, up to 5 MB")],
    model: Annotated[Literal["synthetic", "finetuned"], Query()] = "finetuned",
    conf: Annotated[float, Query(ge=0.0, le=1.0, description="Minimum confidence")] = 0.25,
):
    image = read_upload(file)
    return predict(request.app.state.models, model, image, conf)


@app.post("/position", response_model=PositionResponse)
def position_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG or PNG, up to 5 MB")],
    corners: Annotated[
        str,
        Form(
            description="JSON with the board corners in pixels of the uploaded image, "
            'for example {"bottom_left": [x, y], "bottom_right": [x, y], '
            '"top_right": [x, y], "top_left": [x, y]}. bottom_left is the a1 corner, '
            "bottom_right h1, top_right h8 and top_left a8."
        ),
    ],
    model: Annotated[Literal["synthetic", "finetuned"], Query()] = "finetuned",
    conf: Annotated[float, Query(ge=0.0, le=1.0, description="Minimum confidence")] = 0.5,
    anchor_y: Annotated[
        float, Query(ge=0.5, le=1.0, description="Where on a box the piece stands")
    ] = 0.9,
):
    try:
        board_corners = parse_corners(corners)
    except ValueError as err:
        raise HTTPException(422, str(err)) from None

    image = read_upload(file)
    result = predict(request.app.state.models, model, image, conf)
    squares = detections_to_position(result["detections"], board_corners, anchor_y)
    placement = fen_placement(squares)
    return {
        "model": model,
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "squares": squares,
        "fen_placement": placement,
        "pieces": len(squares),
        "warnings": placement_problems(placement),
        "ignored_detections": len(result["detections"]) - len(squares),
        "inference_ms": result["inference_ms"],
    }


@app.post("/analyse", response_model=AnalyseResponse)
def analyse_endpoint(body: AnalyseRequest):
    engine_path = find_engine()
    if engine_path is None:
        raise HTTPException(
            503,
            "No chess engine is set up on this server. Set STOCKFISH_PATH or install Stockfish.",
        )
    try:
        problems = fen_problems(body.fen)
    except ValueError:
        raise HTTPException(422, "That is not a valid FEN") from None
    if problems:
        raise HTTPException(422, "This position cannot be analysed. " + " ".join(problems))
    try:
        return analyse(engine_path, body.fen, body.seconds)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError):
        log.exception("Engine failed")
        raise HTTPException(502, "The chess engine failed to answer") from None


# Serve the pages from the API so a page and its API share one address. That
# avoids cross-site calls (and CORS) and lets one tunnel or one Cloud Run URL
# carry everything. Skipped if the folder is absent.
if WEB_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/app/")
