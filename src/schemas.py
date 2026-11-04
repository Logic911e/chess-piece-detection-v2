"""Response models. They double as the API documentation at /docs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_name: str = Field(examples=["white-knight"])
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(
        min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixels of the uploaded image"
    )


class PredictResponse(BaseModel):
    model: str
    image_width: int
    image_height: int
    detections: list[Detection]
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    models: list[str]


class PositionResponse(BaseModel):
    model: str
    image_width: int
    image_height: int
    squares: dict[str, str] = Field(
        description="Square to piece, e.g. {'e4': 'white-pawn'}. Empty squares are left out.",
        examples=[{"e4": "white-pawn", "e5": "black-pawn"}],
    )
    fen_placement: str = Field(
        description="Piece-placement field of a FEN string. Side to move, castling and "
        "move counters cannot be read from a photo.",
        examples=["8/8/8/4p3/4P3/8/8/8"],
    )
    pieces: int = Field(description="Number of pieces placed on the board")
    ignored_detections: int = Field(
        description="Detections not placed: outside the corners, or beaten by a more "
        "confident detection on the same square"
    )
    inference_ms: float
