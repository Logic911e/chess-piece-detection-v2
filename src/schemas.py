"""Response models. They double as the API documentation at /docs."""

from __future__ import annotations

from typing import Literal

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
    auto_corners: bool = Field(
        default=False, description="True if the server can find the board corners itself"
    )
    engine: bool = Field(default=False, description="True if /analyse has a chess engine to use")


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
    corners: dict[str, list[float]] = Field(
        description="The corners used, in pixels of the uploaded image: bottom_left is a1, "
        "bottom_right h1, top_right h8, top_left a8. When they were found automatically "
        "this is where the board was placed, so a client can show them for adjusting."
    )
    corners_source: Literal["given", "auto"]
    alignment_error: float | None = Field(
        default=None,
        description="Only for automatic corners: the mean distance, in squares, from each piece "
        "to the centre of its square. Lower means the grid sits better on the pieces.",
    )
    low_confidence: bool | None = Field(
        default=None,
        description="Only for automatic corners: true when alignment_error is high enough that "
        "the reading is less reliable than most. A weak signal; it cannot see a board that is "
        "placed well but turned half a revolution.",
    )
    pieces: int = Field(description="Number of pieces placed on the board")
    warnings: list[str] = Field(
        default_factory=list,
        description="Reasons the placement cannot be a real position, e.g. a missing king. "
        "Usually a sign the photo or corners were misread.",
    )
    ignored_detections: int = Field(
        description="Detections not placed: outside the corners, or beaten by a more "
        "confident detection on the same square"
    )
    inference_ms: float


class AnalyseRequest(BaseModel):
    fen: str = Field(min_length=1, max_length=120, examples=["8/8/8/4p3/4P3/8/8/4K2k w - - 0 1"])
    seconds: float = Field(default=1.0, ge=0.1, le=5.0, description="How long the engine thinks")


class AnalyseResponse(BaseModel):
    engine: str
    fen: str
    score_kind: str = Field(description="'cp' for centipawns or 'mate'")
    score_value: int = Field(description="From White's point of view")
    score_text: str = Field(examples=["+0.35", "#-3"])
    best_move_uci: str | None
    best_move_san: str | None
    line: str = Field(description="The engine's main line, in algebraic notation")
    depth: int
