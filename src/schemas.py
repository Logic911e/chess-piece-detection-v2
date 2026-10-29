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
