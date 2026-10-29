"""Shared fixtures. The models are loaded once per test session because
loading both takes a few seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def models():
    from predict import load_models

    return load_models(ROOT / "models")


@pytest.fixture()
def tiny_image() -> Image.Image:
    """A generated board-like image, so no dataset images live in the repo."""
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    for row in range(4):
        for col in range(4):
            if (row + col) % 2:
                draw.rectangle(
                    [col * 60 + 40, row * 50 + 20, col * 60 + 99, row * 50 + 69],
                    fill="black",
                )
    return img
