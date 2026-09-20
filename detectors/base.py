"""Shared detector contract and image/device helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

ImageInput = str | Path | Image.Image | np.ndarray


@dataclass(frozen=True)
class Detection:
    """One pixel-space detection in xyxy form."""

    label: str
    score: float
    bbox: tuple[float, float, float, float]
    model_name: str


@runtime_checkable
class Detector(Protocol):
    model_name: str

    def detect(self, image: ImageInput, prompts: Sequence[str]) -> list[Detection]: ...

    def metadata(self) -> dict[str, Any]: ...


def load_rgb_image(image: ImageInput) -> Image.Image:
    """Return a detached RGB PIL image from a path, PIL image, or RGB array."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB").copy()
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy images must have shape (height, width, 3)")
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all() or image.min() < 0:
                raise ValueError("Floating-point images must contain finite values >= 0")
            maximum = float(image.max()) if image.size else 0.0
            if maximum <= 1.0:
                image = image * 255.0
            elif maximum > 255.0:
                raise ValueError("Floating-point image values must be in [0, 1] or [0, 255]")
        array = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")
    raise TypeError("image must be a path, PIL image, or NumPy RGB array")


def clean_prompts(prompts: Sequence[str]) -> list[str]:
    cleaned = [str(prompt).strip().lower() for prompt in prompts if str(prompt).strip()]
    if not cleaned:
        raise ValueError("At least one non-empty prompt is required")
    return cleaned


def resolve_device(requested: str) -> str:
    """Resolve auto/cpu/cuda without importing any model adapter."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to select a detector device") from error

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
