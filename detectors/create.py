"""Lazy detector factory."""

from __future__ import annotations

from .base import Detector


def create_detector(name: str, device: str = "auto", **kwargs: object) -> Detector:
    normalized = name.strip().lower()
    if normalized == "grounding_dino":
        from .grounding_dino import GroundingDINO

        return GroundingDINO(device=device, **kwargs)
    if normalized == "yolo_world":
        from .yolo_world import YOLOWorld

        return YOLOWorld(device=device, **kwargs)
    if normalized == "sam3":
        from .sam3 import SAM3

        return SAM3(device=device, **kwargs)
    raise ValueError(f"Unknown detector {name!r}. Choose grounding_dino, yolo_world, or sam3.")
