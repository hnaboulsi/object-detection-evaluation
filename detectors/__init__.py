"""Detector contracts and the validated Grounding DINO adapter."""

from .base import Detection, Detector
from .grounding_dino import GroundingDINO

__all__ = ["Detection", "Detector", "GroundingDINO"]
