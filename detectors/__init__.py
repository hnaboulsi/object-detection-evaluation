"""Object detector adapters with one shared output type."""

from .base import Detection, Detector
from .create import create_detector

__all__ = ["Detection", "Detector", "create_detector"]
