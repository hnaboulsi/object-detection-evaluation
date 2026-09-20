"""Camera configuration with explicit pixel-space intrinsics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("Camera intrinsics must be finite")
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("Camera resolution must use integers")
        if self.width < 1 or self.height < 1:
            raise ValueError("Camera width and height must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths must be positive")
        if not (0 <= self.cx < self.width and 0 <= self.cy < self.height):
            raise ValueError("Camera principal point must lie inside the image")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CameraIntrinsics:
        required = {"width", "height", "fx", "fy", "cx", "cy"}
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"Camera config is missing: {', '.join(missing)}")
        return cls(
            width=values["width"],
            height=values["height"],
            fx=float(values["fx"]),
            fy=float(values["fy"]),
            cx=float(values["cx"]),
            cy=float(values["cy"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> CameraIntrinsics:
        resolved = Path(path).expanduser().resolve()
        try:
            values = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read camera config {resolved}: {error}") from error
        if not isinstance(values, dict):
            raise ValueError("Camera config must be a JSON object")
        return cls.from_dict(values)


@dataclass(frozen=True)
class CameraPose:
    position: Vector3
    look_at: Vector3

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "look_at": list(self.look_at),
        }
