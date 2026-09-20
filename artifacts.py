"""Small, shared helpers for verified experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"Expected an object in {path}")
    return result


def safe_artifact_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes dataset: {relative}")
    return path


def validate_instances(instances: list[dict[str, Any]]) -> None:
    if not isinstance(instances, list):
        raise ValueError("Ground truth must be a list of instances")
    seen = set()
    for item in instances:
        if not isinstance(item, dict):
            raise ValueError("Each ground-truth instance must be an object")
        identifier = item.get("instance_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Instance IDs must be nonempty and unique")
        seen.add(identifier)
        if not isinstance(item.get("label"), str) or not item["label"].strip():
            raise ValueError(f"Instance {identifier} requires a label")
        if type(item.get("visible")) is not bool:
            raise ValueError(f"Instance {identifier} requires a Boolean visible field")
        box = item.get("bbox")
        if item["visible"]:
            validate_box(box)
        elif box is not None:
            raise ValueError(f"Invisible instance {identifier} must have bbox=null")


def validate_scene_instances(instances: list[dict[str, Any]]) -> None:
    """Validate the additional fields required in rendered scene schema v2."""
    validate_instances(instances)
    for item in instances:
        identifier = item["instance_id"]
        visible_pixels = item.get("visible_pixels")
        if not isinstance(visible_pixels, int) or visible_pixels < 0:
            raise ValueError(f"Instance {identifier} requires a nonnegative visible_pixels count")
        if item["visible"] != (visible_pixels > 0):
            raise ValueError(f"Instance {identifier} has inconsistent visibility fields")
        position = item.get("position")
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 3
            or not all(_is_finite_number(value) for value in position)
        ):
            raise ValueError(f"Instance {identifier} requires a finite 3D position")
        yaw = item.get("yaw_deg")
        if not _is_finite_number(yaw):
            raise ValueError(f"Instance {identifier} requires a finite yaw_deg")
        for field in ("asset_id", "asset_path", "role"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Instance {identifier} requires a nonempty {field}")


def validate_box(box: Any) -> None:
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or not all(_is_finite_number(value) for value in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"Invalid pixel-space xyxy box: {box}")


def validate_detections(detections: list[dict[str, Any]]) -> None:
    if not isinstance(detections, list):
        raise ValueError("Detections must be a list")
    for item in detections:
        if not isinstance(item, dict):
            raise ValueError("Each detection must be an object")
        validate_box(item.get("bbox"))
        score = item.get("score")
        if not _is_finite_number(score) or not 0 <= score <= 1:
            raise ValueError("Detection scores must be finite values in [0, 1]")
        if not isinstance(item.get("label"), str) or not item["label"].strip():
            raise ValueError("Detection label cannot be empty")
        if not isinstance(item.get("model_name"), str) or not item["model_name"].strip():
            raise ValueError("Detection model_name cannot be empty")


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)
