"""Small shared helpers for saved experiment results."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from detectors.base import Detection

SCHEMA_VERSION = 1


def bbox_iou(first: Sequence[float] | None, second: Sequence[float] | None) -> float:
    if first is None or second is None:
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_single_target(
    detections: Sequence[Detection],
    ground_truth: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match at most one detection to the one simulated or manually labeled target."""
    target_visible = bool(ground_truth.get("target_visible", False))
    target_bbox = ground_truth.get("bbox")
    target_label = ground_truth.get("label")
    overlaps = [
        bbox_iou(detection.bbox, target_bbox)
        if _labels_match(detection.label, target_label)
        else 0.0
        for detection in detections
    ]
    best_index = (
        max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else None
    )
    best_iou = overlaps[best_index] if best_index is not None else 0.0
    matched = bool(target_visible and best_iou >= iou_threshold)
    return {
        "iou_threshold": iou_threshold,
        "matched_detection_index": best_index if matched else None,
        "best_iou": best_iou,
        "true_positives": int(matched),
        "false_positives": len(detections) - int(matched),
        "false_negatives": int(target_visible and not matched),
    }


def _labels_match(detection_label: str, target_label: str | None) -> bool:
    if not target_label:
        return False
    detected = set(re.findall(r"[a-z0-9]+", detection_label.lower()))
    target = set(re.findall(r"[a-z0-9]+", target_label.lower()))
    return bool(detected and target and (detected <= target or target <= detected))


def draw_annotations(
    image: Image.Image,
    detections: Sequence[Detection],
    ground_truth_bbox: Sequence[float] | None = None,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    if ground_truth_bbox is not None:
        draw.rectangle(tuple(ground_truth_bbox), outline="cyan", width=3)
        draw.text(
            (float(ground_truth_bbox[0]) + 3, float(ground_truth_bbox[1]) + 3),
            "ground truth",
            fill="cyan",
            stroke_width=2,
            stroke_fill="black",
        )
    for detection in detections:
        x_min, y_min, x_max, y_max = detection.bbox
        draw.rectangle((x_min, y_min, x_max, y_max), outline="lime", width=3)
        draw.text(
            (x_min + 3, max(0.0, y_min - 15)),
            f"{detection.label} {detection.score:.2f}",
            fill="lime",
            stroke_width=2,
            stroke_fill="black",
        )
    return annotated


def make_record(
    *,
    source: dict[str, Any],
    scenario: dict[str, Any] | None,
    detector: dict[str, Any],
    prompts: Sequence[str],
    detections: Sequence[Detection],
    ground_truth: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    inference_time_seconds: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    detector_info = dict(detector)
    detector_info["prompts"] = list(prompts)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "scenario": scenario,
        "detector": detector_info,
        "ground_truth": ground_truth,
        "detections": [asdict(detection) for detection in detections],
        "evaluation": evaluation,
        "inference_time_seconds": inference_time_seconds,
        "metadata": metadata,
    }


def create_output_directory(path: Path) -> None:
    """Create one result directory but never mix with existing result files."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rebuild_summary(output_root: Path) -> Path:
    """Build summary.csv solely from independent result.json files."""
    rows: list[dict[str, Any]] = []
    for result_path in sorted(output_root.glob("*/result.json")):
        record = json.loads(result_path.read_text(encoding="utf-8"))
        scenario = record.get("scenario") or {}
        ground_truth = record.get("ground_truth") or {}
        evaluation = record.get("evaluation") or {}
        detections = record.get("detections") or []
        rows.append(
            {
                "id": record["source"]["id"],
                "source_type": record["source"]["type"],
                "room": scenario.get("room"),
                "seed": scenario.get("seed"),
                "target_type": scenario.get("target_type") or ground_truth.get("label"),
                "target_present": ground_truth.get("target_present"),
                "target_visible": ground_truth.get("target_visible"),
                "detection_count": len(detections),
                "best_confidence": max(
                    (item["score"] for item in detections), default=0.0
                ),
                "best_iou": evaluation.get("best_iou"),
                "true_positives": evaluation.get("true_positives"),
                "false_positives": evaluation.get("false_positives"),
                "false_negatives": evaluation.get("false_negatives"),
                "inference_time_seconds": record.get("inference_time_seconds"),
                "result_path": str(result_path.relative_to(output_root)),
            }
        )
    summary_path = output_root / "summary.csv"
    fieldnames = [
        "id",
        "source_type",
        "room",
        "seed",
        "target_type",
        "target_present",
        "target_visible",
        "detection_count",
        "best_confidence",
        "best_iou",
        "true_positives",
        "false_positives",
        "false_negatives",
        "inference_time_seconds",
        "result_path",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path
