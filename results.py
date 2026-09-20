"""Small shared helpers for saved experiment results."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageDraw

from detectors.base import Detection

LABEL_ALIASES = {
    "sofa": "couch",
    "couch": "couch",
    "chair": "chair",
    "bed": "bed",
    "table": "dining table",
    "dining table": "dining table",
}


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


def evaluate_instances(
    detections: Sequence[Detection],
    ground_truth: Sequence[dict[str, Any]],
    prompts: Sequence[str],
    *,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedily match score-ordered predictions to visible prompted instances."""
    prompted = {canonical_label(prompt) for prompt in prompts}
    truth = [
        item
        for item in ground_truth
        if item.get("visible")
        and item.get("bbox") is not None
        and canonical_label(str(item.get("label", ""))) in prompted
    ]
    predictions = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection.score >= confidence_threshold
    ]
    predictions.sort(key=lambda item: item[1].score, reverse=True)
    unmatched_truth = set(range(len(truth)))
    matches: list[dict[str, Any]] = []
    unmatched_predictions: list[int] = []
    best_unmatched_ious: list[float] = []
    for original_index, detection in predictions:
        label = canonical_label(detection.label)
        candidates = [
            (truth_index, bbox_iou(detection.bbox, truth[truth_index]["bbox"]))
            for truth_index in sorted(unmatched_truth)
            if canonical_label(str(truth[truth_index]["label"])) == label
        ]
        best = max(candidates, key=lambda item: item[1], default=None)
        if best is not None and best[1] >= iou_threshold:
            truth_index, overlap = best
            unmatched_truth.remove(truth_index)
            matches.append(
                {
                    "detection_index": original_index,
                    "instance_id": truth[truth_index]["instance_id"],
                    "iou": overlap,
                }
            )
        else:
            unmatched_predictions.append(original_index)
            best_unmatched_ious.append(best[1] if best is not None else 0.0)
    tp = len(matches)
    fp = len(unmatched_predictions)
    fn = len(unmatched_truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "unmatched_detection_indices": unmatched_predictions,
        "unmatched_instance_ids": [
            truth[index]["instance_id"] for index in sorted(unmatched_truth)
        ],
        "best_iou_for_unmatched_predictions": best_unmatched_ious,
    }


def draw_instance_annotations(
    image: Image.Image,
    detections: Sequence[Detection],
    ground_truth: Sequence[dict[str, Any]],
    *,
    confidence_threshold: float = 0.35,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for item in ground_truth:
        bbox = item.get("bbox")
        if bbox is None:
            continue
        draw.rectangle(tuple(bbox), outline="cyan", width=3)
        draw.text(
            (float(bbox[0]) + 3, float(bbox[1]) + 3),
            f"GT {item['label']}:{item['instance_id']}",
            fill="cyan",
            stroke_width=2,
            stroke_fill="black",
        )
    for detection in detections:
        if detection.score < confidence_threshold:
            continue
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


def canonical_label(value: str) -> str:
    cleaned = " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    return LABEL_ALIASES.get(cleaned, cleaned)
