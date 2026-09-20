"""Metric and failure-analysis functions for saved detector runs."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from artifacts import (
    SCHEMA_VERSION,
    json_hash,
    read_json,
    validate_detections,
    validate_scene_instances,
)
from detectors.base import Detection
from results import bbox_iou, canonical_label, evaluate_instances

REPORTING_THRESHOLD = 0.35
CONFIDENCE_SWEEP = tuple(round(value / 100, 2) for value in range(10, 91, 5))

Record = dict[str, Any]


def load_results(run: Path, manifest_hash: str) -> list[Record]:
    """Load and validate every result belonging to one detector run."""
    run_info = read_json(run / "run.json")
    records: list[Record] = []
    seen: set[str] = set()
    for path in sorted((run / "results").glob("*/result.json")):
        result = read_json(path)
        _validate_result(result, path, run_info, manifest_hash)
        scene_id = result["scene_id"]
        if scene_id in seen:
            raise ValueError(f"Duplicate result for {scene_id}")
        seen.add(scene_id)
        result["_result_path"] = str(path)
        records.append(result)
    return records


def _validate_result(
    result: Record,
    path: Path,
    run_info: Record,
    manifest_hash: str,
) -> None:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema: {path}")
    if result.get("manifest_hash") != manifest_hash:
        raise ValueError(f"Result belongs to a different manifest: {path}")
    if result.get("model_hash") != run_info.get("model_hash"):
        raise ValueError(f"Mixed model configurations in results: {path}")
    if json_hash(result.get("detector")) != result.get("model_hash"):
        raise ValueError(f"Detector metadata hash mismatch: {path}")
    validate_scene_instances(result.get("ground_truth"))
    validate_detections(result.get("detections"))
    stored_evaluation = result.get("evaluation_at_0_35")
    if evaluate_record(result, REPORTING_THRESHOLD) != stored_evaluation:
        raise ValueError(f"Stored evaluation disagrees with predictions: {path}")


def evaluate_record(record: Record, threshold: float) -> Record:
    detections = [
        Detection(
            label=item["label"],
            score=float(item["score"]),
            bbox=tuple(float(value) for value in item["bbox"]),
            model_name=item["model_name"],
        )
        for item in record["detections"]
    ]
    return evaluate_instances(
        detections,
        record["ground_truth"],
        record["prompts"],
        confidence_threshold=threshold,
    )


def aggregate(records: list[Record], threshold: float) -> Record:
    true_positives = false_positives = false_negatives = 0
    matched_ious: list[float] = []
    inference_times: list[float] = []
    for record in records:
        evaluation = evaluate_record(record, threshold)
        true_positives += evaluation["true_positives"]
        false_positives += evaluation["false_positives"]
        false_negatives += evaluation["false_negatives"]
        matched_ious.extend(match["iou"] for match in evaluation["matches"])
        inference_times.append(float(record["inference_time_seconds"]))

    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        "threshold": threshold,
        "scene_count": len(records),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives_per_image": _safe_ratio(false_positives, len(records)),
        "median_matched_iou": statistics.median(matched_ious) if matched_ious else 0.0,
        "median_inference_time_seconds": (
            statistics.median(inference_times) if inference_times else 0.0
        ),
    }


def grouped_metrics(records: list[Record]) -> list[Record]:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        case = record["case"]
        groups[("target_class", case["target_label"])].append(record)
        groups[("asset", case["target_asset_id"])].append(record)
        groups[("factor_level", f"{case['factor']}:{case['level']}")].append(record)

    rows: list[Record] = []
    for (dimension, value), group in sorted(groups.items()):
        metrics = aggregate(group, REPORTING_THRESHOLD)
        metrics["ap50"] = detection_ap50(group)
        rows.append({"dimension": dimension, "value": value, **metrics})
    return rows


def detection_ap50(records: list[Record]) -> float | None:
    """Return 101-point interpolated AP50 over every saved detection score."""
    scored: list[tuple[float, str, int, int]] = []
    total_truth = 0
    for record in records:
        evaluation = evaluate_record(record, 0.0)
        matched = {match["detection_index"] for match in evaluation["matches"]}
        total_truth += evaluation["true_positives"] + evaluation["false_negatives"]
        for index, prediction in enumerate(record["detections"]):
            scored.append(
                (float(prediction["score"]), record["scene_id"], index, int(index in matched))
            )
    if total_truth == 0:
        return None

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    true_positives = 0
    precision_recall = []
    for rank, (_, _, _, is_match) in enumerate(scored, start=1):
        true_positives += is_match
        precision_recall.append((true_positives / total_truth, true_positives / rank))
    return average_precision(precision_recall)


def average_precision(points: list[tuple[float, float]]) -> float:
    return (
        sum(
            max((precision for recall, precision in points if recall >= level), default=0.0)
            for level in (index / 100 for index in range(101))
        )
        / 101.0
    )


def failure_patterns(records: list[Record]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        evaluation = evaluate_record(record, REPORTING_THRESHOLD)
        truth = _prompted_visible_truth(record)
        _classify_false_positives(record, evaluation, truth, counts)
        _classify_false_negatives(record, evaluation, truth, counts)
    return dict(sorted(counts.items()))


def _prompted_visible_truth(record: Record) -> list[Record]:
    prompted = {canonical_label(label) for label in record["prompts"]}
    return [
        item
        for item in record["ground_truth"]
        if item["visible"] and canonical_label(item["label"]) in prompted
    ]


def _classify_false_positives(
    record: Record,
    evaluation: Record,
    truth: list[Record],
    counts: Counter[str],
) -> None:
    for index in evaluation["unmatched_detection_indices"]:
        prediction = record["detections"][index]
        overlaps = [(bbox_iou(prediction["bbox"], item["bbox"]), item) for item in truth]
        overlap, item = max(overlaps, key=lambda pair: pair[0], default=(0.0, None))
        if overlap >= 0.5:
            labels_match = canonical_label(prediction["label"]) == canonical_label(item["label"])
            reason = "duplicate_fp" if labels_match else "wrong_label_fp"
        else:
            reason = "localization_fp" if overlap >= 0.1 else "background_fp"
        counts[reason] += 1


def _classify_false_negatives(
    record: Record,
    evaluation: Record,
    truth: list[Record],
    counts: Counter[str],
) -> None:
    truth_by_id = {item["instance_id"]: item for item in truth}
    for identifier in evaluation["unmatched_instance_ids"]:
        item = truth_by_id[identifier]
        candidates = [
            prediction
            for prediction in record["detections"]
            if canonical_label(prediction["label"]) == canonical_label(item["label"])
        ]
        if any(
            bbox_iou(prediction["bbox"], item["bbox"]) >= 0.5
            and prediction["score"] < REPORTING_THRESHOLD
            for prediction in candidates
        ):
            reason = "below_threshold_fn"
        elif any(
            bbox_iou(prediction["bbox"], item["bbox"]) >= 0.1
            and prediction["score"] >= REPORTING_THRESHOLD
            for prediction in candidates
        ):
            reason = "localization_fn"
        else:
            reason = "no_saved_matching_candidate_fn"
        counts[reason] += 1


def paired_conditions(records: list[Record]) -> list[Record]:
    baselines = {
        record["case"]["target_asset_id"]: evaluate_record(record, REPORTING_THRESHOLD)
        for record in records
        if record["case"]["factor"] == "baseline"
    }
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        case = record["case"]
        if case["factor"] in {"baseline", "target_presence"}:
            continue
        if case["target_asset_id"] in baselines:
            groups[f"{case['factor']}:{case['level']}"].append(record)

    rows = []
    for condition, group in sorted(groups.items()):
        new_misses = recoveries = false_positive_delta = 0
        for record in group:
            baseline = baselines[record["case"]["target_asset_id"]]
            evaluation = evaluate_record(record, REPORTING_THRESHOLD)
            baseline_detected = _target_was_matched(baseline)
            condition_detected = _target_was_matched(evaluation)
            new_misses += int(baseline_detected and not condition_detected)
            recoveries += int(condition_detected and not baseline_detected)
            false_positive_delta += evaluation["false_positives"] - baseline["false_positives"]
        rows.append(
            {
                "condition": condition,
                "pairs": len(group),
                "new_misses": new_misses,
                "recoveries": recoveries,
                "false_positive_delta": false_positive_delta,
            }
        )
    return rows


def _target_was_matched(evaluation: Record) -> bool:
    return any(match["instance_id"] == "target" for match in evaluation["matches"])


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
