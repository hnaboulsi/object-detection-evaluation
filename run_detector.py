"""Run one detector over an immutable rendered experiment manifest."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from artifacts import (
    SCHEMA_VERSION,
    file_hash,
    json_hash,
    read_json,
    safe_artifact_path,
    validate_detections,
    validate_scene_instances,
    write_json,
)
from detectors import create_detector
from experiment import read_manifest
from results import draw_instance_annotations, evaluate_instances

REPORTING_THRESHOLD = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/controlled_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dino_controlled_v1"))
    parser.add_argument(
        "--detector",
        choices=("grounding_dino", "yolo_world", "sam3"),
        default="grounding_dino",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Run only the first N scenes for smoke tests")
    parser.add_argument("--box-threshold", type=float, default=0.10)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = read_manifest(dataset / "manifest.json")
    detector_kwargs: dict[str, object] = {}
    if args.detector == "grounding_dino":
        detector_kwargs = {
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
        }
    detector = create_detector(args.detector, device=args.device, **detector_kwargs)
    detector_metadata = detector.metadata()
    model_hash = json_hash(detector_metadata)
    output.mkdir(parents=True, exist_ok=True)
    _prepare_run_manifest(output, manifest, detector_metadata, model_hash)

    cases = manifest["cases"]
    if args.limit is not None:
        cases = cases[: args.limit]
    for index, case in enumerate(cases, start=1):
        scene_id = case["scene_id"]
        scene_dir = dataset / "scenes" / scene_id
        scene_record = _read_scene(scene_dir, manifest["manifest_hash"])
        if scene_record["case"] != case:
            raise ValueError(f"Scene specification disagrees with manifest: {scene_id}")
        result_dir = output / "results" / scene_id
        if _is_complete(result_dir, scene_record, manifest["manifest_hash"], model_hash):
            if args.resume:
                print(f"[{index}/{len(cases)}] {scene_id}: already complete")
                continue
            raise FileExistsError(
                f"Result already exists: {result_dir}. Use --resume or a new output directory."
            )
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError(f"Incomplete result directory requires review: {result_dir}")
        rgb_path = dataset / scene_record["rgb_path"]
        with Image.open(rgb_path) as opened:
            image = opened.convert("RGB").copy()
        prompts = [case["target_label"]]
        started = time.perf_counter()
        detections = detector.detect(image, prompts)
        inference_seconds = time.perf_counter() - started
        validate_detections([asdict(item) for item in detections])
        evaluation = evaluate_instances(
            detections,
            scene_record["objects"],
            prompts,
            confidence_threshold=REPORTING_THRESHOLD,
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = result_dir / "annotated.png"
        draw_instance_annotations(
            image,
            detections,
            scene_record["objects"],
            confidence_threshold=REPORTING_THRESHOLD,
        ).save(annotated_path)
        result = {
            "schema_version": SCHEMA_VERSION,
            "manifest_hash": manifest["manifest_hash"],
            "model_hash": model_hash,
            "scene_id": scene_id,
            "scene_sha256": file_hash(scene_dir / "scene.json"),
            "case": case,
            "source": {
                "rgb_path": scene_record["rgb_path"],
                "rgb_sha256": scene_record["rgb_sha256"],
                "annotated_path": str(annotated_path.relative_to(output)),
            },
            "detector": detector_metadata,
            "prompts": prompts,
            "ground_truth": scene_record["objects"],
            "detections": [asdict(item) for item in detections],
            "evaluation_at_0_35": evaluation,
            "inference_time_seconds": inference_seconds,
            "scene_metadata": scene_record["metadata"],
        }
        write_json(result_dir / "result.json", result)
        print(
            f"[{index}/{len(cases)}] {scene_id}: "
            f"TP={evaluation['true_positives']} FP={evaluation['false_positives']} "
            f"FN={evaluation['false_negatives']} {inference_seconds:.2f}s"
        )
    _write_summary(output)
    print(f"Completed {len(cases)} scene(s) under {output}")


def _prepare_run_manifest(
    output: Path,
    manifest: dict[str, Any],
    detector: dict[str, Any],
    model_hash: str,
) -> None:
    path = output / "run.json"
    record = {
        "schema_version": SCHEMA_VERSION,
        "input_manifest_hash": manifest["manifest_hash"],
        "case_count": manifest["case_count"],
        "detector": detector,
        "model_hash": model_hash,
        "reporting_confidence_threshold": REPORTING_THRESHOLD,
        "reporting_iou_threshold": 0.5,
    }
    if path.exists() and read_json(path) != record:
        raise FileExistsError(f"Output directory belongs to a different run: {output}")
    write_json(path, record)


def _read_scene(scene_dir: Path, manifest_hash: str) -> dict[str, Any]:
    path = scene_dir / "scene.json"
    if not path.is_file():
        raise FileNotFoundError(f"Rendered scene is missing: {path}")
    record = read_json(path)
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported scene schema: {path}")
    if record.get("manifest_hash") != manifest_hash:
        raise ValueError(f"Scene does not belong to this manifest: {path}")
    rgb = safe_artifact_path(scene_dir.parents[1], record["rgb_path"])
    if file_hash(rgb) != record.get("rgb_sha256"):
        raise ValueError(f"RGB checksum mismatch: {rgb}")
    validate_scene_instances(record["objects"])
    record["_scene_sha256"] = file_hash(path)
    return record


def _is_complete(
    result_dir: Path,
    scene: dict[str, Any],
    manifest_hash: str,
    model_hash: str,
) -> bool:
    path = result_dir / "result.json"
    if not path.is_file() or not (result_dir / "annotated.png").is_file():
        return False
    try:
        result = read_json(path)
    except ValueError:
        return False
    return bool(
        result.get("schema_version") == SCHEMA_VERSION
        and result.get("manifest_hash") == manifest_hash
        and result.get("model_hash") == model_hash
        and result.get("source", {}).get("rgb_sha256") == scene.get("rgb_sha256")
        and result.get("scene_sha256") == scene.get("_scene_sha256")
    )


def _write_summary(output: Path) -> None:
    rows = []
    for path in sorted((output / "results").glob("*/result.json")):
        result = read_json(path)
        case = result["case"]
        evaluation = result["evaluation_at_0_35"]
        rows.append(
            {
                "scene_id": result["scene_id"],
                "target_label": case["target_label"],
                "asset_id": case["target_asset_id"],
                "target_present": case["target_present"],
                "factor": case["factor"],
                "level": case["level"],
                "true_positives": evaluation["true_positives"],
                "false_positives": evaluation["false_positives"],
                "false_negatives": evaluation["false_negatives"],
                "inference_time_seconds": result["inference_time_seconds"],
                "result_path": str(path.relative_to(output)),
            }
        )
    fields = list(rows[0]) if rows else ["scene_id"]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
