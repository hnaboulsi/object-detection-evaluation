"""Verify complete dataset/run integrity without loading detector weights."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

from PIL import Image

from analysis import aggregate, load_results
from artifacts import file_hash, read_json, safe_artifact_path, validate_scene_instances, write_json
from evaluation import REPORTING_THRESHOLD
from experiment import build_manifest, read_manifest
from scenes.camera import CameraIntrinsics


def verify_dataset(dataset: Path) -> dict:
    manifest = read_manifest(dataset / "manifest.json")
    expected = build_manifest(CameraIntrinsics.from_dict(manifest["camera"]))
    if manifest != expected:
        raise ValueError("Dataset does not match this source's controlled experiment")
    scene_ids = {case["scene_id"] for case in manifest["cases"]}
    actual_ids = {path.parent.name for path in (dataset / "scenes").glob("*/scene.json")}
    if actual_ids != scene_ids:
        raise ValueError(
            f"Scene coverage mismatch: missing={scene_ids - actual_ids}, "
            f"extra={actual_ids - scene_ids}"
        )
    occlusion_errors = []
    positive_count = negative_count = 0
    for case in manifest["cases"]:
        record = read_json(dataset / "scenes" / case["scene_id"] / "scene.json")
        if record["case"] != case or record["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError(f"Scene metadata mismatch: {case['scene_id']}")
        validate_scene_instances(record["objects"])
        rgb_path = safe_artifact_path(dataset, record["rgb_path"])
        if file_hash(rgb_path) != record["rgb_sha256"]:
            raise ValueError(f"Image checksum mismatch: {rgb_path}")
        with Image.open(rgb_path) as image:
            if image.size != (manifest["camera"]["width"], manifest["camera"]["height"]):
                raise ValueError(f"Image resolution mismatch: {rgb_path}")
        prompted = [
            item
            for item in record["objects"]
            if item["label"] == case["target_label"] and item["visible"]
        ]
        if case["target_present"]:
            positive_count += 1
            if not any(item["instance_id"] == "target" for item in prompted):
                raise ValueError(f"Invisible positive target: {case['scene_id']}")
        else:
            negative_count += 1
            if prompted:
                raise ValueError(f"Negative control contains queried class: {case['scene_id']}")
        if case["factor"] == "occlusion":
            error = abs(
                record["metadata"]["target_visible_fraction_actual"]
                - case["target_visible_fraction"]
            )
            occlusion_errors.append(error)
            if error > 0.03:
                raise ValueError(f"Occlusion tolerance exceeded: {case['scene_id']}")
    return {
        "manifest_hash": manifest["manifest_hash"],
        "scene_count": len(scene_ids),
        "positive_scenes": positive_count,
        "negative_scenes": negative_count,
        "occlusion_max_absolute_error": max(occlusion_errors),
    }


def verify_run(dataset: Path, run: Path) -> dict:
    validation = verify_dataset(dataset)
    records = load_results(run, validation["manifest_hash"])
    manifest = read_manifest(dataset / "manifest.json")
    if {r["scene_id"] for r in records} != {c["scene_id"] for c in manifest["cases"]}:
        raise ValueError("Run must contain exactly the manifest's 216 scene IDs")
    for record in records:
        scene_path = dataset / "scenes" / record["scene_id"] / "scene.json"
        scene = read_json(scene_path)
        if (
            record["scene_sha256"] != file_hash(scene_path)
            or record["source"]["rgb_sha256"] != scene["rgb_sha256"]
            or record["ground_truth"] != scene["objects"]
            or record["case"] != scene["case"]
        ):
            raise ValueError(f"Result disagrees with source scene: {record['scene_id']}")
    validation["completed_results"] = len(records)
    validation["metrics"] = aggregate(records, REPORTING_THRESHOLD)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/controlled_v1"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validation = verify_run(args.dataset, args.run) if args.run else verify_dataset(args.dataset)
    validation["environment"] = {
        "system": platform.system(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
    }
    if args.output:
        write_json(args.output, validation)
    print(
        f"Verified {validation['scene_count']} scenes; "
        f"{validation.get('completed_results', 0)} inference records"
    )


if __name__ == "__main__":
    main()
