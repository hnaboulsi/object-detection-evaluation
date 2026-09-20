"""Render the deterministic 216-scene controlled experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from artifacts import SCHEMA_VERSION, file_hash, json_hash, read_json, write_json
from experiment import ExperimentCase, build_manifest, read_manifest, write_manifest
from scenes.camera import CameraIntrinsics
from scenes.render import render_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, default=Path("configs/camera.json"))
    parser.add_argument("--output", type=Path, default=Path("data/controlled_v1"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Render only the first N scenes for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = CameraIntrinsics.from_json(args.camera_config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        expected_camera = intrinsics.to_dict()
        if manifest["camera"] != expected_camera:
            raise SystemExit(
                "Existing manifest uses different camera intrinsics; choose a new output directory"
            )
        if manifest["manifest_hash"] != build_manifest(intrinsics)["manifest_hash"]:
            raise ValueError("Rendering recipe changed; choose a new output directory")
    else:
        manifest = write_manifest(manifest_path, intrinsics)

    cases = [ExperimentCase.from_dict(value) for value in manifest["cases"]]
    if args.limit is not None:
        cases = cases[: args.limit]
    for index, case in enumerate(cases, start=1):
        scene_dir = output / "scenes" / case.scene_id
        if _is_complete(scene_dir, manifest["manifest_hash"], case):
            if args.resume:
                print(f"[{index}/{len(cases)}] {case.scene_id}: already complete")
                continue
            raise FileExistsError(
                f"Scene already exists: {scene_dir}. Use --resume or a new output directory."
            )
        if scene_dir.exists() and any(scene_dir.iterdir()):
            raise FileExistsError(f"Incomplete scene directory requires manual review: {scene_dir}")
        rendered = render_case(case, intrinsics)
        scene_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = scene_dir / "rgb.png"
        Image.fromarray(rendered.rgb, mode="RGB").save(rgb_path)
        scene_record = {
            "schema_version": SCHEMA_VERSION,
            "manifest_hash": manifest["manifest_hash"],
            "case": case.to_dict(),
            "case_hash": case_hash(case),
            "rgb_path": str(rgb_path.relative_to(output)),
            "rgb_sha256": file_hash(rgb_path),
            "xml_sha256": rendered.xml_sha256,
            "objects": rendered.ground_truth,
            "metadata": rendered.metadata,
        }
        write_json(scene_dir / "scene.json", scene_record)
        actual = rendered.actual_target_visible_fraction
        fraction = "absent" if actual is None else f"{actual:.2f} visible"
        print(f"[{index}/{len(cases)}] {case.scene_id}: {fraction}")

    print(f"Rendered {len(cases)} scene(s) under {output}")


def case_hash(case: ExperimentCase) -> str:
    return json_hash(case.to_dict())


def _is_complete(scene_dir: Path, manifest_hash: str, case: ExperimentCase) -> bool:
    record_path = scene_dir / "scene.json"
    rgb_path = scene_dir / "rgb.png"
    if not record_path.is_file() or not rgb_path.is_file():
        return False
    try:
        record = read_json(record_path)
    except ValueError:
        return False
    return bool(
        record.get("manifest_hash") == manifest_hash
        and record.get("case_hash") == case_hash(case)
        and record.get("rgb_sha256") == file_hash(rgb_path)
    )


if __name__ == "__main__":
    main()
