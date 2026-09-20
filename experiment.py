"""Deterministic manifest for the controlled house experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from artifacts import file_hash
from scenes.assets import TARGET_ASSETS
from scenes.camera import CameraIntrinsics
from scenes.house import room_for_label

MANIFEST_VERSION = 1
BASE_SEED = 20260919


@dataclass(frozen=True)
class ExperimentCase:
    scene_id: str
    seed: int
    target_label: str
    target_asset_id: str
    target_present: bool
    room: str
    factor: str
    level: str
    camera_azimuth_deg: float = 0.0
    camera_distance_m: float = 2.5
    target_visible_fraction: float = 1.0
    lighting: str = "normal"
    clutter: str = "none"
    focal_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentCase:
        return cls(**values)


def controlled_cases() -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    index = 0
    for label, asset_ids in TARGET_ASSETS.items():
        for asset_id in asset_ids:
            base = ExperimentCase(
                scene_id="",
                seed=BASE_SEED + index,
                target_label=label,
                target_asset_id=asset_id,
                target_present=True,
                room=room_for_label(label),
                factor="baseline",
                level="baseline",
            )
            conditions = [base]
            conditions.extend(
                replace(
                    base,
                    factor="viewpoint",
                    level=f"azimuth_{azimuth:+g}",
                    camera_azimuth_deg=float(azimuth),
                )
                for azimuth in (-90, -45, 45, 90)
            )
            conditions.extend(
                replace(
                    base,
                    factor="distance",
                    level=f"distance_{distance:g}m",
                    camera_distance_m=float(distance),
                )
                for distance in (1.5, 4.0)
            )
            conditions.extend(
                replace(
                    base,
                    factor="occlusion",
                    level=f"visible_{int(fraction * 100)}pct",
                    target_visible_fraction=float(fraction),
                )
                for fraction in (0.75, 0.50, 0.25)
            )
            conditions.extend(
                replace(base, factor="lighting", level=lighting, lighting=lighting)
                for lighting in ("dim", "bright", "side")
            )
            conditions.extend(
                replace(base, factor="clutter", level=clutter, clutter=clutter)
                for clutter in ("mixed", "similar_shape")
            )
            conditions.extend(
                replace(
                    base,
                    factor="focal_length",
                    level=f"focal_{int(focal)}px",
                    focal_px=float(focal),
                )
                for focal in (400, 700)
            )
            for condition in conditions:
                scene_id = f"scene_{index:03d}_{_slug(label)}_{_slug(condition.level)}"
                cases.append(replace(condition, scene_id=scene_id, seed=BASE_SEED + index))
                index += 1

    for label, asset_ids in TARGET_ASSETS.items():
        for negative_index, asset_id in enumerate(asset_ids):
            cases.append(
                ExperimentCase(
                    scene_id=f"scene_{index:03d}_{_slug(label)}_absent_{negative_index}",
                    seed=BASE_SEED + index,
                    target_label=label,
                    target_asset_id=asset_id,
                    target_present=False,
                    room=room_for_label(label),
                    factor="target_presence",
                    level=f"absent_{negative_index}",
                    lighting=("normal", "dim", "normal")[negative_index],
                    clutter=("none", "mixed", "similar_shape")[negative_index],
                )
            )
            index += 1

    if len(cases) != 216:
        raise AssertionError(f"Controlled manifest must contain 216 scenes, got {len(cases)}")
    return cases


def build_manifest(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "base_seed": BASE_SEED,
        "camera": intrinsics.to_dict(),
        "case_count": 216,
        "cases": [case.to_dict() for case in controlled_cases()],
        "render_recipe": render_recipe(),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def render_recipe() -> dict[str, str]:
    """Bind the dataset to source, assets, and the rendering dependency pins."""
    root = Path(__file__).resolve().parent
    paths = [
        root / "experiment.py",
        root / "requirements-render.txt",
        *sorted((root / "scenes").glob("*.py")),
        *sorted((root / "assets" / "furniture").glob("*.obj")),
    ]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def manifest_hash(manifest: dict[str, Any]) -> str:
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    encoded = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(path: Path, intrinsics: CameraIntrinsics) -> dict[str, Any]:
    manifest = build_manifest(intrinsics)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"Refusing to replace a different manifest: {path}")
    path.write_text(rendered, encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read manifest {path}: {error}") from error
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version in {path}")
    if manifest.get("case_count") != len(manifest.get("cases", [])):
        raise ValueError(f"Manifest case count is inconsistent: {path}")
    identifiers = [case["scene_id"] for case in manifest["cases"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Manifest contains duplicate scene IDs")
    expected_hash = manifest_hash(manifest)
    if manifest.get("manifest_hash") != expected_hash:
        raise ValueError(f"Manifest hash does not match its contents: {path}")
    return manifest


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("+", "plus").replace("-", "minus")
