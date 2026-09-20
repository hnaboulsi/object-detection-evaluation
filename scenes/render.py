"""Headless rendering and segmentation-derived instance ground truth."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment import ExperimentCase

from .assets import ASSETS, get_asset, validate_assets
from .camera import CameraIntrinsics, CameraPose
from .house import (
    CAMERA_NAME,
    OBJECT_GEOM_PREFIX,
    Occluder,
    SceneObject,
    build_house_xml,
    camera_pose_for,
    clutter_objects,
    default_context_objects,
    make_occluder,
    target_position,
)

OCCLUDER_WIDTH_M = {
    "couch": 2.1,
    "chair": 0.65,
    "bed": 1.6,
    "dining table": 1.5,
}


@dataclass(frozen=True)
class RenderedScene:
    rgb: np.ndarray
    ground_truth: list[dict[str, Any]]
    objects: list[SceneObject]
    actual_target_visible_fraction: float | None
    xml_sha256: str
    metadata: dict[str, Any]


def render_case(
    case: ExperimentCase,
    base_intrinsics: CameraIntrinsics,
) -> RenderedScene:
    validate_assets()
    intrinsics = CameraIntrinsics(
        width=base_intrinsics.width,
        height=base_intrinsics.height,
        fx=case.focal_px if case.focal_px is not None else base_intrinsics.fx,
        fy=case.focal_px if case.focal_px is not None else base_intrinsics.fy,
        cx=base_intrinsics.cx,
        cy=base_intrinsics.cy,
    )
    camera = camera_pose_for(
        case.target_label,
        distance_m=case.camera_distance_m,
        azimuth_deg=case.camera_azimuth_deg,
    )
    objects = default_context_objects(case.target_label)
    objects.extend(clutter_objects(case.target_label, case.clutter))
    if case.target_present:
        objects.append(
            SceneObject(
                instance_id="target",
                label=case.target_label,
                asset_id=case.target_asset_id,
                position=target_position(case.target_label),
                yaw_deg=-90.0,
                role="target",
            )
        )

    occluder = None
    reference_target_pixels: int | None = None
    if case.target_present and case.target_visible_fraction < 1.0:
        reference_xml = build_house_xml(
            model_name=f"{case.scene_id}_reference",
            camera=camera,
            intrinsics=intrinsics,
            objects=objects,
            lighting=case.lighting,
        )
        _, reference_segmentation, reference_model, _ = _render_xml(
            reference_xml, intrinsics, rgb=False
        )
        reference_gt = _instance_ground_truth(reference_model, reference_segmentation, objects)
        target_gt = _target_record(reference_gt)
        reference_target_pixels = int(target_gt["visible_pixels"])
        if reference_target_pixels == 0:
            raise RuntimeError(f"Target is not visible in unoccluded scene {case.scene_id}")
        occluder = _find_occluder(
            case,
            camera,
            intrinsics,
            objects,
            reference_target_pixels,
        )

    xml = build_house_xml(
        model_name=case.scene_id,
        camera=camera,
        intrinsics=intrinsics,
        objects=objects,
        lighting=case.lighting,
        occluder=occluder,
    )
    rgb, segmentation, model, mujoco_version = _render_xml(xml, intrinsics, rgb=True)
    ground_truth = _instance_ground_truth(model, segmentation, objects)
    target_gt = _target_record(ground_truth) if case.target_present else None
    actual_fraction = None
    if target_gt is not None:
        if reference_target_pixels is None:
            actual_fraction = 1.0
        else:
            actual_fraction = float(target_gt["visible_pixels"]) / reference_target_pixels
    metadata = {
        "mujoco_version": mujoco_version,
        "camera_intrinsics": intrinsics.to_dict(),
        "camera_pose": camera.to_dict(),
        "segmentation_ground_truth": True,
        "target_visible_fraction_requested": case.target_visible_fraction,
        "target_visible_fraction_actual": actual_fraction,
        "house_shell_uses_primitives": True,
        "furniture_uses_separate_obj_meshes": True,
    }
    return RenderedScene(
        rgb=rgb,
        ground_truth=ground_truth,
        objects=objects,
        actual_target_visible_fraction=actual_fraction,
        xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        metadata=metadata,
    )


def _find_occluder(
    case: ExperimentCase,
    camera: CameraPose,
    intrinsics: CameraIntrinsics,
    objects: list[SceneObject],
    reference_pixels: int,
) -> Occluder:
    width = OCCLUDER_WIDTH_M[case.target_label]
    target_fraction = case.target_visible_fraction
    lower = 0.0
    upper = width * 1.5
    best: tuple[float, Occluder] | None = None
    for _ in range(12):
        offset = (lower + upper) / 2.0
        candidate = make_occluder(
            camera,
            target_position(case.target_label),
            width_m=width,
            lateral_offset_m=offset,
        )
        xml = build_house_xml(
            model_name=f"{case.scene_id}_occlusion_search",
            camera=camera,
            intrinsics=intrinsics,
            objects=objects,
            lighting=case.lighting,
            occluder=candidate,
        )
        _, segmentation, model, _ = _render_xml(xml, intrinsics, rgb=False)
        target_gt = _target_record(_instance_ground_truth(model, segmentation, objects))
        fraction = float(target_gt["visible_pixels"]) / reference_pixels
        error = abs(fraction - target_fraction)
        if best is None or error < best[0]:
            best = (error, candidate)
        if fraction < target_fraction:
            lower = offset
        else:
            upper = offset
    if best is None or best[0] > 0.10:
        error = best[0] if best else float("inf")
        raise RuntimeError(
            f"Could not achieve requested occlusion in {case.scene_id}; fraction error={error:.3f}"
        )
    return best[1]


def _render_xml(
    xml: str,
    intrinsics: CameraIntrinsics,
    *,
    rgb: bool,
) -> tuple[np.ndarray, np.ndarray, Any, str]:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("MuJoCo is required; install requirements.txt") from error

    assets = {asset.filename: asset.path.read_bytes() for asset in ASSETS.values()}
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    image = np.empty((0, 0, 3), dtype=np.uint8)
    with mujoco.Renderer(model, height=intrinsics.height, width=intrinsics.width) as renderer:
        if rgb:
            renderer.update_scene(data, camera=CAMERA_NAME)
            image = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=CAMERA_NAME)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    return image, segmentation, model, mujoco.__version__


def _instance_ground_truth(
    model: Any,
    segmentation: np.ndarray,
    objects: list[SceneObject],
) -> list[dict[str, Any]]:
    import mujoco

    object_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    records: list[dict[str, Any]] = []
    for item in objects:
        geom_ids = []
        prefix = f"{OBJECT_GEOM_PREFIX}{item.instance_id}"
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name == prefix:
                geom_ids.append(geom_id)
        mask = np.isin(segmentation[:, :, 0], geom_ids) & (segmentation[:, :, 1] == object_type)
        rows, columns = np.nonzero(mask)
        bbox = None
        if len(columns):
            bbox = [
                float(columns.min()),
                float(rows.min()),
                float(columns.max() + 1),
                float(rows.max() + 1),
            ]
        asset = get_asset(item.asset_id)
        records.append(
            {
                "instance_id": item.instance_id,
                "label": item.label,
                "bbox": bbox,
                "visible_pixels": int(mask.sum()),
                "visible": bool(len(columns)),
                "position": list(item.position),
                "yaw_deg": item.yaw_deg,
                "asset_id": item.asset_id,
                "asset_path": str(asset.path.relative_to(asset.path.parents[1])),
                "role": item.role,
            }
        )
    return records


def _target_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        if record["role"] == "target":
            return record
    raise RuntimeError("Rendered scene does not contain a target record")
