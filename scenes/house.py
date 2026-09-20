"""Three-room MuJoCo house built from primitives and separate OBJ furniture."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from .assets import floor_offset, get_asset
from .camera import CameraIntrinsics, CameraPose

CAMERA_NAME = "evaluation_camera"
OBJECT_GEOM_PREFIX = "object__"
SHELL_COLOR = (0.86, 0.84, 0.79, 1.0)

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SceneObject:
    instance_id: str
    label: str
    asset_id: str
    position: Vector3
    yaw_deg: float
    role: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Occluder:
    position: Vector3
    yaw_deg: float
    width_m: float
    height_m: float = 2.2
    depth_m: float = 0.08


def room_for_label(label: str) -> str:
    return {
        "couch": "living_room",
        "chair": "dining_room",
        "bed": "bedroom",
        "dining table": "dining_room",
    }[label]


def target_position(label: str) -> Vector3:
    return {
        "couch": (-5.0, 0.0, 0.0),
        "chair": (5.0, 0.0, 0.0),
        "bed": (-5.0, 10.0, 0.0),
        "dining table": (5.0, 0.0, 0.0),
    }[label]


def target_aim_height(label: str) -> float:
    return {"couch": 0.55, "chair": 0.65, "bed": 0.45, "dining table": 0.65}[label]


def camera_pose_for(
    label: str,
    *,
    distance_m: float,
    azimuth_deg: float,
    height_m: float = 0.55,
) -> CameraPose:
    target = target_position(label)
    radians = math.radians(azimuth_deg)
    position = (
        target[0] - distance_m * math.cos(radians),
        target[1] - distance_m * math.sin(radians),
        height_m,
    )
    return CameraPose(
        position=position,
        look_at=(target[0], target[1], target_aim_height(label)),
    )


def default_context_objects(target_label: str) -> list[SceneObject]:
    objects = [
        SceneObject(
            "living_bookshelf", "bookshelf", "Bookshelf", (-1.2, 3.0, 0.0), -90.0, "context"
        ),
        SceneObject("living_plant", "houseplant", "Houseplant_1", (-3.0, 2.0, 0.0), 0.0, "context"),
        SceneObject(
            "dining_fridge", "refrigerator", "Kitchen_Fridge", (8.0, 3.0, 0.0), -90.0, "context"
        ),
        SceneObject(
            "bedroom_nightstand", "nightstand", "NightStand_1", (-3.3, 11.5, 0.0), 0.0, "context"
        ),
    ]
    furnishings = [
        ("couch", "Couch_Medium1", (-5.0, 0.0, 0.0)),
        ("bed", "Bed_King", (-5.0, 10.0, 0.0)),
        ("dining table", "Table_RoundLarge", (5.0, 0.0, 0.0)),
        ("chair", "Chair_2", (6.2, 2.0, 0.0)),
    ]
    for index, (label, asset_id, position) in enumerate(furnishings):
        if label != target_label and room_for_label(label) != room_for_label(target_label):
            objects.append(
                SceneObject(f"room_furniture_{index}", label, asset_id, position, -90.0, "context")
            )
    return objects


def clutter_objects(target_label: str, level: str) -> list[SceneObject]:
    if level == "none":
        return []
    x, y, _ = target_position(target_label)
    if level == "mixed":
        choices = [
            ("Houseplant_1", (x + 1.0, y + 1.15, 0.0)),
            ("NightStand_1", (x + 0.8, y - 1.25, 0.0)),
            ("Bookshelf", (x + 2.0, y + 1.7, 0.0)),
        ]
    elif level == "similar_shape":
        asset_id = {
            "couch": "Chair_1",
            "chair": "Table_RoundSmall",
            "bed": "Couch_Small1",
            "dining table": "Chair_3",
        }[target_label]
        choices = [
            (asset_id, (x + 1.0, y - 1.25, 0.0)),
            (asset_id, (x + 1.5, y + 1.35, 0.0)),
        ]
    else:
        raise ValueError(f"Unknown clutter level: {level}")
    return [
        SceneObject(f"clutter_{index}", get_asset(asset_id).label, asset_id, position, -90.0, level)
        for index, (asset_id, position) in enumerate(choices)
    ]


def make_occluder(
    camera: CameraPose,
    target: Vector3,
    *,
    width_m: float,
    lateral_offset_m: float,
) -> Occluder:
    camera_xy = np.asarray(camera.position[:2], dtype=float)
    target_xy = np.asarray(target[:2], dtype=float)
    direction = target_xy - camera_xy
    direction /= np.linalg.norm(direction)
    right = np.array([-direction[1], direction[0]])
    center = camera_xy + 0.32 * (target_xy - camera_xy) + lateral_offset_m * right
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    return Occluder((float(center[0]), float(center[1]), 0.0), yaw, width_m)


def build_house_xml(
    *,
    model_name: str,
    camera: CameraPose,
    intrinsics: CameraIntrinsics,
    objects: Iterable[SceneObject],
    lighting: str,
    occluder: Occluder | None = None,
) -> str:
    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(root, "compiler", angle="degree")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    asset_node = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    _add_lighting(root, worldbody, lighting)
    ET.SubElement(
        root.find("visual"),
        "global",
        offwidth=str(intrinsics.width),
        offheight=str(intrinsics.height),
    )
    _add_shell(worldbody)
    _add_camera(worldbody, camera, intrinsics)
    for item in objects:
        _add_furniture(worldbody, asset_node, item)
    if occluder is not None:
        _add_occluder(worldbody, occluder)
    return ET.tostring(root, encoding="unicode")


def _add_shell(worldbody: ET.Element) -> None:
    _box(worldbody, "house_floor", (10.0, 10.0, 0.05), (0.0, 5.0, -0.05), (0.58, 0.56, 0.52, 1.0))
    _box(worldbody, "house_ceiling", (10.0, 10.0, 0.05), (0.0, 5.0, 3.0), (0.78, 0.78, 0.75, 1.0))
    _box(worldbody, "outer_west", (0.06, 10.0, 1.5), (-10.0, 5.0, 1.5), SHELL_COLOR)
    _box(worldbody, "outer_east", (0.06, 10.0, 1.5), (10.0, 5.0, 1.5), SHELL_COLOR)
    _box(worldbody, "outer_south", (10.0, 0.06, 1.5), (0.0, -5.0, 1.5), SHELL_COLOR)
    _box(worldbody, "outer_north", (10.0, 0.06, 1.5), (0.0, 15.0, 1.5), SHELL_COLOR)
    _wall_with_door(worldbody, "living_dining", "vertical", -5.0, 5.0, 0.0, -1.0, 1.0)
    _wall_with_door(worldbody, "bedroom", "horizontal", -10.0, 10.0, 5.0, -1.0, 1.0)


def _wall_with_door(
    worldbody: ET.Element,
    name: str,
    orientation: str,
    start: float,
    end: float,
    fixed: float,
    door_start: float,
    door_end: float,
) -> None:
    segments = ((start, door_start), (door_end, end))
    for index, (low, high) in enumerate(segments):
        if high <= low:
            continue
        center = (low + high) / 2.0
        half_length = (high - low) / 2.0
        if orientation == "vertical":
            size = (0.06, half_length, 1.4)
            position = (fixed, center, 1.4)
        else:
            size = (half_length, 0.06, 1.4)
            position = (center, fixed, 1.4)
        _box(worldbody, f"{name}_{index}", size, position, SHELL_COLOR)
    if orientation == "vertical":
        lintel_size = (0.06, (door_end - door_start) / 2.0, 0.35)
        lintel_position = (fixed, (door_start + door_end) / 2.0, 2.45)
    else:
        lintel_size = ((door_end - door_start) / 2.0, 0.06, 0.35)
        lintel_position = ((door_start + door_end) / 2.0, fixed, 2.45)
    _box(worldbody, f"{name}_lintel", lintel_size, lintel_position, SHELL_COLOR)


def _box(
    parent: ET.Element,
    name: str,
    size: Vector3,
    position: Vector3,
    color: tuple[float, float, float, float],
) -> None:
    ET.SubElement(
        parent,
        "geom",
        name=name,
        type="box",
        size=_numbers(size),
        pos=_numbers(position),
        rgba=_numbers(color),
        contype="0",
        conaffinity="0",
    )


def _add_camera(worldbody: ET.Element, camera: CameraPose, intrinsics: CameraIntrinsics) -> None:
    ET.SubElement(
        worldbody,
        "camera",
        name=CAMERA_NAME,
        pos=_numbers(camera.position),
        xyaxes=_numbers(_camera_xyaxes(camera)),
        resolution=f"{intrinsics.width} {intrinsics.height}",
        sensorsize=f"{intrinsics.width} {intrinsics.height}",
        focalpixel=f"{intrinsics.fx:.8g} {intrinsics.fy:.8g}",
        principalpixel=(
            f"{(intrinsics.width - 1) / 2 - intrinsics.cx:.8g} "
            f"{intrinsics.cy - (intrinsics.height - 1) / 2:.8g}"
        ),
    )


def _camera_xyaxes(camera: CameraPose) -> tuple[float, ...]:
    position = np.asarray(camera.position, dtype=float)
    look_at = np.asarray(camera.look_at, dtype=float)
    forward = look_at - position
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError("Camera position and look-at point must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return tuple(float(value) for value in np.concatenate((right, up)))


def _add_furniture(worldbody: ET.Element, asset_node: ET.Element, item: SceneObject) -> None:
    asset = get_asset(item.asset_id)
    mesh_name = f"mesh__{item.instance_id}"
    ET.SubElement(
        asset_node,
        "mesh",
        name=mesh_name,
        file=asset.filename,
        scale=_numbers(asset.scale),
    )
    body = ET.SubElement(
        worldbody,
        "body",
        name=f"body__{item.instance_id}",
        pos=_numbers((item.position[0], item.position[1], item.position[2] + floor_offset(asset))),
        euler=f"0 0 {item.yaw_deg:.8g}",
    )
    ET.SubElement(
        body,
        "geom",
        name=f"{OBJECT_GEOM_PREFIX}{item.instance_id}",
        type="mesh",
        mesh=mesh_name,
        euler=_numbers(asset.mesh_rotation_deg),
        rgba=_numbers(asset.color),
        contype="0",
        conaffinity="0",
    )


def _add_occluder(worldbody: ET.Element, occluder: Occluder) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        name="experimental_occluder",
        pos=_numbers(occluder.position),
        euler=f"0 0 {occluder.yaw_deg:.8g}",
    )
    _box(
        body,
        "experimental_occluder_geom",
        (occluder.depth_m / 2.0, occluder.width_m / 2.0, occluder.height_m / 2.0),
        (0.0, 0.0, occluder.height_m / 2.0),
        (0.32, 0.33, 0.35, 1.0),
    )


def _add_lighting(root: ET.Element, worldbody: ET.Element, lighting: str) -> None:
    settings = {
        "normal": ((0.15, 0.15, 0.15), (0.75, 0.75, 0.75)),
        "dim": ((0.025, 0.025, 0.025), (0.16, 0.16, 0.16)),
        "bright": ((0.30, 0.30, 0.30), (1.0, 1.0, 1.0)),
        "side": ((0.05, 0.05, 0.05), (0.25, 0.25, 0.25)),
    }
    if lighting not in settings:
        raise ValueError(f"Unknown lighting level: {lighting}")
    ambient, diffuse = settings[lighting]
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        ambient=_numbers(ambient),
        diffuse=_numbers(diffuse),
        specular="0.1 0.1 0.1",
    )
    # Ceiling lighting keeps horizontal surfaces visible under normal conditions.
    for index, (x, y) in enumerate(((-5, 0), (5, 0), (-5, 10))):
        ET.SubElement(
            worldbody,
            "light",
            name=f"ceiling_light_{index}",
            pos=f"{x} {y} 2.7",
            dir="0 0 -1",
            diffuse=_numbers(tuple(value * 0.55 for value in diffuse)),
            ambient="0.03 0.03 0.03",
            castshadow="false",
        )
    if lighting == "side":
        ET.SubElement(
            worldbody,
            "light",
            name="side_light",
            pos="-4 -1 2.5",
            dir="0.8 0.2 -0.5",
            directional="true",
            diffuse="0.95 0.92 0.82",
        )


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
