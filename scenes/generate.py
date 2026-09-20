"""Deterministic scenario generation and direct MuJoCo rendering."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .objects import ObjectSpec, add_object_xml, make_primitive
from .rooms import get_room, room_background

CAMERA_NAME = "robot_camera"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
TARGET_PREFIX = "target__"


@dataclass(frozen=True)
class CameraRig:
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    fov_y_deg: float = 70.0
    mount_description: str = "provisional Go2-like front camera"


@dataclass(frozen=True)
class ObjectPlacement:
    kind: str
    position: tuple[float, float, float]
    yaw_deg: float = 0.0
    color: tuple[float, float, float, float] | None = None
    dimensions: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    room: str
    target_type: str
    target_present: bool
    target_position: tuple[float, float, float]
    target_yaw_deg: float
    camera: CameraRig
    lighting: str
    occlusion: str
    occluders: tuple[ObjectPlacement, ...]
    clutter: tuple[ObjectPlacement, ...]
    distractors: tuple[ObjectPlacement, ...]
    prompts: tuple[str, ...]
    custom_target: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RenderedScenario:
    rgb: np.ndarray
    depth: np.ndarray | None
    segmentation: np.ndarray
    ground_truth: dict[str, Any]
    metadata: dict[str, Any]
    xml: str


def derived_seed(base_seed: int, scenario_index: int) -> int:
    """Derive one stable uint32 seed without touching global random state."""
    return int(np.random.SeedSequence([base_seed, scenario_index]).generate_state(1)[0])


def target_height(kind: str) -> float:
    return {
        "box": 0.25,
        "couch": 0.43,
        "bed": 0.43,
        "table": 0.76,
        "chair": 0.55,
        "cabinet": 0.63,
        "custom": 0.35,
    }.get(kind, 0.35)


def place_camera_looking_at(
    target: tuple[float, float, float],
    *,
    distance: float,
    azimuth_deg: float = 0.0,
    height: float = 0.45,
    pitch_offset_deg: float = 0.0,
    fov_y_deg: float = 70.0,
) -> CameraRig:
    """Place a fixed camera around a target and point it toward the target."""
    azimuth = math.radians(azimuth_deg)
    position = (
        target[0] - distance * math.cos(azimuth),
        target[1] - distance * math.sin(azimuth),
        height,
    )
    horizontal = math.hypot(target[0] - position[0], target[1] - position[1])
    yaw = math.degrees(math.atan2(target[1] - position[1], target[0] - position[0]))
    pitch = math.degrees(math.atan2(target[2] - position[2], horizontal))
    return CameraRig(position, yaw, pitch + pitch_offset_deg, fov_y_deg)


def _target_prompt(kind: str) -> str:
    return "red box" if kind == "box" else kind


def _occluders(
    level: str,
    camera: CameraRig,
    target_position: tuple[float, float, float],
) -> tuple[ObjectPlacement, ...]:
    if level == "none":
        return ()
    fraction = 0.62
    x = camera.position[0] + fraction * (target_position[0] - camera.position[0])
    y = camera.position[1] + fraction * (target_position[1] - camera.position[1])
    if level == "partial":
        return (
            ObjectPlacement(
                "box",
                (x, y + 0.18, 0.0),
                dimensions=(0.22, 0.42, 0.7),
                color=(0.25, 0.25, 0.27, 1.0),
            ),
        )
    if level == "heavy":
        return (
            ObjectPlacement(
                "box",
                (x, y, 0.0),
                dimensions=(0.35, 0.85, 1.05),
                color=(0.25, 0.25, 0.27, 1.0),
            ),
        )
    raise ValueError(f"Unknown occlusion level: {level}")


def random_scenario(room_name: str, base_seed: int, scenario_index: int) -> Scenario:
    """Generate one bounded scenario whose complete state is serializable."""
    room = get_room(room_name)
    seed = derived_seed(base_seed, scenario_index)
    rng = np.random.default_rng(seed)
    target_kind = str(rng.choice(room.target_types))
    target_position = (
        float(rng.uniform(*room.target_x_range)),
        float(rng.uniform(*room.target_y_range)),
        0.0,
    )
    aim = (target_position[0], target_position[1], target_height(target_kind))
    for _ in range(100):
        camera = place_camera_looking_at(
            aim,
            distance=float(rng.uniform(*room.camera_distance_range)),
            azimuth_deg=float(rng.uniform(-45.0, 45.0)),
            height=float(rng.uniform(*room.camera_height_range)),
            pitch_offset_deg=float(rng.uniform(-15.0, 15.0)),
            fov_y_deg=float(rng.choice((55.0, 70.0, 85.0))),
        )
        x, y, _ = camera.position
        if abs(x) <= room.half_size[0] - 0.1 and abs(y) <= room.half_size[1] - 0.1:
            break
    else:
        raise RuntimeError(f"Could not place a camera inside {room_name}")
    lighting = str(rng.choice(("dim", "normal", "bright", "side")))
    occlusion = str(rng.choice(("none", "none", "partial", "heavy")))

    clutter = tuple(
        ObjectPlacement(
            "box",
            (
                float(rng.uniform(-0.2, 2.8)),
                float(rng.uniform(-2.2, 2.2)),
                0.0,
            ),
            float(rng.uniform(-180.0, 180.0)),
            tuple(
                float(value)
                for value in rng.uniform((0.2, 0.2, 0.2, 1.0), (0.75, 0.75, 0.75, 1.0))
            ),
            tuple(
                float(value)
                for value in rng.uniform((0.12, 0.12, 0.15), (0.45, 0.45, 0.65))
            ),
        )
        for _ in range(int(rng.integers(0, 6)))
    )
    distractors = tuple(
        ObjectPlacement(
            target_kind,
            (
                float(rng.uniform(0.0, 2.7)),
                float(rng.choice((-1.0, 1.0)) * rng.uniform(1.1, 2.2)),
                0.0,
            ),
            float(rng.uniform(-180.0, 180.0)),
            (0.72, 0.12, 0.10, 1.0) if target_kind == "box" else None,
        )
        for _ in range(int(rng.integers(0, 3)))
    )
    return Scenario(
        scenario_id=f"scenario_{scenario_index:06d}",
        seed=seed,
        room=room_name,
        target_type=target_kind,
        target_present=bool(rng.random() >= 0.1),
        target_position=target_position,
        target_yaw_deg=float(rng.uniform(-180.0, 180.0)),
        camera=camera,
        lighting=lighting,
        occlusion=occlusion,
        occluders=_occluders(occlusion, camera, target_position),
        clutter=clutter,
        distractors=distractors,
        prompts=(_target_prompt(target_kind),),
    )


def _manual(
    scenario_id: str,
    index: int,
    *,
    room: str,
    target_type: str,
    distance: float = 2.2,
    azimuth: float = 0.0,
    camera_height: float = 0.45,
    pitch_offset: float = 0.0,
    yaw_extra: float = 0.0,
    lighting: str = "normal",
    occlusion: str = "none",
    target_present: bool = True,
    target_position: tuple[float, float, float] = (1.0, 0.0, 0.0),
    clutter: tuple[ObjectPlacement, ...] = (),
    distractors: tuple[ObjectPlacement, ...] = (),
    extra_occluders: tuple[ObjectPlacement, ...] = (),
) -> Scenario:
    aim = (target_position[0], target_position[1], target_height(target_type))
    camera = place_camera_looking_at(
        aim,
        distance=distance,
        azimuth_deg=azimuth,
        height=camera_height,
        pitch_offset_deg=pitch_offset,
    )
    camera = replace(camera, yaw_deg=camera.yaw_deg + yaw_extra)
    return Scenario(
        scenario_id,
        derived_seed(1000, index),
        room,
        target_type,
        target_present,
        target_position,
        0.0,
        camera,
        lighting,
        occlusion,
        _occluders(occlusion, camera, target_position) + extra_occluders,
        clutter,
        distractors,
        (_target_prompt(target_type),),
    )


def handwritten_scenarios() -> list[Scenario]:
    """Return 18 readable examples, each stressing one recognizable condition."""
    clutter = tuple(
        ObjectPlacement(
            "box",
            (0.2 + 0.38 * i, (-1) ** i * 0.72, 0.0),
            yaw_deg=13.0 * i,
            dimensions=(0.28, 0.32, 0.36),
        )
        for i in range(7)
    )
    distractors = (
        ObjectPlacement("box", (0.8, 1.0, 0.0), color=(0.80, 0.10, 0.08, 1.0)),
        ObjectPlacement("box", (1.4, -1.0, 0.0), color=(0.72, 0.08, 0.08, 1.0)),
    )
    table_between = (ObjectPlacement("table", (0.15, 0.0, 0.0)),)
    return [
        _manual("direct_couch", 0, room="living_room", target_type="couch"),
        _manual("couch_side", 1, room="living_room", target_type="couch", azimuth=90.0),
        _manual(
            "couch_rear", 2, room="living_room", target_type="couch", azimuth=180.0
        ),
        _manual("bed_close", 3, room="bedroom", target_type="bed", distance=1.0),
        _manual("bed_far", 4, room="bedroom", target_type="bed", distance=3.6),
        _manual(
            "low_floor_box",
            5,
            room="living_room",
            target_type="box",
            camera_height=0.65,
        ),
        _manual(
            "box_under_table",
            6,
            room="dining_room",
            target_type="box",
            camera_height=0.20,
            extra_occluders=table_between,
        ),
        _manual(
            "through_table_legs",
            7,
            room="dining_room",
            target_type="box",
            camera_height=0.18,
            distance=2.8,
            extra_occluders=table_between,
        ),
        _manual(
            "partial_occlusion",
            8,
            room="living_room",
            target_type="box",
            occlusion="partial",
        ),
        _manual(
            "heavy_occlusion",
            9,
            room="living_room",
            target_type="box",
            occlusion="heavy",
        ),
        _manual(
            "heavy_clutter", 10, room="living_room", target_type="box", clutter=clutter
        ),
        _manual(
            "similar_distractors",
            11,
            room="living_room",
            target_type="box",
            distractors=distractors,
        ),
        _manual(
            "target_at_edge", 12, room="living_room", target_type="box", yaw_extra=24.0
        ),
        _manual(
            "target_outside_frame",
            13,
            room="living_room",
            target_type="box",
            yaw_extra=70.0,
        ),
        _manual(
            "target_absent",
            14,
            room="living_room",
            target_type="box",
            target_present=False,
        ),
        _manual("dim_light", 15, room="bedroom", target_type="bed", lighting="dim"),
        _manual(
            "bright_side_light",
            16,
            room="dining_room",
            target_type="table",
            lighting="side",
        ),
        _manual(
            "steep_camera_pitch",
            17,
            room="living_room",
            target_type="box",
            camera_height=1.5,
            pitch_offset=-22.0,
        ),
    ]


def _camera_xyaxes(camera: CameraRig) -> tuple[float, ...]:
    yaw = math.radians(camera.yaw_deg)
    pitch = math.radians(camera.pitch_deg)
    forward = np.array(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=float,
    )
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return tuple(float(value) for value in np.concatenate((right, up)))


def _numbers(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _lighting(root: ET.Element, worldbody: ET.Element, name: str) -> None:
    visual = ET.SubElement(root, "visual")
    settings = {
        "dim": ((0.025, 0.025, 0.025), (0.18, 0.18, 0.18)),
        "normal": ((0.12, 0.12, 0.12), (0.7, 0.7, 0.7)),
        "bright": ((0.28, 0.28, 0.28), (1.0, 1.0, 1.0)),
        "side": ((0.06, 0.06, 0.06), (0.28, 0.28, 0.28)),
    }
    if name not in settings:
        raise ValueError(f"Unknown lighting: {name}")
    ambient, diffuse = settings[name]
    ET.SubElement(
        visual,
        "headlight",
        ambient=_numbers(ambient),
        diffuse=_numbers(diffuse),
        specular="0.1 0.1 0.1",
    )
    if name == "side":
        ET.SubElement(
            worldbody,
            "light",
            name="side_light",
            pos="0 -2 2.4",
            dir="0.3 0.8 -0.5",
            directional="true",
            diffuse="0.95 0.95 0.85",
        )


def _placement_object(placement: ObjectPlacement, name: str) -> ObjectSpec:
    return make_primitive(
        placement.kind,
        name,
        placement.position,
        placement.yaw_deg,
        placement.color,
        placement.dimensions,
    )


def build_scene_xml(scenario: Scenario) -> str:
    """Build an in-memory MJCF scene for one scenario."""
    room = get_room(scenario.room)
    root = ET.Element("mujoco", model=f"detection_{scenario.scenario_id}")
    ET.SubElement(root, "compiler", angle="degree")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    _lighting(root, worldbody, scenario.lighting)

    half_x, half_y = room.half_size
    ET.SubElement(
        worldbody,
        "geom",
        name="room_floor",
        type="plane",
        size=f"{half_x} {half_y} 0.1",
        rgba="0.62 0.60 0.56 1",
    )
    wall_color = "0.86 0.84 0.79 1"
    ET.SubElement(
        worldbody,
        "geom",
        name="wall_back",
        type="box",
        size=f"0.05 {half_y} 1.4",
        pos=f"{half_x} 0 1.4",
        rgba=wall_color,
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="wall_front",
        type="box",
        size=f"0.05 {half_y} 1.4",
        pos=f"{-half_x} 0 1.4",
        rgba=wall_color,
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="wall_left",
        type="box",
        size=f"{half_x} 0.05 1.4",
        pos=f"0 {half_y} 1.4",
        rgba=wall_color,
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="wall_right",
        type="box",
        size=f"{half_x} 0.05 1.4",
        pos=f"0 {-half_y} 1.4",
        rgba=wall_color,
    )
    ET.SubElement(
        worldbody,
        "camera",
        name=CAMERA_NAME,
        pos=_numbers(scenario.camera.position),
        xyaxes=_numbers(_camera_xyaxes(scenario.camera)),
        fovy=f"{scenario.camera.fov_y_deg:.8g}",
    )

    for item in room_background(scenario.room):
        add_object_xml(worldbody, asset, item)

    if scenario.target_present:
        if scenario.custom_target:
            values = dict(scenario.custom_target)
            values["name"] = "target"
            values["label"] = scenario.target_type
            values["position"] = scenario.target_position
            values["rotation_deg"] = (0.0, 0.0, scenario.target_yaw_deg)
            target = ObjectSpec.from_obj_dict(values)
        else:
            target = make_primitive(
                scenario.target_type,
                "target",
                scenario.target_position,
                scenario.target_yaw_deg,
            )
        add_object_xml(worldbody, asset, target)

    groups = (
        ("occluder", scenario.occluders),
        ("clutter", scenario.clutter),
        ("distractor", scenario.distractors),
    )
    for group_name, placements in groups:
        for index, placement in enumerate(placements):
            add_object_xml(
                worldbody,
                asset,
                _placement_object(placement, f"{group_name}_{index}"),
            )
    return ET.tostring(root, encoding="unicode")


def _ground_truth(
    model: Any, segmentation: np.ndarray, target_present: bool
) -> dict[str, Any]:
    import mujoco

    if not target_present:
        return {
            "label": None,
            "target_present": False,
            "target_visible": False,
            "bbox": None,
            "visible_pixels": 0,
        }
    geom_ids = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(TARGET_PREFIX):
            geom_ids.append(geom_id)
    object_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    mask = np.isin(segmentation[:, :, 0], geom_ids) & (
        segmentation[:, :, 1] == object_type
    )
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return {
            "label": None,
            "target_present": True,
            "target_visible": False,
            "bbox": None,
            "visible_pixels": 0,
        }
    return {
        "label": None,
        "target_present": True,
        "target_visible": True,
        "bbox": [
            float(columns.min()),
            float(rows.min()),
            float(columns.max() + 1),
            float(rows.max() + 1),
        ],
        "visible_pixels": int(mask.sum()),
    }


def render_scenario(
    scenario: Scenario,
    *,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    include_depth: bool = False,
) -> RenderedScenario:
    """Compile and render RGB, segmentation, and optional depth directly."""
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("MuJoCo is required; install requirements.txt") from error

    xml = build_scene_xml(scenario)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render().copy()

        depth = None
        if include_depth:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=CAMERA_NAME)
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=CAMERA_NAME)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()

    ground_truth = _ground_truth(model, segmentation, scenario.target_present)
    ground_truth["label"] = scenario.target_type if scenario.target_present else None
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    target_point = np.asarray(scenario.target_position, dtype=float)
    target_point[2] += target_height(scenario.target_type)
    camera_position = data.cam_xpos[camera_id].astype(float)
    metadata = {
        "mujoco_version": mujoco.__version__,
        "image_width": width,
        "image_height": height,
        "camera_world_position": camera_position.tolist(),
        "camera_to_target_distance_m": float(
            np.linalg.norm(target_point - camera_position)
        ),
        "camera_mount_is_provisional": True,
        "depth_used_by_detector": False,
        "segmentation_ground_truth": True,
    }
    return RenderedScenario(rgb, depth, segmentation, ground_truth, metadata, xml)
