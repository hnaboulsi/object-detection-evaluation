"""Small MuJoCo furniture primitives and optional one-OBJ objects."""

from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Color = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class GeomSpec:
    suffix: str
    geom_type: str
    size: tuple[float, ...]
    position: Vector3
    color: Color | None = None


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    label: str
    position: Vector3
    rotation_deg: Vector3 = (0.0, 0.0, 0.0)
    color: Color = (0.65, 0.65, 0.65, 1.0)
    geoms: tuple[GeomSpec, ...] = ()
    obj_path: str | None = None
    mesh_scale: Vector3 = (1.0, 1.0, 1.0)

    @classmethod
    def from_obj_dict(cls, values: dict[str, Any]) -> ObjectSpec:
        """Create one semantic object from one OBJ file and a plain dict."""
        required = {"name", "label", "obj_path", "position"}
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"Custom OBJ is missing: {', '.join(missing)}")
        path = Path(values["obj_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OBJ file not found: {path}")
        if path.suffix.lower() != ".obj":
            raise ValueError(
                "Custom assets must be .obj files; MTL and textures are not loaded"
            )
        warnings.warn(
            "Custom OBJ origin and scale are asset-specific; adjust position and scale manually if needed.",
            stacklevel=2,
        )
        return cls(
            name=_safe_name(str(values["name"])),
            label=str(values["label"]),
            position=_vector(values["position"], "position"),
            rotation_deg=_vector(values.get("rotation_deg", (0, 0, 0)), "rotation_deg"),
            color=_color(values.get("color", (0.7, 0.7, 0.7, 1.0))),
            obj_path=str(path),
            mesh_scale=_vector(values.get("scale", (1, 1, 1)), "scale"),
        )


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", value.strip())
    if not name:
        raise ValueError("Object name cannot be empty")
    return name


def _vector(values: Iterable[float], field: str) -> Vector3:
    result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{field} must have three values")
    return result  # type: ignore[return-value]


def _color(values: Iterable[float]) -> Color:
    result = tuple(float(value) for value in values)
    if len(result) != 4 or any(value < 0 or value > 1 for value in result):
        raise ValueError("color must contain four values in [0, 1]")
    return result  # type: ignore[return-value]


def _box(
    suffix: str, dimensions: Vector3, position: Vector3, color: Color | None = None
) -> GeomSpec:
    return GeomSpec(
        suffix,
        "box",
        tuple(float(value) / 2.0 for value in dimensions),
        position,
        color,
    )


def box(
    name: str,
    position: Vector3,
    yaw_deg: float = 0.0,
    dimensions: Vector3 = (0.5, 0.5, 0.5),
    color: Color = (0.85, 0.08, 0.08, 1.0),
    label: str = "box",
) -> ObjectSpec:
    return ObjectSpec(
        name=_safe_name(name),
        label=label,
        position=position,
        rotation_deg=(0.0, 0.0, yaw_deg),
        color=color,
        geoms=(_box("body", dimensions, (0.0, 0.0, dimensions[2] / 2.0)),),
    )


def couch(
    name: str,
    position: Vector3,
    yaw_deg: float = 0.0,
    color: Color = (0.18, 0.35, 0.72, 1.0),
) -> ObjectSpec:
    return ObjectSpec(
        _safe_name(name),
        "couch",
        position,
        (0.0, 0.0, yaw_deg),
        color,
        (
            _box("seat", (0.85, 1.8, 0.28), (0.0, 0.0, 0.25)),
            _box("back", (0.22, 1.8, 0.72), (0.34, 0.0, 0.55)),
            _box("arm_left", (0.85, 0.18, 0.48), (0.0, 0.9, 0.34)),
            _box("arm_right", (0.85, 0.18, 0.48), (0.0, -0.9, 0.34)),
        ),
    )


def bed(
    name: str,
    position: Vector3,
    yaw_deg: float = 0.0,
    color: Color = (0.72, 0.72, 0.78, 1.0),
) -> ObjectSpec:
    return ObjectSpec(
        _safe_name(name),
        "bed",
        position,
        (0.0, 0.0, yaw_deg),
        color,
        (
            _box("base", (2.0, 1.45, 0.24), (0.0, 0.0, 0.18), (0.35, 0.22, 0.14, 1.0)),
            _box("mattress", (1.9, 1.35, 0.24), (0.0, 0.0, 0.42)),
            _box(
                "headboard",
                (0.18, 1.5, 0.85),
                (0.91, 0.0, 0.55),
                (0.38, 0.22, 0.14, 1.0),
            ),
            _box(
                "pillow", (0.45, 0.9, 0.12), (0.58, 0.0, 0.59), (0.93, 0.93, 0.9, 1.0)
            ),
        ),
    )


def table(
    name: str,
    position: Vector3,
    yaw_deg: float = 0.0,
    color: Color = (0.45, 0.25, 0.12, 1.0),
) -> ObjectSpec:
    legs = tuple(
        _box(f"leg_{index}", (0.10, 0.10, 0.72), (x, y, 0.36))
        for index, (x, y) in enumerate(
            ((-0.65, -0.4), (-0.65, 0.4), (0.65, -0.4), (0.65, 0.4))
        )
    )
    return ObjectSpec(
        _safe_name(name),
        "table",
        position,
        (0.0, 0.0, yaw_deg),
        color,
        (_box("top", (1.55, 1.0, 0.12), (0.0, 0.0, 0.76)), *legs),
    )


def chair(
    name: str,
    position: Vector3,
    yaw_deg: float = 0.0,
    color: Color = (0.32, 0.18, 0.10, 1.0),
) -> ObjectSpec:
    legs = tuple(
        _box(f"leg_{index}", (0.07, 0.07, 0.42), (x, y, 0.21))
        for index, (x, y) in enumerate(
            ((-0.2, -0.2), (-0.2, 0.2), (0.2, -0.2), (0.2, 0.2))
        )
    )
    return ObjectSpec(
        _safe_name(name),
        "chair",
        position,
        (0.0, 0.0, yaw_deg),
        color,
        (
            _box("seat", (0.5, 0.5, 0.09), (0.0, 0.0, 0.45)),
            _box("back", (0.10, 0.5, 0.58), (0.2, 0.0, 0.73)),
            *legs,
        ),
    )


def cabinet(
    name: str,
    position: Vector3,
    yaw_deg: float = 0.0,
    color: Color = (0.42, 0.24, 0.13, 1.0),
) -> ObjectSpec:
    return ObjectSpec(
        _safe_name(name),
        "cabinet",
        position,
        (0.0, 0.0, yaw_deg),
        color,
        (_box("body", (0.55, 1.15, 1.25), (0.0, 0.0, 0.625)),),
    )


def make_primitive(
    kind: str,
    name: str,
    position: Vector3,
    yaw_deg: float = 0.0,
    color: Color | None = None,
    dimensions: Vector3 | None = None,
) -> ObjectSpec:
    builders = {
        "box": box,
        "couch": couch,
        "bed": bed,
        "table": table,
        "chair": chair,
        "cabinet": cabinet,
    }
    if kind not in builders:
        raise ValueError(f"Unknown object primitive: {kind}")
    if kind == "box":
        return box(
            name,
            position,
            yaw_deg,
            dimensions or (0.5, 0.5, 0.5),
            color or (0.85, 0.08, 0.08, 1.0),
        )
    builder = builders[kind]
    return (
        builder(name, position, yaw_deg, color)
        if color
        else builder(name, position, yaw_deg)
    )


def add_object_xml(worldbody: ET.Element, asset: ET.Element, spec: ObjectSpec) -> None:
    """Append one object with stable body/geom names to an MJCF tree."""
    body = ET.SubElement(
        worldbody,
        "body",
        name=spec.name,
        pos=_numbers(spec.position),
        euler=_numbers(spec.rotation_deg),
    )
    if spec.obj_path:
        mesh_name = f"{spec.name}__mesh_asset"
        ET.SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=spec.obj_path,
            scale=_numbers(spec.mesh_scale),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{spec.name}__mesh",
            type="mesh",
            mesh=mesh_name,
            rgba=_numbers(spec.color),
            contype="0",
            conaffinity="0",
        )
        return
    for geom in spec.geoms:
        attributes = {
            "name": f"{spec.name}__{geom.suffix}",
            "type": geom.geom_type,
            "size": _numbers(geom.size),
            "pos": _numbers(geom.position),
            "rgba": _numbers(geom.color or spec.color),
            "contype": "0",
            "conaffinity": "0",
        }
        ET.SubElement(body, "geom", **attributes)


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
