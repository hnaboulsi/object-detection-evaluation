"""Three intentionally simple room templates and placement ranges."""

from __future__ import annotations

from dataclasses import dataclass

from .objects import ObjectSpec, cabinet, chair


@dataclass(frozen=True)
class RoomSpec:
    name: str
    half_size: tuple[float, float]
    target_x_range: tuple[float, float]
    target_y_range: tuple[float, float]
    camera_distance_range: tuple[float, float]
    camera_height_range: tuple[float, float]
    target_types: tuple[str, ...]


ROOMS = {
    "living_room": RoomSpec(
        "living_room",
        (4.0, 3.2),
        (0.5, 1.4),
        (-0.8, 0.8),
        (1.0, 4.0),
        (0.28, 0.75),
        ("couch", "box", "cabinet"),
    ),
    "bedroom": RoomSpec(
        "bedroom",
        (3.8, 3.0),
        (0.4, 1.2),
        (-0.7, 0.7),
        (1.0, 3.7),
        (0.25, 0.72),
        ("bed", "box", "cabinet"),
    ),
    "dining_room": RoomSpec(
        "dining_room",
        (4.0, 3.2),
        (0.5, 1.3),
        (-0.7, 0.7),
        (1.0, 4.0),
        (0.25, 0.75),
        ("table", "chair", "box"),
    ),
}


def get_room(name: str) -> RoomSpec:
    try:
        return ROOMS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown room {name!r}. Choose: {', '.join(ROOMS)}"
        ) from error


def room_background(name: str) -> list[ObjectSpec]:
    """Return sparse fixed furniture that is not the labeled target."""
    if name == "living_room":
        return [
            cabinet("room_cabinet", (2.9, 2.35, 0.0), 90.0),
            chair("room_chair", (1.8, -2.25, 0.0), -25.0),
        ]
    if name == "bedroom":
        return [
            cabinet("room_cabinet", (2.6, 2.15, 0.0), 90.0),
            chair("room_chair", (1.7, -2.0, 0.0), 15.0),
        ]
    if name == "dining_room":
        return [
            cabinet("room_cabinet", (2.9, 2.35, 0.0), 90.0),
            chair("room_chair_left", (1.8, 1.8, 0.0), 90.0),
            chair("room_chair_right", (1.8, -1.8, 0.0), -90.0),
        ]
    get_room(name)
    raise AssertionError("unreachable")
