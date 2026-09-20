"""Curated CC0 furniture assets and their MuJoCo transforms."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

Color = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "furniture"


@dataclass(frozen=True)
class FurnitureAsset:
    asset_id: str
    label: str
    filename: str
    scale: Vector3
    color: Color
    floor_offset_m: float = 0.01
    mesh_rotation_deg: Vector3 = (90.0, 0.0, 0.0)

    @property
    def path(self) -> Path:
        return ASSET_ROOT / self.filename


def _asset(
    asset_id: str,
    label: str,
    scale: Vector3,
    color: Color,
    *,
    floor_offset_m: float = 0.01,
) -> FurnitureAsset:
    return FurnitureAsset(
        asset_id=asset_id,
        label=label,
        filename=f"{asset_id}.obj",
        scale=scale,
        color=color,
        floor_offset_m=floor_offset_m,
    )


ASSETS: dict[str, FurnitureAsset] = {
    "Couch_Large1": _asset("Couch_Large1", "couch", (0.42, 0.42, 0.42), (0.22, 0.38, 0.68, 1.0)),
    "Couch_Medium1": _asset("Couch_Medium1", "couch", (0.45, 0.45, 0.45), (0.55, 0.25, 0.18, 1.0)),
    "Couch_Small1": _asset("Couch_Small1", "couch", (0.48, 0.48, 0.48), (0.26, 0.52, 0.40, 1.0)),
    "Chair_1": _asset("Chair_1", "chair", (0.75, 0.50, 0.75), (0.55, 0.30, 0.14, 1.0)),
    "Chair_2": _asset("Chair_2", "chair", (0.75, 0.50, 0.75), (0.20, 0.42, 0.62, 1.0)),
    "Chair_3": _asset("Chair_3", "chair", (0.75, 0.50, 0.75), (0.58, 0.20, 0.22, 1.0)),
    "Bed_King": _asset("Bed_King", "bed", (0.60, 0.50, 0.55), (0.65, 0.65, 0.72, 1.0)),
    "Bed_Single": _asset("Bed_Single", "bed", (0.60, 0.50, 0.55), (0.76, 0.72, 0.62, 1.0)),
    "Bed_Bunk": _asset("Bed_Bunk", "bed", (0.60, 0.55, 0.55), (0.46, 0.28, 0.16, 1.0)),
    "Table_RoundLarge": _asset(
        "Table_RoundLarge", "dining table", (0.50, 0.70, 0.50), (0.44, 0.25, 0.13, 1.0)
    ),
    "Table_RoundSmall": _asset(
        "Table_RoundSmall", "dining table", (0.62, 0.70, 0.62), (0.52, 0.32, 0.17, 1.0)
    ),
    "Table_RoundSmall2": _asset(
        "Table_RoundSmall2", "dining table", (0.62, 0.70, 0.62), (0.30, 0.34, 0.38, 1.0)
    ),
    "Bookshelf": _asset("Bookshelf", "bookshelf", (0.45, 0.45, 0.45), (0.40, 0.22, 0.12, 1.0)),
    "Houseplant_1": _asset("Houseplant_1", "houseplant", (1.0, 1.0, 1.0), (0.18, 0.48, 0.20, 1.0)),
    "Kitchen_Fridge": _asset(
        "Kitchen_Fridge", "refrigerator", (0.65, 0.55, 0.65), (0.72, 0.75, 0.78, 1.0)
    ),
    "NightStand_1": _asset(
        "NightStand_1", "nightstand", (0.65, 0.65, 0.65), (0.48, 0.28, 0.15, 1.0)
    ),
}

TARGET_ASSETS: dict[str, tuple[str, str, str]] = {
    "couch": ("Couch_Large1", "Couch_Medium1", "Couch_Small1"),
    "chair": ("Chair_1", "Chair_2", "Chair_3"),
    "bed": ("Bed_King", "Bed_Single", "Bed_Bunk"),
    "dining table": (
        "Table_RoundLarge",
        "Table_RoundSmall",
        "Table_RoundSmall2",
    ),
}


def get_asset(asset_id: str) -> FurnitureAsset:
    try:
        return ASSETS[asset_id]
    except KeyError as error:
        raise ValueError(f"Unknown furniture asset: {asset_id}") from error


def validate_assets() -> None:
    missing = [asset.path for asset in ASSETS.values() if not asset.path.is_file()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing furniture assets: {paths}")


@lru_cache
def floor_offset(asset: FurnitureAsset) -> float:
    """Y-up OBJ vertices become Z-up after the registry's X-axis rotation."""
    vertices = [
        float(line.split()[2])
        for line in asset.path.read_text().splitlines()
        if line.startswith("v ")
    ]
    return -min(vertices) * asset.scale[1] + asset.floor_offset_m
