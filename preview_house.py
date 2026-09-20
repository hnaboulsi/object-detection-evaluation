"""Render a roof-hidden overview for inspecting the three-room house layout."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from scenes.camera import CameraIntrinsics, CameraPose
from scenes.house import SceneObject, build_house_xml, default_context_objects, target_position
from scenes.render import _render_xml


def render_overview(output: Path) -> None:
    """Render the same house and furnishings with the ceiling omitted."""
    intrinsics = CameraIntrinsics(1280, 960, 1050, 1050, 639.5, 479.5)
    furniture = default_context_objects("couch")
    furniture.append(
        SceneObject("target", "couch", "Couch_Large1", target_position("couch"), -90, "target")
    )
    xml = build_house_xml(
        model_name="house_overview",
        camera=CameraPose((18, -20, 24), (0, 5, 0)),
        intrinsics=intrinsics,
        objects=furniture,
        lighting="bright",
    )
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.attrib.get("name") == "house_ceiling":
            world.remove(geom)
    rgb, _, _, _ = _render_xml(ET.tostring(root, encoding="unicode"), intrinsics, rgb=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/house_overview.png"))
    args = parser.parse_args()
    render_overview(args.output)
    print(f"Saved roof-hidden layout preview: {args.output}")


if __name__ == "__main__":
    main()
