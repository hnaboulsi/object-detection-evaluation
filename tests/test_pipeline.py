"""Behavioral checks that need no detector weights."""

import json
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pytest

from detectors.base import Detection
from experiment import build_manifest, controlled_cases, read_manifest, write_manifest
from results import evaluate_instances
from scenes.assets import ASSETS, floor_offset
from scenes.camera import CameraIntrinsics, CameraPose
from scenes.house import build_house_xml
from scenes.render import _render_xml, render_case


def test_manifest_design_and_determinism(tmp_path):
    cases = controlled_cases()
    assert len(cases) == len({c.scene_id for c in cases}) == 216
    assert Counter(c.target_present for c in cases) == {True: 204, False: 12}
    positive_counts = Counter(c.target_asset_id for c in cases if c.target_present)
    assert len(positive_counts) == 12 and set(positive_counts.values()) == {17}
    path = tmp_path / "manifest.json"
    expected = write_manifest(path, CameraIntrinsics())
    assert read_manifest(path) == expected == build_manifest(CameraIntrinsics())
    values = json.loads(path.read_text())
    values["cases"][0]["target_present"] = False
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="hash"):
        read_manifest(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 0),
        ("height", 1.5),
        ("fx", -1),
        ("fy", float("nan")),
        ("cx", 640),
        ("cy", -1),
        ("fx", float("inf")),
    ],
)
def test_invalid_intrinsics(field, value):
    with pytest.raises(ValueError):
        CameraIntrinsics.from_dict({**CameraIntrinsics().to_dict(), field: value})


def test_floor_alignment():
    for asset in ASSETS.values():
        points = np.array(
            [
                [float(x) for x in line.split()[1:4]]
                for line in asset.path.read_text().splitlines()
                if line.startswith("v ")
            ]
        )
        bottom = points[:, 1].min() * asset.scale[1] + floor_offset(asset)
        assert bottom == pytest.approx(asset.floor_offset_m)


@pytest.mark.parametrize(
    "intrinsics",
    [
        CameraIntrinsics(),
        CameraIntrinsics(800, 600, 610, 480, 320, 360),
    ],
)
def test_rendered_projection_matches_requested_intrinsics(intrinsics):
    # Camera looks down -Z, so world X is image right and world Y is image up.
    xml = build_house_xml(
        model_name="projection",
        camera=CameraPose((0, 0, 2), (0, 0, 0)),
        intrinsics=intrinsics,
        objects=[],
        lighting="normal",
    )
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        world.remove(geom)
    # Use explicit orientation for the vertical camera to avoid look-at singularity.
    world.find("camera").set("xyaxes", "1 0 0 0 1 0")
    ET.SubElement(
        world, "geom", name="marker", type="sphere", pos="0.3 0.2 0", size="0.012", rgba="1 0 0 1"
    )
    _, segmentation, model, _ = _render_xml(
        ET.tostring(root, encoding="unicode"), intrinsics, rgb=False
    )
    import mujoco

    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "marker")
    rows, columns = np.nonzero(segmentation[:, :, 0] == marker)
    assert len(rows) > 0
    assert columns.mean() == pytest.approx(intrinsics.cx + intrinsics.fx * 0.3 / 2, abs=1.0)
    assert rows.mean() == pytest.approx(intrinsics.cy - intrinsics.fy * 0.2 / 2, abs=1.0)


def truth(identifier="target", label="chair", box=(0, 0, 10, 10)):
    return {"instance_id": identifier, "label": label, "visible": True, "bbox": box}


def detection(score=0.9, box=(0, 0, 10, 10), label="chair"):
    return Detection(label, score, box, "test")


def test_multi_instance_and_duplicate_matching():
    truths = [truth(), truth("other", box=(20, 0, 30, 10))]
    predictions = [detection(), detection(0.8), detection(0.7, (20, 0, 30, 10))]
    result = evaluate_instances(predictions, truths, ["chair"])
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (
        2,
        1,
        0,
    )
    assert result["unmatched_detection_indices"] == [1]


def test_absent_hidden_unprompted_and_low_score():
    truths = [truth(label="bed"), {**truth(), "visible": False, "bbox": None}]
    result = evaluate_instances([detection(), detection(0.1)], truths, ["chair"])
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (
        0,
        1,
        0,
    )
    result = evaluate_instances([detection(label="wheelchair")], [truth()], ["chair"])
    assert result["true_positives"] == 0
    assert result["false_negatives"] == 1


@pytest.mark.parametrize("index", [0, 51, 102, 153])
def test_positive_baselines_are_visible(index):
    rendered = render_case(controlled_cases()[index], CameraIntrinsics())
    assert rendered.rgb.shape == (480, 640, 3)
    target = next(item for item in rendered.ground_truth if item["role"] == "target")
    assert target["visible_pixels"] > 100
    assert target["bbox"] is not None


def test_occlusion_is_measured():
    rendered = render_case(controlled_cases()[8], CameraIntrinsics())
    assert rendered.actual_target_visible_fraction == pytest.approx(0.5, abs=0.03)


def test_intrinsics_config_changes_render():
    case = controlled_cases()[0]
    normal = render_case(case, CameraIntrinsics())
    wide = render_case(case, CameraIntrinsics(fx=400, fy=420))
    assert wide.metadata["camera_intrinsics"]["fx"] == 400
    assert wide.metadata["camera_intrinsics"]["fy"] == 420
    assert not np.array_equal(normal.rgb, wide.rgb)
