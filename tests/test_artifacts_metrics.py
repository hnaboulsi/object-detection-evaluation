"""Checks for malformed evidence, metric accounting, and resume integrity."""

import json
from dataclasses import asdict

import pytest

from analysis import (
    aggregate,
    detection_ap50,
    failure_patterns,
    grouped_metrics,
    paired_conditions,
)
from artifacts import (
    file_hash,
    safe_artifact_path,
    validate_box,
    validate_detections,
    validate_instances,
    validate_scene_instances,
    write_json,
)
from detectors.base import Detection
from run_detector import _is_complete, _read_scene


def sample_record(identifier, predictions, truth=None):
    return {
        "scene_id": identifier,
        "case": {
            "target_label": "chair",
            "target_asset_id": "Chair_1",
            "factor": "baseline",
            "level": "baseline",
        },
        "prompts": ["chair"],
        "ground_truth": truth
        if truth is not None
        else [{"instance_id": "target", "label": "chair", "visible": True, "bbox": [0, 0, 10, 10]}],
        "detections": [
            asdict(Detection("chair", score, box, "test")) for score, box in predictions
        ],
        "inference_time_seconds": 1.0,
    }


def test_ap_uses_actual_ranked_predictions():
    # A high-scoring FP followed by one TP yields interpolated precision 1/2.
    record = sample_record("a", [(0.9, [20, 20, 30, 30]), (0.8, [0, 0, 10, 10])])
    assert detection_ap50([record]) == pytest.approx(0.5)
    assert detection_ap50([sample_record("a", [(0.9, [0, 0, 10, 10])])]) == 1.0
    assert detection_ap50([sample_record("a", [])]) == 0.0
    assert detection_ap50([sample_record("a", [], truth=[])]) is None


def test_metric_aggregation_reconciles_duplicate_and_absent_cases():
    records = [
        sample_record("a", [(0.9, [0, 0, 10, 10]), (0.8, [0, 0, 10, 10])]),
        sample_record("b", []),
        sample_record("c", [(0.7, [0, 0, 10, 10])], truth=[]),
    ]
    result = aggregate(records, 0.35)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (
        1,
        2,
        1,
    )
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(0.4)
    assert result["false_positives_per_image"] == pytest.approx(2 / 3)
    assert result["median_matched_iou"] == 1.0
    assert all(row["scene_count"] == 3 for row in grouped_metrics(records))
    patterns = failure_patterns(records)
    assert patterns == {"duplicate_fp": 1, "background_fp": 1, "no_saved_matching_candidate_fn": 1}


def test_failure_categories_and_paired_regression():
    baseline = sample_record("baseline", [(0.9, [0, 0, 10, 10])])
    changed = sample_record("occluded", [(0.2, [0, 0, 10, 10])])
    changed["case"].update({"factor": "occlusion", "level": "visible_25pct"})
    assert failure_patterns([changed]) == {"below_threshold_fn": 1}
    paired = paired_conditions([baseline, changed])
    assert paired == [
        {
            "condition": "occlusion:visible_25pct",
            "pairs": 1,
            "new_misses": 1,
            "recoveries": 0,
            "false_positive_delta": 0,
        }
    ]


@pytest.mark.parametrize(
    "box",
    [
        None,
        [],
        [0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, float("nan")],
        [0, 0, float("inf"), 1],
        ["0", 0, 1, 1],
    ],
)
def test_bad_boxes_are_rejected(box):
    with pytest.raises(ValueError):
        validate_box(box)


def test_instance_and_prediction_validation():
    item = {"instance_id": "a", "label": "chair", "bbox": [0, 0, 10, 10], "visible": True}
    validate_instances([item])
    with pytest.raises(ValueError, match="unique"):
        validate_instances([item, item])
    with pytest.raises(ValueError, match="Invisible"):
        validate_instances([{**item, "visible": False}])
    with pytest.raises(ValueError, match="scores"):
        validate_detections([{"label": "chair", "score": float("nan"), "bbox": [0, 0, 1, 1]}])


def test_rendered_scene_instance_schema():
    item = {
        "instance_id": "target",
        "label": "chair",
        "bbox": [0, 0, 10, 10],
        "visible_pixels": 100,
        "visible": True,
        "position": [1.0, 2.0, 0.0],
        "yaw_deg": -90.0,
        "asset_id": "Chair_1",
        "asset_path": "furniture/Chair_1.obj",
        "role": "target",
    }
    validate_scene_instances([item])
    with pytest.raises(ValueError, match="visible_pixels"):
        validate_scene_instances([{**item, "visible_pixels": -1}])
    with pytest.raises(ValueError, match="visibility"):
        validate_scene_instances([{**item, "visible": False, "bbox": None}])
    with pytest.raises(ValueError, match="position"):
        validate_scene_instances([{**item, "position": [0, float("nan"), 0]}])


def test_safe_paths_and_atomic_json(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_artifact_path(tmp_path, "../outside")
    path = tmp_path / "record.json"
    write_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not path.with_suffix(".json.tmp").exists()


def test_scene_integrity_and_resume_hashes(tmp_path):
    scene_dir = tmp_path / "scenes" / "a"
    scene_dir.mkdir(parents=True)
    rgb = scene_dir / "rgb.png"
    rgb.write_bytes(b"verified pixels")
    scene = {
        "schema_version": 2,
        "manifest_hash": "manifest",
        "rgb_path": "scenes/a/rgb.png",
        "rgb_sha256": file_hash(rgb),
        "objects": [],
    }
    write_json(scene_dir / "scene.json", scene)
    loaded = _read_scene(scene_dir, "manifest")
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "annotated.png").write_bytes(b"annotation")
    result = {
        "schema_version": 2,
        "manifest_hash": "manifest",
        "model_hash": "model",
        "source": {"rgb_sha256": loaded["rgb_sha256"]},
        "scene_sha256": loaded["_scene_sha256"],
    }
    write_json(result_dir / "result.json", result)
    assert _is_complete(result_dir, loaded, "manifest", "model")
    assert not _is_complete(result_dir, loaded, "manifest", "different-model")
    loaded["_scene_sha256"] = "edited ground truth"
    assert not _is_complete(result_dir, loaded, "manifest", "model")
    rgb.write_bytes(b"corrupt pixels")
    with pytest.raises(ValueError, match="checksum"):
        _read_scene(scene_dir, "manifest")
