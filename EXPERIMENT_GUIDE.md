# Experiment guide

This guide is for a student who knows basic Python but did not build the repository. The recommended path is one controlled change in `experiment.py`, followed by a fresh render directory, cached DINO inference, and inspection of the saved JSON. The canonical 216-scene evidence in `data/controlled_v1`, `outputs/dino_controlled_v1`, and `reports/controlled_v1` should be treated as frozen.

## How one scenario works

A scenario is one `ExperimentCase` record. The pipeline is:

```text
ExperimentCase
  -> MuJoCo XML house and independent OBJ furniture bodies
  -> RGB image plus geom-ID segmentation
  -> visible instance masks, pixel counts, and boxes
  -> Grounding DINO detections for the case prompt
  -> score-ordered label-and-IoU matching
  -> TP / FP / FN at confidence 0.35 and IoU 0.50
  -> scene.json and result.json
```

This is one real case from the canonical manifest, `scene_000_couch_baseline`:

```python
ExperimentCase(
    scene_id="scene_000_couch_baseline",
    seed=20260919,
    target_label="couch",
    target_asset_id="Couch_Large1",
    target_present=True,
    room="living_room",
    factor="baseline",
    level="baseline",
    camera_azimuth_deg=0.0,
    camera_distance_m=2.5,
    target_visible_fraction=1.0,
    lighting="normal",
    clutter="none",
    focal_px=None,
)
```

`scene_id` is the stable directory and result identifier. `seed` is the stored scenario identifier described below. `target_label` is the detector prompt and canonical ground-truth label. `target_asset_id` selects one OBJ variant from `scenes/assets.py`. `target_present` chooses a positive or target-absent control. `room` records the semantic room selected by `scenes.house.room_for_label`. `factor` and `level` name the controlled change, such as `viewpoint:azimuth_-45` or `lighting:dim`. `camera_azimuth_deg` rotates the camera around the target, and `camera_distance_m` changes its distance. `target_visible_fraction` requests an occlusion fraction relative to an unoccluded reference render. `lighting` selects normal, dim, bright, or side lights. `clutter` selects none, mixed, or similar-shape context objects. `focal_px` overrides both focal lengths for the focal-length conditions; `None` uses the JSON camera configuration.

## Make and run one new controlled scene

The simple, supported workflow is to edit `controlled_cases()` in `experiment.py`, render into a new directory, run DINO into a new output directory, then inspect the scene and result JSON. Do not edit the canonical directory in place. A new condition changes the manifest hash, so it is a different experiment by design.

For example, to add a couch viewed from 30 degrees, add a `replace` entry beside the existing viewpoint entries:

```python
conditions.extend(
    replace(
        base,
        factor="viewpoint",
        level="azimuth_+30",
        camera_azimuth_deg=30.0,
    )
    for _ in (0,)
)
```

In a real edit, give the new case a unique index and update the expected case count if you are intentionally making a dataset larger than 216. For a one-off experiment, the smallest direct rendering example is:

```python
from pathlib import Path

from PIL import Image

from experiment import ExperimentCase
from scenes.camera import CameraIntrinsics
from scenes.render import render_case

case = ExperimentCase(
    scene_id="student_couch_angle",
    seed=20260919,
    target_label="couch",
    target_asset_id="Couch_Large1",
    target_present=True,
    room="living_room",
    factor="viewpoint",
    level="azimuth_+30",
    camera_azimuth_deg=30.0,
)
rendered = render_case(case, CameraIntrinsics.from_json("configs/camera.json"))
output = Path("outputs/student_couch_angle")
output.mkdir(parents=True, exist_ok=True)
Image.fromarray(rendered.rgb, mode="RGB").save(output / "rgb.png")
print(rendered.ground_truth)
```

That direct snippet is useful for learning and quick visual checks. It does not write the full manifest-bound `scene.json`; use the normal CLI to produce the complete reproducible artifact:

```bash
docker compose run --rm eval python render_experiment.py --output data/student-angle --limit 1
docker compose run --rm eval python run_detector.py --dataset data/student-angle --output outputs/student-angle --device cpu --limit 1
```

The CLI uses the cases returned by `controlled_cases()`, so for a changed case it will create `data/student-angle/manifest.json`, `data/student-angle/scenes/<scene_id>/rgb.png`, and `data/student-angle/scenes/<scene_id>/scene.json`. DINO writes `outputs/student-angle/results/<scene_id>/result.json`, `annotated.png`, and a summary CSV. For a complete report over a full custom dataset, run `analyze_experiment.py`; a one-scene report needs `--allow-incomplete`.

## Common changes and the exact edit point

Change a viewing angle, distance, visible fraction, lighting, clutter level, target presence, or focal length in `experiment.py:controlled_cases`. The existing controlled conditions are the clearest examples to copy. A target-absent case uses `target_present=False` and `factor="target_presence"`; no target OBJ is added.

Change the label-to-room map in `scenes/house.py:room_for_label`. Change the canonical target position in `scenes/house.py:target_position`, the camera look-at height in `target_aim_height`, or the camera orbit calculation in `camera_pose_for`. Change the target rotation in `scenes/render.py:render_case`, where the target `SceneObject` is constructed with `yaw_deg=-90.0`. These geometry changes alter the render recipe hash and require a fresh dataset.

Change context or clutter furniture in `scenes/house.py:default_context_objects` and `clutter_objects`. Add or replace a furniture variant in `scenes/assets.py:ASSETS`, add its OBJ under `assets/furniture`, and include target variants in `TARGET_ASSETS` if the object should be a prompted target. Keep each furniture item as its own `SceneObject` and body.

Change camera intrinsics in `configs/camera.json`. `scenes.camera.CameraIntrinsics` validates and serializes the values, and `scenes.house._add_camera` passes them to MuJoCo. Camera pose is separate and is controlled by `camera_pose_for` and the case angle/distance fields.

Change detector prompts in `run_detector.py:main`, where the controlled runner uses `[case["target_label"]]`. Photograph prompts are the required `--prompts` arguments in `collect_images.py`. Change the Grounding DINO model or preprocessing only in `detectors/grounding_dino.py`; keep its `Detection` output unchanged.

Change ground truth in `scenes/render.py:_instance_ground_truth`. Change matching, thresholds, or annotation colors in `evaluation.py:evaluate_instances` and `draw_instance_annotations`. Change aggregate metrics and failure categories in `analysis.py`; change report layout and plots in `reporting.py`.

## Seeds and deterministic behavior

The current implementation stores `BASE_SEED + case_index` as `ExperimentCase.seed`. `BASE_SEED` is `20260919` in `experiment.py`, and `case_index` is the index as cases are built. For arithmetic, if a project used base seed 7 and scenario index 12, the stored identifier would be `7 + 12 = 19`.

Important: the renderer currently does not create a random-number generator and does not derive placement, camera, clutter, or lighting from `seed`. Those values come from explicit case fields and deterministic functions. Reproducibility today comes from those fields, the source/asset hashes, and the frozen manifest. The seed is still useful as a stable scenario identifier and as a place to add a carefully designed RNG later, but changing it alone does not change the current image.

## Controlled versus randomized experiments

The canonical study uses handwritten controlled cases. Each case changes one named factor from a baseline, which makes a question such as “does a 700-pixel focal length reduce recall?” easy to answer. Use this style when testing one hypothesis or when results must be paired with a baseline.

Seeded randomized scenarios would sample positions, camera poses, clutter, or lighting from an RNG derived from a base seed and case index. That would provide broader coverage while allowing any individual case to be recreated. Randomized scenario generation is not currently implemented in this repository. Do not describe the current `seed` field as if it already drives random sampling.

## How segmentation becomes ground truth

`scenes.house.build_house_xml` gives every furniture body a stable geom name beginning with `object__`. MuJoCo's segmentation render returns the geom ID at each pixel. `scenes.render._instance_ground_truth` maps those IDs back to the named `SceneObject`, counts the visible pixels, and takes the minimum and maximum visible x/y coordinates as the pixel-space `bbox`. An object with no pixels has `visible=False`, `visible_pixels=0`, and `bbox=null`.

This produces visible, not amodal, boxes. An occluded couch's box encloses only the pixels that can be seen by the camera. That differs from a manually labeled photograph, where a person may draw an amodal box around the presumed full object and may not have an exact object-ID mask. MuJoCo ground truth is repeatable because it is derived from geometry and renderer pixels, not hand judgment.

## Inspect one result

The canonical example is `outputs/dino_controlled_v1/results/scene_000_couch_baseline/result.json`. Its useful fields are:

```json
{
  "scene_id": "scene_000_couch_baseline",
  "case": {
    "seed": 20260919,
    "target_label": "couch",
    "target_present": true,
    "factor": "baseline",
    "level": "baseline"
  },
  "prompts": ["couch"],
  "ground_truth": [{"instance_id": "target", "bbox": [13.0, 169.0, 627.0, 384.0], "visible": true}],
  "detections": [{"label": "couch", "score": 0.4327195585, "bbox": [13.4805, 130.4870, 626.8204, 384.8937]}],
  "evaluation_at_0_35": {
    "confidence_threshold": 0.35,
    "iou_threshold": 0.5,
    "true_positives": 1,
    "false_positives": 1,
    "false_negatives": 0,
    "matches": [{"detection_index": 0, "instance_id": "target", "iou": 0.8443358997}]
  }
}
```

The full ground-truth list also includes context furniture, each with a stable instance ID, asset, pose, role, visible pixel count, and optional box. `target_present` says whether the queried object should exist. `detections` are every prediction saved above the collection threshold of 0.10, with a confidence score and pixel box. At the reporting threshold of 0.35, the highest-scoring box matches the target at IoU 0.844 and is one TP; the other qualifying prediction is a duplicate/background FP; there is no FN. A target-absent case has no prompted visible target, so any qualifying prediction is an FP. The `case.seed` and factor/level tell you which controlled condition produced the image.

Use these commands to inspect the actual files:

```bash
jq '.case, .ground_truth, .detections, .evaluation_at_0_35' \
  outputs/dino_controlled_v1/results/scene_000_couch_baseline/result.json
open data/controlled_v1/scenes/scene_000_couch_baseline/rgb.png
open outputs/dino_controlled_v1/results/scene_000_couch_baseline/annotated.png
```

The source scene JSON contains the exact case, RGB/XML hashes, metadata, and segmentation-derived object list. The result JSON repeats the ground truth and adds detector metadata, predictions, matching, and inference time.

## The one recommended workflow for a new couch angle

Copy an existing viewpoint condition in `experiment.py:controlled_cases`, set `level="azimuth_+30"` and `camera_azimuth_deg=30.0`, and give the case a unique scene ID/index. Render into a new directory with `render_experiment.py`, run `run_detector.py` on that directory, open its `rgb.png` and `annotated.png`, then inspect `scene.json` and `result.json` with `jq`. Do not edit the canonical manifest or reuse its output directories for the new condition.
