# Codebase guide

This repository is a small, manifest-first evaluation pipeline. `experiment.py` defines the 216 controlled cases. `render_experiment.py` turns those cases into immutable RGB images, segmentation-derived ground truth, and scene JSON. `run_detector.py` reads those saved scenes and writes Grounding DINO predictions and evaluation JSON. `analyze_experiment.py` loads the saved results, aggregates metrics, makes plots, and writes the research report.

The code is intentionally split by responsibility. A target of 21 Python files is not a design requirement. Each remaining file has an independent job that can be described in one sentence; files should only be merged if that remains true after the merge.

## End-to-end flow

```text
ExperimentCase
  -> experiment.py builds the deterministic manifest
  -> scenes/house.py builds a MuJoCo XML house and furniture bodies
  -> scenes/render.py renders RGB plus object-ID segmentation
  -> scenes/render.py converts visible pixels into ground truth boxes
  -> run_detector.py loads the frozen RGB and calls GroundingDINO
  -> evaluation.py greedily matches score-ordered detections by label and IoU
  -> TP / FP / FN are saved in result.json
  -> analysis.py aggregates results and reporting.py writes CSV, plots, gallery, and REPORT.md
```

The renderer and detector are separate on purpose. A future detector sees exactly the same manifest and pixels rather than a newly rendered approximation.

## Root modules

`artifacts.py` contains the schema version, JSON/file hashing, atomic JSON writing, safe relative-path handling, and validation for instance and detection records. It is called by the renderer, detector runner, analysis, verification, and tests. Inputs are Python values or artifact paths; outputs are validated dictionaries, hashes, or files. It is essential infrastructure.

`experiment.py` defines the `ExperimentCase` dataclass, the `controlled_cases()` list of 216 cases, manifest construction and hashing, and manifest read/write validation. `render_experiment.py`, `verify_experiment.py`, and tests call it; it depends on the target asset registry and room mapping. It takes camera intrinsics and case definitions and produces the manifest consumed by every later stage. It is essential and remains separate because the manifest is the experiment contract.

`render_experiment.py` is the command-line renderer. It loads `configs/camera.json`, creates or validates `manifest.json`, calls `scenes.render.render_case`, saves `rgb.png`, and writes `scene.json`. It depends on `experiment.py`, `scenes.camera`, `scenes.render`, and artifact helpers. Its input is a manifest configuration and output directory; its output is one directory per scene. It is essential.

`run_detector.py` is the command-line inference runner. It validates each frozen scene, calls the Grounding DINO adapter, draws annotations, evaluates at the reporting threshold, and writes per-scene `result.json` plus `summary.csv`. It depends on `detectors`, `evaluation.py`, `experiment.py`, and artifact helpers. Its inputs are rendered scenes and model settings; its outputs are predictions and TP/FP/FN records. It is essential for the DINO study.

`analysis.py` loads and validates a completed run, recomputes evaluations, aggregates overall and grouped metrics, computes AP50 and confidence sweeps, classifies failure patterns, and pairs conditions with their baselines. `analyze_experiment.py`, `verify_experiment.py`, `reporting.py`, and tests call it. It depends on `evaluation.py` and artifact validation. Its input is result JSON; its output is metric dictionaries. It is essential analysis logic.

`analyze_experiment.py` is the report-generation command. It checks coverage against the manifest, calls `analysis.py`, copies evidence, calls `reporting.py`, and writes `reports/controlled_v1`. It depends on `preview_house.py` for the overview image and on the analysis/reporting modules. Its input is a dataset and detector run; its output is aggregate JSON, CSV, plots, gallery, and Markdown. It is an essential CLI, not a second metric implementation.

`evaluation.py` owns reusable image-level evaluation. `bbox_iou` computes pixel-space IoU, `canonical_label` normalizes prompts and labels, `evaluate_instances` performs score-ordered one-to-one matching at the shared 0.5 IoU threshold, and `draw_instance_annotations` creates cyan ground-truth and green prediction overlays. The detector runner, photograph collector, analysis, and tests call it. Inputs are detections, instance records, and prompts; output is an evaluation dictionary or annotated image. It is essential and deliberately separate from batch aggregation.

`reporting.py` formats analysis outputs. It writes `metrics.csv`, selects a small failure gallery, creates the three plots, and renders the Markdown report. It is called only by `analyze_experiment.py` and depends on `analysis.py` for the reporting threshold and record evaluation. Inputs are aggregate rows and saved records; outputs are report files and gallery metadata. It is essential report presentation code.

`verify_experiment.py` verifies that rendered scenes and detector results still match the manifest, source checksums, schema, and recomputed metrics without loading model weights. `verify_dataset` checks all 216 scenes; `verify_run` additionally checks all 216 result records. It is called as a CLI and by `analyze_experiment.py`. It depends on `analysis.py`, `experiment.py`, camera serialization, and artifact validation. Its output is a validation dictionary or an error. It is essential evidence validation.

`preview_house.py` renders a top-down overview with only the ceiling hidden. `render_overview` is called by report generation and the file can also be run as a CLI. It depends on the house builder and camera types and produces one PNG. It is optional presentation code, not part of per-scene evaluation.

`collect_images.py` is the explicit photograph path. It loads one image or a directory, runs the same Grounding DINO adapter, optionally loads filename-to-instance ground truth, reuses evaluation and annotation helpers, and writes one result directory per image. It depends on `artifacts.py`, `detectors`, and `evaluation.py`. It is optional for the canonical MuJoCo experiment but useful for real-image smoke checks; it remains separate because photograph inputs do not have MuJoCo scene metadata.

## Detector modules

`detectors/base.py` defines the shared `Detection` dataclass, the `Detector` protocol, RGB image loading, prompt cleaning, and device resolution. Grounding DINO, the two CLIs, photograph collection, and tests depend on it. Inputs are paths, PIL images, or NumPy RGB arrays and prompt sequences; outputs are normalized images, prompts, or detector records. It is essential shared contract code.

`detectors/grounding_dino.py` implements the only validated detector adapter, `GroundingDINO`. It loads the pinned Hugging Face processor/model revision, runs prompt-based inference, converts boxes to pixel-space `Detection` objects, and reports model/checkpoint metadata. `run_detector.py` and `collect_images.py` instantiate it directly. Its inputs are an image and canonical prompts; its output is a list of detections. It is essential for this study.

`detectors/__init__.py` is the small public detector namespace exporting `Detection`, `Detector`, and `GroundingDINO`. It has no logic and exists to make imports readable. It is convenience infrastructure and could be removed without changing behavior, but keeping it makes the adapter boundary obvious.

## MuJoCo scene modules

`scenes/camera.py` defines `CameraIntrinsics` and `CameraPose`. Intrinsics validate and serialize width, height, focal lengths, and principal point; poses hold camera position and look-at coordinates. The renderer, house builder, tests, and manifest use these types. Inputs are JSON or numeric values; outputs are validated camera objects and dictionaries. It is essential and keeps camera configuration independent from scene geometry.

`scenes/assets.py` defines `FurnitureAsset`, the `ASSETS` registry, target asset variants, OBJ lookup, asset validation, and floor offsets. `scenes.house`, `scenes.render`, and tests use it. Inputs are asset IDs and OBJ files; outputs are scale/color/label metadata and floor-alignment values. It is essential asset configuration.

`scenes/house.py` defines `SceneObject` and `Occluder`, maps labels to rooms and positions, computes camera poses, chooses context/clutter furniture, and builds the complete MuJoCo XML from primitive house geoms plus separate OBJ bodies. `scenes.render.py`, `preview_house.py`, and tests call it. Inputs are case factors, camera types, object lists, lighting, and optional occluder; output is XML text. It is essential scene construction and intentionally separate from rendering.

`scenes/render.py` defines `RenderedScene` and `render_case`. It turns an `ExperimentCase` into object placements, performs the reference render used to tune occlusion, renders RGB and segmentation, and converts segmentation IDs into schema-v2 instance records. It depends on `scenes.assets`, `scenes.camera`, and `scenes.house`, plus MuJoCo and NumPy. Inputs are one case and base intrinsics; outputs are RGB, ground truth, XML hash, and metadata. It is essential rendering/evidence code.

`scenes/__init__.py` marks the scene directory as a package and carries no behavior. It is optional in modern namespace-package Python, but retained for clear package structure.

## Tests

`tests/test_pipeline.py` checks manifest determinism and 216-case coverage, camera validation and projection, asset floor alignment, rendering smoke cases, occlusion measurement, multi-instance matching, absent targets, and camera changes. It calls `experiment.py`, `scenes.*`, and `evaluation.py`. It is verification code and is essential to maintain the scientific contract.

`tests/test_artifacts_metrics.py` checks JSON/path safety, schema validation, AP and aggregate accounting, failure categories, paired conditions, and detector resume hashes. It calls `artifacts.py`, `analysis.py`, and private resume helpers in `run_detector.py`. It is verification code and is essential to prevent silent evidence corruption.

## Data and report paths

The canonical rendered dataset is `data/controlled_v1`. Each scene contains `rgb.png` and `scene.json`, and the dataset manifest fixes case fields, camera values, source hashes, and the 216-scene manifest hash. The canonical DINO run is `outputs/dino_controlled_v1`; each result directory contains `result.json` and `annotated.png`. The shareable research evidence is `reports/controlled_v1`, including `aggregate.json`, `metrics.csv`, `per_scene.csv`, plots, and the final `REPORT.md`.

## Renamed, removed, and intentionally preserved modules

`results.py` was renamed to `evaluation.py` because it evaluates one image and draws annotations rather than representing all results. `analysis_metrics.py` was renamed to `analysis.py` because it owns the complete batch-analysis layer. The old `detectors/create.py` factory was removed so the only supported path is an explicit `GroundingDINO` adapter. The unvalidated `detectors/yolo_world.py`, `detectors/sam3.py`, `requirements-yolo-world.txt`, and `requirements-sam3.txt` were removed rather than documented as supported code. The house, render, experiment, and asset boundaries were not merged because each has a distinct scientific responsibility and their source hashes are embedded in the canonical manifest.
