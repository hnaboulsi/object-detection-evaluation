# Object Detection Evaluation

A small, reproducible study of when Grounding DINO fails in furnished MuJoCo rooms.

The house shell is built from MuJoCo boxes: floors, walls, door openings, and a ceiling. Furniture comes from separate CC0 OBJ files, with independent poses and instance labels. Render once, then run detectors against the same saved images. Camera intrinsics are explicit and editable.

This release evaluates **Grounding DINO Tiny only**. The shared detector contract is kept for future Linux adapters, but this repository contains one validated adapter and one canonical study.

## Quick start

Install Docker with Docker Compose and start the Docker engine. Allow roughly 8 GB of container memory and 5 GB of free disk space. The CPU image supports Linux ARM64 and AMD64. First inference downloads about 700 MB of public model assets; no account or token is required.

Run these commands from a terminal. The repository is private, so Git must have access to it.

```bash
git clone https://github.com/hnaboulsi/object-detection-evaluation.git
cd object-detection-evaluation
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
mkdir -p .cache data outputs reports
docker compose build
docker compose run --rm eval python render_experiment.py
docker compose run --rm eval python run_detector.py --device cpu
docker compose run --rm eval python analyze_experiment.py
docker compose run --rm eval python verify_experiment.py --run outputs/dino_controlled_v1 --output reports/controlled_v1/validation.json
```

Open [the report](reports/controlled_v1/REPORT.md). It contains class and condition breakdowns, precision-recall plots, and annotated failure examples. Cyan boxes show visible instance ground truth; green boxes show predictions at confidence 0.35.

To inspect the layout from above, run `docker compose run --rm eval python preview_house.py`.
It saves `outputs/house_overview.png` with only the ceiling hidden for the preview.

The full study contains 216 images and takes minutes to tens of minutes on CPU, depending on available cores. Each command prints progress. Model downloads can take longer on the first run. The container uses four inference threads by default.

For a quick end-to-end check in separate directories:

```bash
docker compose run --rm eval python render_experiment.py --output data/smoke --limit 1
docker compose run --rm eval python run_detector.py --dataset data/smoke --output outputs/smoke --device cpu --limit 1
docker compose run --rm eval python analyze_experiment.py --dataset data/smoke --run outputs/smoke --report outputs/smoke-report --allow-incomplete
```

## Resume a run

Use the same configuration and add `--resume`:

```bash
docker compose run --rm eval python render_experiment.py --resume
docker compose run --rm eval python run_detector.py --device cpu --resume
```

Completed scenes are skipped only when the manifest, case, and image checksums match. Completed detector results also require matching scene and model hashes, including the checkpoint checksum and inference settings. A conflicting or corrupted completed artifact produces an error. Use a new output directory for changed code, assets, camera settings, or detector thresholds.

If an interruption leaves an unfinished per-scene directory, the error identifies it. Move that specific directory outside the dataset or run, then resume. Other completed scenes are preserved. Keep the model cache to avoid downloading weights again.

## Change the camera

Edit [configs/camera.json](configs/camera.json):

```json
{
  "width": 640,
  "height": 480,
  "fx": 525.0,
  "fy": 525.0,
  "cx": 319.5,
  "cy": 239.5
}
```

Focal lengths and principal points are in pixels. Pixel centers use a top-left origin; image-right is positive X and image-down is positive Y. The centered principal point is `((width - 1) / 2, (height - 1) / 2)`. Values are converted to MuJoCo's centered sensor convention internally and verified with rendered projection tests. Unequal `fx` and `fy` are supported. This is a pinhole camera without lens distortion.

These values govern every condition except the explicitly named focal-length experiments, which set both focal lengths to 400 or 700 pixels. Image resolution is passed to both the MuJoCo camera and rendering context. Camera pose is separate from intrinsics.

After editing the configuration, select fresh directories:

```bash
docker compose run --rm eval python render_experiment.py --camera-config configs/camera.json --output data/custom-camera --limit 1
docker compose run --rm eval python run_detector.py --dataset data/custom-camera --output outputs/custom-camera --device cpu --limit 1
docker compose run --rm eval python analyze_experiment.py --dataset data/custom-camera --run outputs/custom-camera --report outputs/custom-camera-report --allow-incomplete
```

This preview renders one scene; omit `--limit 1` and `--allow-incomplete` for a full run.
The default camera is synthetic. No physical-camera calibration or simulation-to-real accuracy is claimed.

## What the experiment measures

Four prompted classes, each with three OBJ variants: couch, chair, bed, and dining table. Every variant has 17 positive conditions: baseline; four azimuths; two distances; three measured occlusion levels; three lighting conditions; two clutter conditions; and two focal lengths. Twelve absent-class controls bring the total to 216.

Each positive condition changes one factor relative to its asset's baseline. Azimuth is measured from the front view; the baseline distance is 2.5 m and camera height is 0.55 m. The occlusion search uses a separate reference rendering and adjusts a plain blocker until the target's visible-pixel count is close to 75%, 50%, or 25% of that reference. Exact achieved fractions are saved.

The study uses one class prompt per image. All visible instances of that prompted class are eligible ground truth. A score-ordered, one-to-one matcher at IoU 0.50 counts duplicate detections as false positives. Other furniture is annotated for inspection but is not a missed detection when it was not prompted. Absent-class controls contain no visible instance of the queried class.

Grounding DINO saves predictions above 0.10 with text threshold 0.10. The primary results use confidence 0.35. AP50 uses every saved score with 101-point interpolated precision; results below the collection threshold are unavailable. The report distinguishes pooled AP from the mean of per-class AP values. Thresholds are fixed before the batch; they are not selected to maximize these results.

Ground-truth boxes enclose **visible** segmentation pixels, not the hidden extent of an occluded object. Thus a reasonable whole-object prediction can fail IoU under heavy occlusion. Close views and long focal lengths can also crop objects. Low-poly geometry, flat per-object colors, a fixed house layout, and only three variants per class limit generalization. Treat the results as controlled failure probes, not a broad model ranking.

## Files and code

`data/controlled_v1/manifest.json` fixes every case, camera configuration, asset checksum, and rendering-source checksum. Each scene directory holds `rgb.png` and schema-v2 `scene.json`.

`outputs/dino_controlled_v1/` holds the detector configuration, per-image predictions and annotations, and a summary CSV. Raw runs and model caches are ignored by Git.

`reports/controlled_v1/` holds the compact shareable evidence: manifest, model configuration, checksums, per-scene counts, aggregates, plots, and a curated gallery. Raw model weights and all rendered images are not committed.

The main entry points are `render_experiment.py` (render the frozen manifest), `run_detector.py` (run Grounding DINO), `analyze_experiment.py` (write the report), and `collect_images.py` (optional photographs). Scenario definitions are in `experiment.py`; MuJoCo construction and segmentation ground truth are in `scenes/`; IoU matching is in `evaluation.py`; aggregate metrics are in `analysis.py`. Edit `experiment.py:controlled_cases` to add or change a controlled condition.

`scenes/house.py` defines the house and object placements; `scenes/assets.py` is the small asset registry. Scale is expressed in source OBJ axes, with Y-up converted to MuJoCo Z-up and automatically aligned to the floor. Materials use the registry's flat display colors; original MTL files are retained for provenance but not rendered.

The shared detector contract is small and remains available to future adapters:

```python
from detectors import GroundingDINO

detector = GroundingDINO(device="cpu")
detections = detector.detect("image.png", ["couch", "bed"])
# Detection(label, score, bbox, model_name), with pixel-space xyxy boxes.
```

Schema-v1 random-scene outputs and the old `collect_mujoco.py` path have been replaced by the frozen-manifest workflow. Rerender old experiments using the new commands.

Read [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) for the file-by-file architecture and [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for the practical workflow for making and inspecting scenes.

## Photographs

For photographs, pass a single image or directory. Optional ground truth maps filenames to lists of records with `instance_id`, `label`, `visible`, and `bbox`. An empty list means the queried objects are absent; a missing filename means unlabeled.

```bash
docker compose run --rm eval python collect_images.py --input house_images --prompts couch bed --output outputs/photographs --device cpu
```

The photographs must be inside the checkout so the container can see them. Unlabeled photographs save predictions with null evaluation fields.

## Development and checks

The test image runs without downloading detector weights:

```bash
docker build --target test -t object-detection-evaluation:test .
docker run --rm object-detection-evaluation:test
docker run --rm object-detection-evaluation:test sh -c 'ruff check . && ruff format --check .'
```

CI builds and tests on Linux AMD64 and ARM64. It covers rendering, camera projection, asset alignment, deterministic manifests, matching, metrics, and artifact validation.

## Troubleshooting

If Docker cannot connect, start the Docker engine and retry. If a mounted directory is not writable, export `LOCAL_UID` and `LOCAL_GID` as shown above, and create the output/cache directories before running containers.

For a model-download timeout, rerun with `--resume`; Hugging Face reuses cached files. The default configuration disables Xet transport. The container is CPU-only, so do not request `--device cuda` inside it.

An out-of-memory exit usually means Docker's memory allowance is too small. Increase it to at least 8 GB and keep one detector process running at a time. Thread count is adjustable through the Compose environment if CPU oversubscription affects throughput.

To reclaim storage, stop the containers, then remove only the specific generated dataset/run directories you no longer need. Deleting `.cache/huggingface` also removes downloaded model weights and forces another download. Reports and source assets are separate from those generated directories.

## Assets and model provenance

Furniture: Quaternius, “Ultimate House Interior Pack,” June 2020, CC0 1.0. The selected 16 OBJ/MTL pairs and supplied license are included in `assets/`.

Grounding DINO Tiny: `IDEA-Research/grounding-dino-tiny`, pinned revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. Each run records the downloaded weight checksum, PyTorch and Transformers versions, device, and thresholds. Linux container validation and executed experiment evidence are documented in the report.
