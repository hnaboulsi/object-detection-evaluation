# Object Detection Evaluation

A small research utility for asking one question: under which controlled visual conditions does an object detector fail?

It generates static MuJoCo room scenes, renders a fixed quadruped-style camera, runs an open-vocabulary detector, and saves data for analyzing false positives and false negatives. The same detector interface also runs on normal house photographs.

## Install Grounding DINO and MuJoCo

Python 3.11 is the simplest local setup on this Mac:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Grounding DINO downloads `IDEA-Research/grounding-dino-tiny` from Hugging Face on first use. If the Hugging Face Xet transport stalls, run with `HF_HUB_DISABLE_XET=1`.

## Collect MuJoCo scenarios

```bash
python collect_mujoco.py \
  --detector grounding_dino \
  --room living_room \
  --seed 7 \
  --count 20 \
  --output data/mujoco \
  --device auto
```

Add `--save-depth` to save `depth.npy`. Depth is recorded for later analysis and is never passed to the detector. Add `--examples --count 18` to render the 18 handwritten conditions instead of random scenarios. Add `--prompts couch furniture` to override each generated target prompt.

The current macOS renderer warns that `ARB_clip_control` is unavailable and depth accuracy is limited. Treat saved depth as exploratory until it is checked against known scene distances or captured on the calibrated Go2 sensor.

Each scenario is generated from `np.random.default_rng(derived_seed)`. The derived seed depends only on `--seed` and the scenario index, so rerunning the same pair reproduces the same scenario. Use `--start-index` to create a later batch without changing earlier indices.

The three rooms are `living_room`, `bedroom`, and `dining_room`. Generated values directly control target presence and pose, camera position/yaw/pitch/FOV, lighting, occluders, clutter, and similar-object distractors. `place_camera_looking_at(...)` in `scenes/generate.py` is the small helper used for front, side, rear, high, low, close, and far views.

The camera mount is only a provisional Go2-like front viewpoint. Its pose, field of view, and depth alignment have not been calibrated against a physical Unitree Go2 camera.

## Collect real-house images

```bash
python collect_images.py \
  --input house_images \
  --detector grounding_dino \
  --prompts couch bed table \
  --output data/house
```

`--input` can be one image or a directory. With no labels, detections are saved and evaluation fields are null. Optional manual ground truth is a JSON mapping from filename to one target:

```json
{
  "living-room-01.jpg": {
    "label": "couch",
    "target_present": true,
    "target_visible": true,
    "bbox": [115, 92, 540, 410]
  }
}
```

Pass it with `--ground-truth labels.json`. Real-image evaluation currently supports one target per image; visible targets require both `label` and `bbox`.

## Detector interface

All three adapters accept a path, PIL image, or NumPy RGB array and return pixel-space xyxy boxes:

```python
from detectors import create_detector

detector = create_detector("grounding_dino", device="auto")
detections = detector.detect(image, ["couch", "bed"])
```

Each result is `Detection(label, score, bbox, model_name)`. `device="auto"` selects CUDA, then MPS, then CPU. Both collectors call this same interface.

Grounding DINO uses Hugging Face Transformers. YOLO-World uses the official AILab-CVC repository and SAM 3 uses Meta's official image builder and processor. Each optional adapter loads its dependencies only when selected.

## Official YOLO-World environment

YOLO-World's official MMDetection/MMCV stack is separate and is primarily a Linux/CUDA stack:

```bash
python3.11 -m venv .venv-yolo
source .venv-yolo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-yolo-world.txt
python -m pip install --no-build-isolation mmcv==2.0.0
git clone https://github.com/onuralpszr/mmyolo.git third_party/mmyolo
git -C third_party/mmyolo checkout 4d97b3a06609dba94b8ec584be2f2029cfdb7519
python -m pip install --no-build-isolation -e third_party/mmyolo
git clone https://github.com/AILab-CVC/YOLO-World third_party/YOLO-World
git -C third_party/YOLO-World checkout 4f70adbaacf5685bd9ec5bea85f1f91057f6fc0b
python -m pip install --no-build-isolation --no-deps -e third_party/YOLO-World
mkdir -p weights
curl -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage2-4466ab94.pth \
  -o weights/s_stage2-4466ab94.pth
```

Then use `--detector yolo_world --device cpu` or CUDA. The adapter uses the official V2.1-S 1280 configuration, calls `model.reparameterize(texts)`, runs the official MMDetection test pipeline, and maps class indices back to prompts.

On the current Apple M4, MMCV 2.0.0 compiled successfully, but importing its native extension failed on a missing `at::mps::MPSStream::commit` symbol before YOLO-World inference could run. The adapter is implemented against the official API but is not runtime-verified here. Linux with an NVIDIA CUDA environment is the supported practical path.

## Official SAM 3 environment

SAM 3 needs a separate Python 3.12 environment, a supported NVIDIA CUDA system, and access to Meta's gated Hugging Face checkpoint:

```bash
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sam3.txt
huggingface-cli login
```

The adapter calls `build_sam3_image_model`, `Sam3Processor.set_image`, and `set_text_prompt`. It converts each returned mask to its enclosing xyxy bounding box and reports unavailable CUDA or checkpoint access directly.

## Custom OBJ targets

```bash
python collect_mujoco.py \
  --detector grounding_dino \
  --room living_room \
  --seed 4 \
  --count 1 \
  --output data/custom \
  --target-obj assets/my_object.obj \
  --target-label mug \
  --target-scale 0.01 0.01 0.01
```

Only OBJ geometry is loaded. One OBJ is one semantic target, MTL files and textures are ignored, and the target geoms keep stable `target__...` names for segmentation. Programmatic callers can construct the same object with `ObjectSpec.from_obj_dict(...)` using `name`, `label`, `obj_path`, `position`, and optional `rotation_deg`, `scale`, and `color`.

## Saved result

Every test has its own directory containing `rgb.png`, `annotated.png`, `result.json`, and optional `depth.npy`. Existing nonempty test directories are never overwritten. `summary.csv` is rebuilt by reading the independent JSON files.

`result.json` stores the source paths; every controlled scenario value; camera and target pose; prompts; detector model, device, and thresholds; every detection; segmentation visibility, bbox, and visible-pixel count; best IoU; TP/FP/FN; inference time; MuJoCo version; and the provisional-camera/depth flags.

MuJoCo ground truth comes from target geom segmentation. A visible target matches at IoU 0.5. An unmatched visible target is a false negative. Every unmatched prediction is a false positive. If the target is absent or fully outside/hidden from the camera, every prediction is a false positive and no false negative is counted. Real-house ground truth is manual and uses the same record fields.
