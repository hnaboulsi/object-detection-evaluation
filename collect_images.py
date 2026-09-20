"""Run the same detector record format on one image or a house-image folder."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image

from detectors import create_detector
from results import (
    create_output_directory,
    draw_annotations,
    evaluate_single_target,
    make_record,
    rebuild_summary,
    write_json,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--detector",
        required=True,
        choices=("grounding_dino", "yolo_world", "sam3"),
    )
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional JSON mapping image filename to one manually labeled target",
    )
    return parser.parse_args()


def input_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Input not found: {path}")
    images = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No supported images found in {path}")
    return images


def load_ground_truth(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    try:
        values = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read ground truth JSON: {error}") from error
    if not isinstance(values, dict):
        raise SystemExit("Ground truth JSON must map image filenames to target records")
    return values


def normalize_ground_truth(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Each ground-truth entry must be an object")
    bbox = value.get("bbox")
    if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
        raise ValueError("Ground-truth bbox must be [x_min, y_min, x_max, y_max]")
    present = bool(value.get("target_present", bbox is not None))
    visible = bool(value.get("target_visible", present and bbox is not None))
    label = value.get("label")
    if visible and not label:
        raise ValueError("A visible ground-truth target requires a label")
    return {
        "label": label,
        "target_present": present,
        "target_visible": visible,
        "bbox": [float(item) for item in bbox] if bbox is not None else None,
        "visible_pixels": None,
        "annotation_source": "manual",
    }


def safe_stem(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)[:60] or "image"


def main() -> None:
    args = parse_args()
    source_path = args.input.expanduser().resolve()
    images = input_images(source_path)
    labels = load_ground_truth(args.ground_truth)
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    detector = create_detector(args.detector, device=args.device)

    for index, image_path in enumerate(images):
        image_dir = output_root / f"image_{index:06d}_{safe_stem(image_path)}"
        if image_dir.exists() and any(image_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite nonempty output directory: {image_dir}"
            )
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        rgb_path = image_dir / "rgb.png"
        annotated_path = image_dir / "annotated.png"
        result_path = image_dir / "result.json"
        started = time.perf_counter()
        detections = detector.detect(image, args.prompts)
        inference_time = time.perf_counter() - started
        raw_ground_truth = labels.get(image_path.name, labels.get(image_path.stem))
        ground_truth = (
            normalize_ground_truth(raw_ground_truth)
            if raw_ground_truth is not None
            else None
        )
        evaluation = (
            evaluate_single_target(detections, ground_truth)
            if ground_truth is not None
            else None
        )
        create_output_directory(image_dir)
        image.save(rgb_path)
        draw_annotations(
            image,
            detections,
            ground_truth.get("bbox") if ground_truth else None,
        ).save(annotated_path)
        record = make_record(
            source={
                "type": "real_image",
                "id": image_path.stem,
                "original_path": str(image_path),
                "rgb_path": str(rgb_path.relative_to(output_root)),
                "annotated_path": str(annotated_path.relative_to(output_root)),
                "depth_path": None,
            },
            scenario=None,
            detector=detector.metadata(),
            prompts=args.prompts,
            detections=detections,
            ground_truth=ground_truth,
            evaluation=evaluation,
            inference_time_seconds=inference_time,
            metadata={
                "image_width": image.width,
                "image_height": image.height,
                "depth_used_by_detector": False,
            },
        )
        write_json(result_path, record)
        print(f"{image_path.name}: {len(detections)} detection(s)")

    summary_path = rebuild_summary(output_root)
    print(f"Saved {len(images)} image result(s) and {summary_path}")


if __name__ == "__main__":
    main()
