"""Run the shared detector on photographs with optional multi-instance labels."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from artifacts import file_hash, read_json, validate_instances, write_json
from detectors import GroundingDINO
from evaluation import draw_instance_annotations, evaluate_instances

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def input_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        raise ValueError(f"Image or directory not found: {path}")
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images in {path}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompts", required=True, nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="JSON mapping image filenames to lists of instance records",
    )
    args = parser.parse_args()
    labels = read_json(args.ground_truth) if args.ground_truth else {}
    images = input_images(args.input)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {args.output}")
    for instances in labels.values():
        validate_instances(instances)
    detector = GroundingDINO(device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images):
        directory = args.output / f"image_{index:06d}"
        directory.mkdir()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        started = time.perf_counter()
        predictions = detector.detect(image, args.prompts)
        seconds = time.perf_counter() - started
        instances = labels.get(path.name)
        evaluation = (
            evaluate_instances(predictions, instances, args.prompts)
            if instances is not None
            else None
        )
        draw_instance_annotations(image, predictions, instances or []).save(
            directory / "annotated.png"
        )
        write_json(
            directory / "result.json",
            {
                "schema_version": 2,
                "source_type": "photograph",
                "filename": path.name,
                "rgb_sha256": file_hash(path),
                "ground_truth": instances,
                "detections": [asdict(p) for p in predictions],
                "prompts": args.prompts,
                "detector": detector.metadata(),
                "evaluation_at_0_35": evaluation,
                "inference_time_seconds": seconds,
            },
        )
        print(f"{path.name}: {len(predictions)} predictions")


if __name__ == "__main__":
    main()
