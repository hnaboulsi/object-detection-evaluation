"""Generate static MuJoCo scenes, run a detector, and save failure data."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
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
from scenes.generate import handwritten_scenarios, random_scenario, render_scenario
from scenes.rooms import ROOMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        required=True,
        choices=("grounding_dino", "yolo_world", "sam3"),
    )
    parser.add_argument("--room", required=True, choices=tuple(ROOMS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument(
        "--examples", action="store_true", help="Use the 18 handwritten scenarios"
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--target-obj", type=Path)
    parser.add_argument("--target-label", default="custom object")
    parser.add_argument("--target-scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.start_index < 0:
        raise SystemExit("--start-index cannot be negative")
    if args.examples and args.start_index + args.count > 18:
        raise SystemExit("The requested range exceeds the 18 handwritten examples")
    if args.examples and args.target_obj:
        raise SystemExit("--target-obj cannot be combined with --examples")
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    detector = create_detector(args.detector, device=args.device)
    examples = handwritten_scenarios() if args.examples else None
    for offset in range(args.count):
        index = args.start_index + offset
        scenario = (
            examples[index]
            if examples is not None
            else random_scenario(args.room, args.seed, index)
        )
        if args.target_obj:
            obj_path = args.target_obj.expanduser().resolve()
            scenario = replace(
                scenario,
                target_type=args.target_label,
                prompts=(args.target_label,),
                custom_target={
                    "name": "target",
                    "label": args.target_label,
                    "obj_path": str(obj_path),
                    "position": scenario.target_position,
                    "rotation_deg": (0.0, 0.0, scenario.target_yaw_deg),
                    "scale": tuple(args.target_scale),
                    "color": (0.7, 0.7, 0.7, 1.0),
                },
            )

        scenario_dir = output_root / f"scenario_{index:06d}"
        if scenario_dir.exists() and any(scenario_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite nonempty output directory: {scenario_dir}"
            )
        rendered = render_scenario(scenario, include_depth=args.save_depth)
        rgb_image = Image.fromarray(rendered.rgb, mode="RGB")
        rgb_path = scenario_dir / "rgb.png"
        annotated_path = scenario_dir / "annotated.png"
        result_path = scenario_dir / "result.json"
        prompts = args.prompts or list(scenario.prompts)
        started = time.perf_counter()
        detections = detector.detect(rendered.rgb, prompts)
        inference_time = time.perf_counter() - started
        evaluation = evaluate_single_target(detections, rendered.ground_truth)
        create_output_directory(scenario_dir)
        rgb_image.save(rgb_path)
        depth_path = None
        if rendered.depth is not None:
            depth_path = scenario_dir / "depth.npy"
            np.save(depth_path, rendered.depth)
        draw_annotations(rgb_image, detections, rendered.ground_truth.get("bbox")).save(
            annotated_path
        )
        record = make_record(
            source={
                "type": "mujoco",
                "id": scenario.scenario_id,
                "rgb_path": str(rgb_path.relative_to(output_root)),
                "annotated_path": str(annotated_path.relative_to(output_root)),
                "depth_path": str(depth_path.relative_to(output_root))
                if depth_path
                else None,
            },
            scenario=scenario.to_dict(),
            detector=detector.metadata(),
            prompts=prompts,
            detections=detections,
            ground_truth=rendered.ground_truth,
            evaluation=evaluation,
            inference_time_seconds=inference_time,
            metadata=rendered.metadata,
        )
        write_json(result_path, record)
        print(
            f"{scenario_dir.name}: TP={evaluation['true_positives']} "
            f"FP={evaluation['false_positives']} FN={evaluation['false_negatives']} "
            f"IoU={evaluation['best_iou']:.3f}"
        )

    summary_path = rebuild_summary(output_root)
    print(f"Saved {args.count} scenario(s) and {summary_path}")


if __name__ == "__main__":
    main()
