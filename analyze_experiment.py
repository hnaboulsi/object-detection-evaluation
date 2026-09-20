"""Aggregate metrics, plots, and a failure gallery for a completed run."""

from __future__ import annotations

import argparse
import shutil
import statistics
from pathlib import Path
from typing import Any

from analysis import (
    CONFIDENCE_SWEEP,
    REPORTING_THRESHOLD,
    aggregate,
    detection_ap50,
    failure_patterns,
    grouped_metrics,
    load_results,
    paired_conditions,
)
from artifacts import file_hash, write_json
from experiment import read_manifest
from preview_house import render_overview
from reporting import build_gallery, create_plots, render_report, write_metrics_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/controlled_v1"))
    parser.add_argument("--run", type=Path, default=Path("outputs/dino_controlled_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/controlled_v1"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    run = args.run.expanduser().resolve()
    report = args.report.expanduser().resolve()
    manifest = read_manifest(dataset / "manifest.json")
    records = load_results(run, manifest["manifest_hash"])
    _validate_coverage(records, manifest, allow_incomplete=args.allow_incomplete)
    if not args.allow_incomplete:
        from verify_experiment import verify_run

        verify_run(dataset, run)

    report.mkdir(parents=True, exist_ok=True)
    aggregate_record = _build_aggregate(records, manifest)
    _copy_evidence(dataset, run, report, records)
    write_json(report / "aggregate.json", aggregate_record)
    write_metrics_csv(report / "metrics.csv", aggregate_record["groups"])
    gallery = build_gallery(records, run, report)
    render_overview(report / "house_overview.png")
    create_plots(report, aggregate_record["groups"], aggregate_record["precision_recall_curve"])
    (report / "REPORT.md").write_text(
        render_report(aggregate_record, gallery),
        encoding="utf-8",
    )
    print(f"Wrote analysis for {len(records)} scene(s) to {report}")


def _validate_coverage(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    allow_incomplete: bool,
) -> None:
    if not records:
        raise ValueError("No completed results to analyze")
    expected_cases = {case["scene_id"]: case for case in manifest["cases"]}
    for record in records:
        if record["case"] != expected_cases.get(record["scene_id"]):
            raise ValueError(f"Case mismatch: {record['scene_id']}")
    if not allow_incomplete and len(records) != manifest["case_count"]:
        raise SystemExit(
            f"Expected {manifest['case_count']} results, found {len(records)}. "
            "Finish the run or pass --allow-incomplete for a smoke report."
        )


def _build_aggregate(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    overall = aggregate(records, REPORTING_THRESHOLD)
    groups = grouped_metrics(records)
    class_ap = [
        row["ap50"]
        for row in groups
        if row["dimension"] == "target_class" and row["ap50"] is not None
    ]
    overall.update(
        {
            "pooled_ap50": detection_ap50(records),
            "mean_ap50": statistics.mean(class_ap) if class_ap else None,
        }
    )
    return {
        "schema_version": 1,
        "manifest_hash": manifest["manifest_hash"],
        "completed_scenes": len(records),
        "expected_scenes": manifest["case_count"],
        "reporting_threshold": REPORTING_THRESHOLD,
        "overall": overall,
        "precision_recall_curve": [aggregate(records, value) for value in CONFIDENCE_SWEEP],
        "groups": groups,
        "detector": records[0]["detector"],
        "model_hash": records[0]["model_hash"],
        "failure_patterns": failure_patterns(records),
        "paired_conditions": paired_conditions(records),
    }


def _copy_evidence(
    dataset: Path,
    run: Path,
    report: Path,
    records: list[dict[str, Any]],
) -> None:
    shutil.copy2(dataset / "manifest.json", report / "manifest.json")
    shutil.copy2(run / "run.json", report / "run.json")
    shutil.copy2(run / "summary.csv", report / "per_scene.csv")
    checksums = {
        record["scene_id"]: {
            "rgb_sha256": record["source"]["rgb_sha256"],
            "scene_sha256": record["scene_sha256"],
            "result_sha256": file_hash(Path(record["_result_path"])),
        }
        for record in records
    }
    write_json(report / "checksums.json", checksums)


if __name__ == "__main__":
    main()
