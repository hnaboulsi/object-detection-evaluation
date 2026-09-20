"""Write the human-readable and machine-readable analysis artifacts."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from analysis import REPORTING_THRESHOLD, evaluate_record

Record = dict[str, Any]


def write_metrics_csv(path: Path, rows: list[Record]) -> None:
    fields = [
        "dimension",
        "value",
        "scene_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "false_positives_per_image",
        "median_matched_iou",
        "median_inference_time_seconds",
        "ap50",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_gallery(records: list[Record], run: Path, report: Path) -> list[dict[str, str]]:
    gallery_dir = report / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    ranked = _rank_failures(records)
    selected: list[dict[str, str]] = []
    used_groups: set[tuple[str, str]] = set()

    for _, record, evaluation in ranked:
        group = (record["case"]["target_label"], record["case"]["factor"])
        if group in used_groups and len(selected) < 8:
            continue
        source = run / record["source"]["annotated_path"]
        if not source.is_file():
            raise FileNotFoundError(f"Annotated image is missing: {source}")
        destination = gallery_dir / f"{record['scene_id']}.png"
        shutil.copy2(source, destination)
        selected.append(
            {
                "scene_id": record["scene_id"],
                "image": f"gallery/{destination.name}",
                "summary": (
                    f"{record['case']['target_label']}, "
                    f"{record['case']['factor']}:{record['case']['level']}, "
                    f"TP={evaluation['true_positives']} "
                    f"FP={evaluation['false_positives']} FN={evaluation['false_negatives']}"
                ),
            }
        )
        used_groups.add(group)
        if len(selected) == 12:
            break
    return selected


def _rank_failures(records: list[Record]) -> list[tuple[int, Record, Record]]:
    ranked = []
    for record in records:
        evaluation = evaluate_record(record, REPORTING_THRESHOLD)
        severity = 10 * evaluation["false_negatives"] + 4 * evaluation["false_positives"]
        if severity:
            ranked.append((severity, record, evaluation))
    return sorted(ranked, key=lambda item: (-item[0], item[1]["scene_id"]))


def create_plots(report: Path, grouped_rows: list[Record], curve: list[Record]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Analysis requires matplotlib") from error

    class_rows = [item for item in grouped_rows if item["dimension"] == "target_class"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([item["value"] for item in class_rows], [item["recall"] for item in class_rows])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall at confidence 0.35 and IoU 0.50")
    axis.set_title("Grounding DINO recall by target class")
    figure.tight_layout()
    figure.savefig(report / "recall_by_class.png", dpi=160)
    plt.close(figure)

    factor_rows = [
        item
        for item in grouped_rows
        if item["dimension"] == "factor_level" and not item["value"].startswith("target_presence:")
    ]
    factor_rows.sort(key=lambda item: item["recall"])
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh([item["value"] for item in factor_rows], [item["recall"] for item in factor_rows])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Recall at confidence 0.35 and IoU 0.50")
    axis.set_title("Recall by controlled condition")
    figure.tight_layout()
    figure.savefig(report / "recall_by_condition.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([item["recall"] for item in curve], [item["precision"] for item in curve], marker="o")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Confidence-swept precision-recall curve")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(report / "precision_recall.png", dpi=160)
    plt.close(figure)


def render_report(aggregate_record: Record, gallery: list[dict[str, str]]) -> str:
    overall = aggregate_record["overall"]
    lines = [
        "# Controlled Grounding DINO House Evaluation",
        "",
        "This report describes one controlled synthetic evaluation of Grounding DINO Tiny. "
        "It is not a cross-detector comparison and does not validate a physical camera.",
        "",
        "![Three-room MuJoCo layout, ceiling hidden for inspection](house_overview.png)",
        "",
        "## Run summary",
        "",
        f"- Completed scenes: {aggregate_record['completed_scenes']} / "
        f"{aggregate_record['expected_scenes']}",
        f"- Reporting threshold: {aggregate_record['reporting_threshold']:.2f}",
        "- IoU threshold: 0.50",
        f"- Precision: {overall['precision']:.3f}",
        f"- Recall: {overall['recall']:.3f}",
        f"- F1: {overall['f1']:.3f}",
        f"- Pooled AP50: {_format_metric(overall['pooled_ap50'])}",
        f"- Class-mean AP50: {_format_metric(overall['mean_ap50'])}",
        f"- False positives per image: {overall['false_positives_per_image']:.3f}",
        f"- Median matched IoU: {overall['median_matched_iou']:.3f}",
        f"- Median inference time: {overall['median_inference_time_seconds']:.3f} seconds",
        "",
        "Inference times exclude model loading and rendering, include CPU output conversion, "
        "and were collected with concurrent development/verification activity. "
        "They describe this execution and are not an isolated hardware performance benchmark.",
        "",
    ]
    lines.extend(_failure_section(aggregate_record))
    lines.extend(_breakdown_section(aggregate_record))
    lines.extend(_plots_section())
    lines.extend(_gallery_section(gallery))
    return "\n".join(lines).rstrip() + "\n"


def _failure_section(aggregate_record: Record) -> list[str]:
    pairs = sorted(
        aggregate_record["paired_conditions"],
        key=lambda row: (-row["new_misses"], -row["false_positive_delta"], row["condition"]),
    )
    lines = [
        "## Failure patterns",
        "",
        "These counts are deterministic descriptions of this fixed batch. "
        "Each condition is paired with the baseline of the same asset, so baseline "
        "recognition difficulty is not confused with a condition-specific regression.",
        "",
    ]
    for row in pairs:
        if row["new_misses"] or row["recoveries"] or row["false_positive_delta"]:
            lines.append(
                f"- {row['condition']}: {row['new_misses']} new misses, "
                f"{row['recoveries']} recoveries, "
                f"{row['false_positive_delta']:+d} false positives versus "
                f"the corresponding {row['pairs']} baselines."
            )
    lines.extend(["", "Error categories at the fixed reporting threshold:", ""])
    lines.extend(f"- {key}: {count}" for key, count in aggregate_record["failure_patterns"].items())
    lines.extend(
        [
            "",
            "Duplicate false positives overlap an already matched instance at IoU >= 0.50. "
            "Localization errors have overlap but fail the matching rule. "
            "Below-threshold misses have a geometrically matching saved candidate below 0.35. "
            "No-saved-candidate misses are bounded by the 0.10 collection floor. "
            "Grounding DINO predictions are scored as returned, without study-specific NMS.",
            "",
        ]
    )
    return lines


def _plots_section() -> list[str]:
    return [
        "## Plots",
        "",
        "![Recall by class](recall_by_class.png)",
        "",
        "![Recall by condition](recall_by_condition.png)",
        "",
        "![Precision-recall curve](precision_recall.png)",
        "",
        "Detailed per-class, per-asset, and per-condition values are in `metrics.csv`. "
        "Machine-readable aggregate data is in `aggregate.json`.",
        "",
        "AP uses every saved detection score and 101-point interpolated precision at IoU 0.50. "
        "Predictions below the collection threshold of 0.10 are unavailable. "
        "Class-mean AP averages the four per-class AP values; pooled AP ranks all predictions "
        "together. Conditions share asset variants and are paired observations, "
        "so this report does not claim independent statistical trials.",
        "",
        "Ground-truth boxes enclose visible segmentation pixels. Under occlusion they are "
        "smaller than amodal whole-object boxes, which can penalize otherwise plausible "
        "whole-object detector predictions. Distance and focal changes may also cause image "
        "truncation; the recorded boxes and gallery expose that effect.",
        "",
    ]


def _breakdown_section(aggregate_record: Record) -> list[str]:
    groups = aggregate_record["groups"]
    class_rows = [row for row in groups if row["dimension"] == "target_class"]
    asset_rows = [row for row in groups if row["dimension"] == "asset"]
    condition_rows = [
        row
        for row in groups
        if row["dimension"] == "factor_level" and not row["value"].startswith("target_presence:")
    ]
    absent_rows = [
        row
        for row in groups
        if row["dimension"] == "factor_level" and row["value"].startswith("target_presence:")
    ]
    hardest_class = min(class_rows, key=lambda row: (row["recall"], row["value"]))
    hardest_condition = min(condition_rows, key=lambda row: (row["recall"], row["value"]))
    absent_false_positives = sum(row["false_positives"] for row in absent_rows)
    patterns = aggregate_record["failure_patterns"]
    lines = [
        "## Metric breakdowns",
        "",
        f"The lowest class recall was {hardest_class['value']} at "
        f"{hardest_class['recall']:.3f}. The lowest controlled-condition recall was "
        f"{hardest_condition['value']} at {hardest_condition['recall']:.3f}. "
        f"The 12 target-absent controls produced {absent_false_positives} false positives.",
        "",
    ]
    if patterns.get("below_threshold_fn") == aggregate_record["overall"]["false_negatives"]:
        lines.extend(
            [
                "Every miss at confidence 0.35 had a geometrically valid saved candidate below "
                "that threshold. In this batch, the fixed-threshold misses were confidence "
                "failures rather than complete proposal failures above the 0.10 collection floor.",
                "",
            ]
        )
    lines.extend(["### By class", "", *_metric_table(class_rows), ""])
    lines.extend(["### By asset variant", "", *_metric_table(asset_rows), ""])
    lines.extend(
        [
            "### By controlled condition",
            "",
            *_metric_table(sorted(condition_rows, key=lambda row: (row["recall"], row["value"]))),
            "",
        ]
    )
    return lines


def _metric_table(rows: list[Record]) -> list[str]:
    lines = [
        "| Group | Scenes | Precision | Recall | F1 | AP50 | FP/image |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['value']} | {row['scene_count']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {_format_metric(row['ap50'])} | "
            f"{row['false_positives_per_image']:.3f} |"
        )
    return lines


def _gallery_section(gallery: list[dict[str, str]]) -> list[str]:
    lines = ["## Failure gallery", ""]
    if not gallery:
        return lines + ["No false positives or false negatives were found.", ""]
    for item in gallery:
        lines.extend(
            [
                f"### {item['scene_id']}",
                "",
                item["summary"],
                "",
                f"![{item['scene_id']}]({item['image']})",
                "",
            ]
        )
    return lines


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
