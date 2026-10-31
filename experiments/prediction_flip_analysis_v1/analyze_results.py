"""Offline paired Zero-shot → Few-shot analysis for the frozen formal runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path

from experiments.formal_analysis_v1.analyze_results import (
    MODELS,
    ROOT,
    RUNS,
    AnalysisValidationError,
    read_csv,
    validate_metadata,
    validate_predictions,
)
from llm_experiment.dataset import load_frozen_dataset

OUTPUT = Path(__file__).resolve().parent
STATES = ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")
SUMMARY_FIELDS = (
    "model",
    "run_id",
    *STATES,
    "zero_shot_accuracy",
    "few_shot_accuracy",
    "accuracy_delta",
)
DETAIL_FIELDS = (
    "model",
    "run_id",
    "sample_id",
    "gt",
    "primary_distractor",
    "directed_boundary",
    "zero_shot_prediction",
    "few_shot_prediction",
    "flip_state",
)


def pair_predictions(
    predictions: dict[tuple[str, str, int], dict[str, dict[str, str]]],
    labels: dict[str, str],
    distractors: dict[str, str],
) -> list[dict[str, str]]:
    details = []
    for model, run in product(MODELS, RUNS):
        zero = predictions[model, "zero_shot", run]
        few = predictions[model, "few_shot", run]
        for sample_id in sorted(labels):
            gt = labels[sample_id]
            zero_prediction = zero[sample_id]["prediction"]
            few_prediction = few[sample_id]["prediction"]
            first = "correct" if zero_prediction == gt else "wrong"
            second = "correct" if few_prediction == gt else "wrong"
            details.append(
                {
                    "model": model,
                    "run_id": str(run),
                    "sample_id": sample_id,
                    "gt": gt,
                    "primary_distractor": distractors[sample_id],
                    "directed_boundary": f"{gt}→{distractors[sample_id]}",
                    "zero_shot_prediction": zero_prediction,
                    "few_shot_prediction": few_prediction,
                    "flip_state": f"{first}_to_{second}",
                }
            )
    if (
        len(details) != 600
        or len({(r["model"], r["run_id"], r["sample_id"]) for r in details}) != 600
    ):
        raise AnalysisValidationError("Expected 600 unique paired predictions")
    return details


def summarize(details: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    summary = []
    for model in MODELS:
        for run_id in ("1", "2", "combined"):
            group = [
                row
                for row in details
                if row["model"] == model and (run_id == "combined" or row["run_id"] == run_id)
            ]
            expected = 120 if run_id == "combined" else 60
            if len(group) != expected:
                raise AnalysisValidationError(f"Expected {expected} pairs for {model} run {run_id}")
            counts = Counter(row["flip_state"] for row in group)
            zero_correct = counts["correct_to_correct"] + counts["correct_to_wrong"]
            few_correct = counts["correct_to_correct"] + counts["wrong_to_correct"]
            summary.append(
                {
                    "model": model,
                    "run_id": run_id,
                    **{state: counts[state] for state in STATES},
                    "zero_shot_accuracy": zero_correct / expected,
                    "few_shot_accuracy": few_correct / expected,
                    "accuracy_delta": (counts["wrong_to_correct"] - counts["correct_to_wrong"])
                    / expected,
                }
            )
    return summary


def verify_formal_delta(
    summary: list[dict[str, str | int | float]],
    overall: list[dict[str, str]],
) -> None:
    lookup = {(row["model"], row["prompt_type"]): row for row in overall}
    if len(overall) != 10 or len(lookup) != 10:
        raise AnalysisValidationError("Formal Analysis V1 overall table must have ten unique rows")
    for row in summary:
        model = str(row["model"])
        if (model, "zero_shot") not in lookup or (model, "few_shot") not in lookup:
            raise AnalysisValidationError(f"Missing Formal Analysis V1 baseline for {model}")
        run_id = str(row["run_id"])
        field = "mean_accuracy" if run_id == "combined" else f"run{run_id}_accuracy"
        zero = float(lookup[model, "zero_shot"][field])
        few = float(lookup[model, "few_shot"][field])
        if not (
            abs(float(row["zero_shot_accuracy"]) - zero) < 1e-12
            and abs(float(row["few_shot_accuracy"]) - few) < 1e-12
            and abs(float(row["accuracy_delta"]) - (few - zero)) < 1e-12
        ):
            raise AnalysisValidationError(
                f"Flip delta disagrees with Formal Analysis V1: {model}/{run_id}"
            )


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    summary: list[dict[str, str | int | float]],
    details: list[dict[str, str]],
) -> str:
    lines = [
        "# Prediction Flip Analysis V1",
        "",
        "20 formal configurations pair into 600 model × run × Test-sample Zero-shot/Few-shot "
        "comparisons. All input validations passed; per-run and combined accuracy deltas "
        "match Formal Analysis V1.",
        "",
        "| Model | Run | C→C | C→W | W→C | W→W | Accuracy Δ (Few−Zero) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['run_id']} | {row['correct_to_correct']} | "
            f"{row['correct_to_wrong']} | {row['wrong_to_correct']} | "
            f"{row['wrong_to_wrong']} | {float(row['accuracy_delta']):+.3f} |"
        )
    lines += [
        "",
        "## Correct → wrong cases",
        "",
        "Counts below include both runs. Predicted destination is the Few-shot prediction. "
        "A directed boundary denotes the item's GT → primary distractor, not necessarily "
        "the predicted destination.",
        "",
    ]
    for model in MODELS:
        losses = [
            row
            for row in details
            if row["model"] == model and row["flip_state"] == "correct_to_wrong"
        ]
        gt_counts = Counter(row["gt"] for row in losses)
        distractor_counts = Counter(row["primary_distractor"] for row in losses)
        destination_counts = Counter(row["few_shot_prediction"] for row in losses)
        boundary_counts = Counter(row["directed_boundary"] for row in losses)
        lines += [
            f"### {model} ({len(losses)} cases)",
            "",
            f"- GT: {format_counts(gt_counts)}",
            f"- Primary distractor: {format_counts(distractor_counts)}",
            f"- Wrong prediction destination: {format_counts(destination_counts)}",
            f"- Directed boundary: {format_counts(boundary_counts)}",
            "",
        ]
    lines += [
        "These are descriptive prediction changes on fixed Test items; they do not by "
        "themselves establish a mechanism for the prompt effect.",
        "",
    ]
    return "\n".join(lines)


def format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def main() -> None:
    dataset_path = ROOT / "dataset_v2.csv"
    load_frozen_dataset(dataset_path)
    dataset = read_csv(dataset_path)
    v4_path = ROOT / "experiments/challenge_audit_v1/v4/candidates_full_v4.jsonl"
    v4 = [json.loads(line) for line in v4_path.read_text(encoding="utf-8").splitlines()]
    labels, distractors, _ = validate_metadata(dataset, v4)
    predictions = validate_predictions(ROOT / "results", labels)
    details = pair_predictions(predictions, labels, distractors)
    summary = summarize(details)
    overall = read_csv(ROOT / "experiments/formal_analysis_v1/artifacts/overall_results.csv")
    verify_formal_delta(summary, overall)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "flip_summary.csv", SUMMARY_FIELDS, summary)
    write_csv(OUTPUT / "flip_details.csv", DETAIL_FIELDS, details)
    (OUTPUT / "report.md").write_text(make_report(summary, details), encoding="utf-8")
    print(f"Prediction Flip Analysis V1 complete: {OUTPUT}")


if __name__ == "__main__":
    main()
