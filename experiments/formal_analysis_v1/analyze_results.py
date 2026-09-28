"""Validate the frozen 20-run result matrix and produce Formal Analysis V1."""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
MODELS = (
    "qwen3_7_plus",
    "glm_5",
    "deepseek_v4_pro",
    "deepseek_v4_1_flash",
    "kimi_k3",
)
PROMPTS = ("zero_shot", "few_shot")
RUNS = (1, 2)
CLASSES = ("NETWORK_API", "CONTEXT_LIMIT", "ENV_DEPENDENCY", "CODE_RUNTIME")
TEST_IDS = {f"T{i:03d}" for i in range(1, 61)}
REQUIRED_PREDICTION_FIELDS = {
    "sample_id",
    "model",
    "api_model",
    "prompt_type",
    "run_id",
    "prediction",
    "api_status",
    "finish_reason",
}


class AnalysisValidationError(ValueError):
    """The formal inputs do not satisfy the frozen analysis preconditions."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_metadata(
    dataset_rows: list[dict[str, str]],
    v4_rows: list[dict[str, object]],
) -> tuple[dict[str, str], dict[str, str], tuple[tuple[str, str], ...]]:
    tests = [row for row in dataset_rows if row["split"] == "test"]
    labels = {row["id"]: row["label"] for row in tests}
    if len(tests) != 60 or set(labels) != TEST_IDS or len(labels) != 60:
        raise AnalysisValidationError("dataset_v2 Test IDs must be exactly T001–T060")
    if any(label not in CLASSES for label in labels.values()):
        raise AnalysisValidationError("Unknown dataset_v2 Test label")
    v4 = {str(row["id"]): row for row in v4_rows}
    if len(v4_rows) != 60 or set(v4) != TEST_IDS or len(v4) != 60:
        raise AnalysisValidationError("V4 IDs must be exactly T001–T060")
    if any(v4[item_id]["label"] != label for item_id, label in labels.items()):
        raise AnalysisValidationError("dataset_v2 and V4 labels disagree")
    distractors = {item_id: str(v4[item_id]["primary_distractor"]) for item_id in labels}
    if any(
        value not in CLASSES or value == labels[item_id] for item_id, value in distractors.items()
    ):
        raise AnalysisValidationError("Invalid primary distractor")
    boundaries = Counter((labels[item_id], distractors[item_id]) for item_id in labels)
    expected = set(product(CLASSES, repeat=2)) - {(label, label) for label in CLASSES}
    if set(boundaries) != expected or any(count != 5 for count in boundaries.values()):
        raise AnalysisValidationError("Expected 12 directed boundaries with five items each")
    ordered = tuple(
        (gt, distractor) for gt in CLASSES for distractor in CLASSES if gt != distractor
    )
    return labels, distractors, ordered


def validate_predictions(
    results_root: Path,
    labels: dict[str, str],
) -> dict[tuple[str, str, int], dict[str, dict[str, str]]]:
    expected = set(product(MODELS, PROMPTS, RUNS))
    actual = {
        (model_dir.name, prompt_dir.name, int(run_dir.name[4:]))
        for model_dir in results_root.iterdir()
        if model_dir.is_dir()
        for prompt_dir in model_dir.iterdir()
        if prompt_dir.is_dir()
        for run_dir in prompt_dir.iterdir()
        if run_dir.is_dir()
        and run_dir.name in {"run_1", "run_2"}
        and (run_dir / "predictions.csv").exists()
    }
    if actual != expected:
        raise AnalysisValidationError(
            f"Expected exactly 20 formal configurations; got {len(actual)}"
        )
    predictions = {}
    for model, prompt, run in product(MODELS, PROMPTS, RUNS):
        path = results_root / model / prompt / f"run_{run}" / "predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_PREDICTION_FIELDS.issubset(reader.fieldnames or ()):
                raise AnalysisValidationError(f"Missing prediction columns: {path}")
            rows = list(reader)
        ids = [row["sample_id"] for row in rows]
        if len(rows) != 60 or len(set(ids)) != 60 or set(ids) != set(labels):
            raise AnalysisValidationError(f"Expected 60 matching unique Test IDs: {path}")
        for row in rows:
            if (row["model"], row["prompt_type"], row["run_id"]) != (model, prompt, str(run)):
                raise AnalysisValidationError(f"Configuration metadata mismatch: {path}")
            if row["api_status"] != "SUCCESS":
                raise AnalysisValidationError(f"API_FAILURE or unknown API status: {path}")
            if row["finish_reason"] == "length":
                raise AnalysisValidationError(f"finish_reason=length: {path}")
            if row["prediction"] not in (*CLASSES, "INVALID_OUTPUT"):
                raise AnalysisValidationError(f"Unknown prediction label: {path}")
        predictions[model, prompt, run] = {row["sample_id"]: row for row in rows}
    if sum(len(rows) for rows in predictions.values()) != 1200:
        raise AnalysisValidationError("Expected exactly 1200 predictions")
    return predictions


def accuracy(ids: list[str], rows: dict[str, dict[str, str]], labels: dict[str, str]) -> float:
    return sum(rows[item_id]["prediction"] == labels[item_id] for item_id in ids) / len(ids)


def distractor_hit_rate(
    ids: list[str],
    rows: dict[str, dict[str, str]],
    labels: dict[str, str],
    distractors: dict[str, str],
) -> float | None:
    errors = [item_id for item_id in ids if rows[item_id]["prediction"] != labels[item_id]]
    if not errors:
        return None
    return sum(rows[item_id]["prediction"] == distractors[item_id] for item_id in errors) / len(
        errors
    )


def calculate(
    predictions: dict[tuple[str, str, int], dict[str, dict[str, str]]],
    labels: dict[str, str],
    distractors: dict[str, str],
    boundaries: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    all_ids = sorted(labels)
    overall = []
    per_class = []
    boundary = []
    prompt_stability = []
    matrices = {}
    for model, prompt in product(MODELS, PROMPTS):
        r1, r2 = (predictions[model, prompt, run] for run in RUNS)
        a1, a2 = (accuracy(all_ids, rows, labels) for rows in (r1, r2))
        overall.append(
            {
                "model": model,
                "prompt_type": prompt,
                "run1_accuracy": a1,
                "run2_accuracy": a2,
                "mean_accuracy": (a1 + a2) / 2,
                "run_gap": abs(a1 - a2),
                "invalid_outputs": sum(
                    row["prediction"] == "INVALID_OUTPUT"
                    for rows in (r1, r2)
                    for row in rows.values()
                ),
            }
        )
        for label in CLASSES:
            ids = [item_id for item_id in all_ids if labels[item_id] == label]
            c1, c2 = (accuracy(ids, rows, labels) for rows in (r1, r2))
            per_class.append(
                {
                    "model": model,
                    "prompt_type": prompt,
                    "gt_class": label,
                    "run1_accuracy": c1,
                    "run2_accuracy": c2,
                    "mean_accuracy": (c1 + c2) / 2,
                }
            )
        for gt, distractor in boundaries:
            ids = [
                item_id
                for item_id in all_ids
                if labels[item_id] == gt and distractors[item_id] == distractor
            ]
            b1, b2 = (accuracy(ids, rows, labels) for rows in (r1, r2))
            merged_rows = {
                f"{run}:{item_id}": predictions[model, prompt, run][item_id]
                for run in RUNS
                for item_id in ids
            }
            merged_labels = {key: gt for key in merged_rows}
            merged_distractors = {key: distractor for key in merged_rows}
            hit_rate = distractor_hit_rate(
                list(merged_rows),
                merged_rows,
                merged_labels,
                merged_distractors,
            )
            boundary.append(
                {
                    "model": model,
                    "prompt_type": prompt,
                    "gt_class": gt,
                    "primary_distractor": distractor,
                    "run1_boundary_accuracy": b1,
                    "run2_boundary_accuracy": b2,
                    "mean_boundary_accuracy": (b1 + b2) / 2,
                    "distractor_hit_rate_among_errors": hit_rate,
                }
            )
        matrices[model, prompt] = [
            {
                "gt_class": gt,
                **{
                    predicted: sum(
                        rows[item_id]["prediction"] == predicted
                        for rows in (r1, r2)
                        for item_id in all_ids
                        if labels[item_id] == gt
                    )
                    for predicted in CLASSES
                },
            }
            for gt in CLASSES
        ]
    for model in MODELS:
        zero = next(
            row for row in overall if row["model"] == model and row["prompt_type"] == "zero_shot"
        )
        few = next(
            row for row in overall if row["model"] == model and row["prompt_type"] == "few_shot"
        )
        prompt_stability.append(
            {
                "model": model,
                "zero_shot_mean_accuracy": zero["mean_accuracy"],
                "few_shot_mean_accuracy": few["mean_accuracy"],
                "few_shot_delta": few["mean_accuracy"] - zero["mean_accuracy"],
                "zero_shot_run_gap": zero["run_gap"],
                "few_shot_run_gap": few["run_gap"],
            }
        )
    return {
        "overall": overall,
        "per_class": per_class,
        "boundary": boundary,
        "prompt_stability": prompt_stability,
        "matrices": matrices,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_boundary_heatmaps(
    rows: list[dict[str, object]],
    boundaries: tuple[tuple[str, str], ...],
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{gt} → {distractor}" for gt, distractor in boundaries]
    for prompt in PROMPTS:
        lookup = {
            (str(row["model"]), str(row["gt_class"]), str(row["primary_distractor"])): float(
                row["mean_boundary_accuracy"]
            )
            for row in rows
            if row["prompt_type"] == prompt
        }
        matrix = [
            [lookup[model, gt, distractor] for gt, distractor in boundaries] for model in MODELS
        ]
        fig, ax = plt.subplots(figsize=(15, 4.3), layout="constrained")
        heat = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(MODELS)), MODELS)
        ax.set_title(
            f"{prompt.replace('_', ' ').title()} directed boundary accuracy (two-run mean)"
        )
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value >= 0.6 else "#172b4d",
                )
        fig.colorbar(heat, ax=ax, label="Boundary accuracy", shrink=0.8)
        for extension in ("pdf", "png"):
            fig.savefig(out / f"boundary_heatmap_{prompt}.{extension}", dpi=300)
        plt.close(fig)


def write_report(data: dict[str, object], output: Path) -> None:
    lines = [
        "# Formal Analysis V1",
        "",
        "Validated: 20 configurations × 60 Test items = 1200 predictions; "
        "API_FAILURE=0; finish_reason=length=0; 12 directed boundaries × 5 items.",
        "",
        "All accuracies are proportions. Two-run means are arithmetic means; "
        "run gaps are absolute differences. No model ranking or significance testing is implied.",
        "",
        "## Overall performance",
        "",
        "| Model | Prompt | Run 1 | Run 2 | Mean | Run gap | Invalid outputs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in data["overall"]:
        lines.append(
            f"| {row['model']} | {row['prompt_type']} | {row['run1_accuracy']:.3f} | "
            f"{row['run2_accuracy']:.3f} | {row['mean_accuracy']:.3f} | "
            f"{row['run_gap']:.3f} | {row['invalid_outputs']} |"
        )
    lines += [
        "",
        "## Four GT classes",
        "",
        "Two-run mean accuracy for each model × prompt × GT class; separate Run 1 and "
        "Run 2 values are in `artifacts/per_class_results.csv`.",
        "",
        "| Model | Prompt | NETWORK_API | CONTEXT_LIMIT | ENV_DEPENDENCY | CODE_RUNTIME |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, prompt in product(MODELS, PROMPTS):
        values = {
            row["gt_class"]: row["mean_accuracy"]
            for row in data["per_class"]
            if row["model"] == model and row["prompt_type"] == prompt
        }
        lines.append(
            f"| {model} | {prompt} | "
            + " | ".join(f"{values[label]:.3f}" for label in CLASSES)
            + " |"
        )
    lines += [
        "",
        "## Directed boundaries",
        "",
        "The table gives mean boundary accuracy across the ten model × prompt groups "
        "(both runs in each group). Lower values mean more errors on that directed boundary. "
        "Per-group Run 1, Run 2, mean and distractor hit rates are in "
        "`artifacts/boundary_results.csv`; blank hit rate means no errors.",
        "",
        "| GT → primary distractor | Mean boundary accuracy |",
        "|---|---:|",
    ]
    for gt in CLASSES:
        for distractor in CLASSES:
            if gt == distractor:
                continue
            values = [
                row["mean_boundary_accuracy"]
                for row in data["boundary"]
                if row["gt_class"] == gt and row["primary_distractor"] == distractor
            ]
            lines.append(f"| {gt} → {distractor} | {sum(values) / len(values):.3f} |")
    lines += [
        "",
        "## Prompt change and run stability",
        "",
        "| Model | Zero-shot mean | Few-shot mean | Few-shot delta | "
        "Zero-shot gap | Few-shot gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in data["prompt_stability"]:
        lines.append(
            f"| {row['model']} | {row['zero_shot_mean_accuracy']:.3f} | "
            f"{row['few_shot_mean_accuracy']:.3f} | {row['few_shot_delta']:+.3f} | "
            f"{row['zero_shot_run_gap']:.3f} | {row['few_shot_run_gap']:.3f} |"
        )
    lines += [
        "",
        "## Confusion matrices and heatmaps",
        "",
        "Ten 4×4 confusion matrices combine both runs and count only predictions "
        "in the four legal labels; INVALID_OUTPUT counts remain in overall_results.csv.",
        "The Zero-shot and Few-shot heatmaps show two-run mean boundary accuracy "
        "(five models × twelve directed boundaries).",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dataset = read_csv(ROOT / "dataset_v2.csv")
    v4_path = ROOT / "experiments/challenge_audit_v1/v4/candidates_full_v4.jsonl"
    v4_rows = [json.loads(line) for line in v4_path.read_text(encoding="utf-8").splitlines()]
    labels, distractors, boundaries = validate_metadata(dataset, v4_rows)
    predictions = validate_predictions(ROOT / "results", labels)
    data = calculate(predictions, labels, distractors, boundaries)
    artifacts = OUTPUT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("overall_results", data["overall"]),
        ("per_class_results", data["per_class"]),
        ("boundary_results", data["boundary"]),
        ("prompt_and_stability", data["prompt_stability"]),
    ):
        write_csv(artifacts / f"{name}.csv", rows)
    matrices = artifacts / "confusion_matrices"
    matrices.mkdir(exist_ok=True)
    for (model, prompt), rows in data["matrices"].items():
        write_csv(matrices / f"{model}_{prompt}.csv", rows)
    plot_boundary_heatmaps(data["boundary"], boundaries, artifacts)
    write_report(data, OUTPUT)
    print(f"Formal Analysis V1 complete: {OUTPUT}")


if __name__ == "__main__":
    main()
