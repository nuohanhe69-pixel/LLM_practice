import csv
from itertools import product

import pytest

from experiments.formal_analysis_v1.analyze_results import (
    CLASSES,
    MODELS,
    PROMPTS,
    RUNS,
    AnalysisValidationError,
    accuracy,
    calculate,
    distractor_hit_rate,
    validate_metadata,
    validate_predictions,
)


def synthetic_metadata():
    dataset = []
    v4 = []
    boundaries = [item for item in product(CLASSES, CLASSES) if item[0] != item[1]]
    for number, (gt, distractor) in enumerate(
        (item for item in boundaries for _ in range(5)), start=1
    ):
        item_id = f"T{number:03d}"
        dataset.append({"id": item_id, "label": gt, "split": "test"})
        v4.append({"id": item_id, "label": gt, "primary_distractor": distractor})
    return dataset, v4


def write_full_matrix(root, labels, distractors, *, wrong=None):
    wrong = wrong or {}
    for model, prompt, run in product(MODELS, PROMPTS, RUNS):
        folder = root / model / prompt / f"run_{run}"
        folder.mkdir(parents=True)
        with (folder / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "sample_id",
                "model",
                "api_model",
                "prompt_type",
                "run_id",
                "prediction",
                "api_status",
                "finish_reason",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item_id, label in labels.items():
                writer.writerow(
                    {
                        "sample_id": item_id,
                        "model": model,
                        "api_model": model,
                        "prompt_type": prompt,
                        "run_id": run,
                        "prediction": (
                            distractors[item_id]
                            if item_id in wrong.get((model, prompt, run), set())
                            else label
                        ),
                        "api_status": "SUCCESS",
                        "finish_reason": "stop",
                    }
                )


def test_twenty_configurations_1200_rows_and_metrics(tmp_path):
    dataset, v4 = synthetic_metadata()
    labels, distractors, boundaries = validate_metadata(dataset, v4)
    assert len(boundaries) == 12
    assert all(
        sum(labels[item_id] == gt and distractors[item_id] == wrong for item_id in labels) == 5
        for gt, wrong in boundaries
    )
    wrong = {
        ("qwen3_7_plus", "zero_shot", 1): {"T001"},
        ("qwen3_7_plus", "few_shot", 2): {"T001", "T002"},
    }
    write_full_matrix(tmp_path, labels, distractors, wrong=wrong)
    predictions = validate_predictions(tmp_path, labels)
    assert len(predictions) == 20
    assert sum(map(len, predictions.values())) == 1200
    data = calculate(predictions, labels, distractors, boundaries)
    zero = next(
        row
        for row in data["overall"]
        if row["model"] == "qwen3_7_plus" and row["prompt_type"] == "zero_shot"
    )
    assert zero["run1_accuracy"] == pytest.approx(59 / 60)
    assert zero["run2_accuracy"] == 1
    assert zero["mean_accuracy"] == pytest.approx(119 / 120)
    assert zero["run_gap"] == pytest.approx(1 / 60)
    assert zero["invalid_outputs"] == 0
    prompt = next(row for row in data["prompt_stability"] if row["model"] == "qwen3_7_plus")
    assert prompt["few_shot_delta"] == pytest.approx(-1 / 120)
    assert prompt["zero_shot_run_gap"] == pytest.approx(1 / 60)
    assert prompt["few_shot_run_gap"] == pytest.approx(2 / 60)
    assert len(data["per_class"]) == 40
    assert len(data["boundary"]) == 120
    assert len(data["matrices"]) == 10
    boundary = next(
        row
        for row in data["boundary"]
        if row["model"] == "qwen3_7_plus"
        and row["prompt_type"] == "zero_shot"
        and row["gt_class"] == labels["T001"]
        and row["primary_distractor"] == distractors["T001"]
    )
    assert boundary["run1_boundary_accuracy"] == 0.8
    assert boundary["run2_boundary_accuracy"] == 1
    assert boundary["mean_boundary_accuracy"] == 0.9
    assert boundary["distractor_hit_rate_among_errors"] == 1


def test_distractor_rate_counts_invalid_output_as_error():
    labels = {"a": "NETWORK_API", "b": "NETWORK_API", "c": "NETWORK_API"}
    distractors = {item_id: "CODE_RUNTIME" for item_id in labels}
    rows = {
        "a": {"prediction": "CODE_RUNTIME"},
        "b": {"prediction": "INVALID_OUTPUT"},
        "c": {"prediction": "NETWORK_API"},
    }
    assert accuracy(list(rows), rows, labels) == pytest.approx(1 / 3)
    assert distractor_hit_rate(list(rows), rows, labels, distractors) == 0.5
    assert distractor_hit_rate(["c"], rows, labels, distractors) is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failure", "length"])
def test_invalid_formal_results_fail_before_analysis(tmp_path, mutation):
    dataset, v4 = synthetic_metadata()
    labels, distractors, _ = validate_metadata(dataset, v4)
    write_full_matrix(tmp_path, labels, distractors)
    path = tmp_path / MODELS[0] / PROMPTS[0] / "run_1" / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0].copy()
    elif mutation == "failure":
        rows[0]["api_status"] = "API_FAILURE"
    elif mutation == "length":
        rows[0]["finish_reason"] = "length"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AnalysisValidationError):
        validate_predictions(tmp_path, labels)


def test_v4_label_and_boundary_mismatch_are_rejected():
    dataset, v4 = synthetic_metadata()
    v4[0]["label"] = "CONTEXT_LIMIT"
    with pytest.raises(AnalysisValidationError, match="labels disagree"):
        validate_metadata(dataset, v4)
    v4[0]["label"] = dataset[0]["label"]
    v4[0]["primary_distractor"] = "ENV_DEPENDENCY"
    with pytest.raises(AnalysisValidationError, match="12 directed boundaries"):
        validate_metadata(dataset, v4)
