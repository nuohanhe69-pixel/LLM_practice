from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_experiment.constants import API_STATUS_FAILURE, API_STATUS_SUCCESS, INVALID_OUTPUT
from llm_experiment.dataset import DatasetValidationError
from llm_experiment.evaluation import EvaluationError, evaluate_predictions
from llm_experiment.prediction import PREDICTION_FIELDS


def write_predictions(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prediction_row(
    sample_id,
    prediction,
    *,
    raw_output=None,
    api_status=API_STATUS_SUCCESS,
    api_error="",
):
    return {
        "sample_id": sample_id,
        "model": "qwen_test",
        "api_model": "qwen-test",
        "prompt_type": "zero_shot",
        "run_id": "1",
        "temperature": "0.0",
        "raw_output": prediction if raw_output is None else raw_output,
        "prediction": prediction,
        "api_status": api_status,
        "api_error": api_error,
        "latency": "0.1",
    }


def write_frozen_dataset_with_changed_label(path):
    with Path("dataset_v2.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows_by_id = {row["id"]: row for row in rows}
    rows_by_id["T001"]["label"] = "NETWORK_API"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_evaluation_rejects_label_change_after_predictions_exist(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    write_predictions(predictions_path, [prediction_row("T001", "CODE_RUNTIME")])
    dataset_path = write_frozen_dataset_with_changed_label(tmp_path / "dataset_v2.csv")

    with pytest.raises(DatasetValidationError, match="fingerprint"):
        evaluate_predictions(
            dataset_path=dataset_path,
            predictions_path=predictions_path,
            output_dir=tmp_path / "results",
            expected_sample_ids=["T001"],
        )


def test_evaluation_accepts_frozen_dataset_and_generates_metrics(tmp_path):
    dataset_path = Path("dataset_v2.csv")
    predictions_path = tmp_path / "predictions.csv"
    rows = [
        prediction_row("T001", "CODE_RUNTIME"),
        prediction_row("T002", "CODE_RUNTIME"),
        prediction_row("T003", INVALID_OUTPUT, raw_output="maybe environment"),
        prediction_row(
            "T004",
            "",
            raw_output="",
            api_status=API_STATUS_FAILURE,
            api_error="TimeoutError: timed out",
        ),
    ]
    write_predictions(predictions_path, rows)
    output_dir = tmp_path / "results"

    metrics = evaluate_predictions(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
        expected_sample_ids=["T001", "T002", "T003", "T004"],
    )

    assert metrics["total_samples"] == 4
    assert metrics["successful_predictions"] == 3
    assert metrics["invalid_outputs"] == 1
    assert metrics["api_failures"] == 1
    assert metrics["correct_predictions"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["CODE_RUNTIME"] == {
        "total_samples": 1,
        "successful_predictions": 1,
        "correct_predictions": 1,
        "accuracy": 1.0,
    }
    assert metrics["per_class"]["ENV_DEPENDENCY"]["accuracy"] == 0.0

    saved_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics == metrics
    with (output_dir / "error_cases.csv").open(encoding="utf-8", newline="") as handle:
        errors = list(csv.DictReader(handle))
    assert [row["error_type"] for row in errors] == [
        "CLASSIFICATION_ERROR",
        INVALID_OUTPUT,
        API_STATUS_FAILURE,
    ]
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = {row["id"]: row for row in csv.DictReader(handle)}
    assert errors[0]["error_text"] == source_rows["T001"]["error_text"]
    assert errors[1]["raw_output"] == "maybe environment"
    assert errors[2]["api_error"] == "TimeoutError: timed out"


def test_evaluation_requires_exactly_one_prediction_for_each_expected_sample(tmp_path):
    dataset_path = Path("dataset_v2.csv")
    predictions_path = tmp_path / "predictions.csv"
    write_predictions(predictions_path, [prediction_row("T001", "CODE_RUNTIME")])

    with pytest.raises(EvaluationError, match="Missing predictions: T002"):
        evaluate_predictions(
            dataset_path=dataset_path,
            predictions_path=predictions_path,
            output_dir=tmp_path / "results",
            expected_sample_ids=["T001", "T002"],
        )
