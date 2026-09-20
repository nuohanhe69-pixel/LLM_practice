from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

from llm_experiment.api_client import ProviderRequestError
from llm_experiment.constants import ALLOWED_LABELS, API_STATUS_FAILURE, INVALID_OUTPUT
from llm_experiment.difficulty_pilot import (
    PILOT_API_MODELS,
    PILOT_GROUND_TRUTH_FIELDS,
    PILOT_INPUT_FIELDS,
    PILOT_PREDICTION_FIELDS,
    PilotDataError,
    PilotPrediction,
    audit_pilot_predictions,
    load_pilot_input,
    load_pilot_predictions,
    run_pilot_stage_a,
    run_pilot_stage_b,
)


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def write_pilot_input(path: Path, *, extra_field: bool = False) -> Path:
    fields = (*PILOT_INPUT_FIELDS, "label") if extra_field else PILOT_INPUT_FIELDS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(24):
            row = {"id": f"P{index + 1:03d}", "error_text": f"pilot error {index + 1}"}
            if extra_field:
                row["label"] = "CODE_RUNTIME"
            writer.writerow(row)
    return path


def write_model_config(path: Path, model_keys=("qwen_test",)) -> Path:
    models = {}
    for key in model_keys:
        models[key] = {
            "api_model": key.replace("_", "-"),
            "base_url": "https://example.invalid/v1",
            "api_key_env": "TEST_API_KEY",
            "temperature": 0,
            "max_tokens": 16,
            "timeout_seconds": 5,
            "max_retries": 2,
        }
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


def write_ground_truth(path: Path) -> dict[str, str]:
    labels = sorted(ALLOWED_LABELS)
    truth: dict[str, str] = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_GROUND_TRUTH_FIELDS)
        writer.writeheader()
        for index in range(24):
            sample_id = f"P{index + 1:03d}"
            label = labels[index % len(labels)]
            truth[sample_id] = label
            writer.writerow(
                {
                    "id": sample_id,
                    "candidate_id": f"candidate-{index + 1}",
                    "label": label,
                    "difficulty": "Medium" if index % 2 == 0 else "Hard",
                    "difficulty_score": "6",
                    "title": f"title {index + 1}",
                    "candidate_source_plan": "fixture",
                    "source_file": "fixture.md",
                    "split": "pilot",
                }
            )
    return truth


def write_predictions(path: Path, records: list[PilotPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(record.to_csv_row() for record in records)


def make_prediction(
    sample_id: str,
    model: str,
    normalized_output: str,
    *,
    status: str = "SUCCESS",
) -> PilotPrediction:
    if status == API_STATUS_FAILURE:
        raw_output = ""
        normalized_output = ""
        error_type = "ProviderRequestError: timed out"
    else:
        raw_output = normalized_output if normalized_output != INVALID_OUTPUT else "not-a-label"
        error_type = ""
    return PilotPrediction(
        id=sample_id,
        model=model,
        run=1,
        prompt_type="zero_shot",
        raw_output=raw_output,
        normalized_output=normalized_output,
        status=status,
        error_type=error_type,
        latency_ms=10.0,
        actual_temperature=0.0,
        timestamp="2026-09-21T00:00:00+00:00",
    )


def test_stage_a_loader_rejects_any_ground_truth_column(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv", extra_field=True)

    with pytest.raises(PilotDataError, match="columns must be exactly"):
        load_pilot_input(input_path)


def test_stage_a_uses_frozen_zero_shot_prompt_and_writes_one_by_one(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    config_path = write_model_config(tmp_path / "models.json")
    client = QueueClient([" code_runtime \n", "NETWORK_API"])

    audit = run_pilot_stage_a(
        input_path=input_path,
        predictions_path=tmp_path / "predictions.csv",
        model_config_path=config_path,
        prompt_dir="prompts",
        model_keys=("qwen_test",),
        sample_limit=2,
        client_factory=lambda _config: client,
    )

    assert audit.expected_combinations == 2
    assert audit.success == 2
    records = load_pilot_predictions(tmp_path / "predictions.csv")
    assert [record.normalized_output for record in records] == ["CODE_RUNTIME", "NETWORK_API"]
    assert all("只输出一个标签，不要解释" in prompt for prompt in client.prompts)
    assert all(
        "candidate_id" not in prompt and "difficulty" not in prompt for prompt in client.prompts
    )


def test_stage_a_resume_only_retries_api_failure(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    config_path = write_model_config(tmp_path / "models.json")
    predictions_path = tmp_path / "predictions.csv"
    first_client = QueueClient(
        ["not-a-label", ProviderRequestError("TimeoutError: temporary outage")]
    )
    run_pilot_stage_a(
        input_path=input_path,
        predictions_path=predictions_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        model_keys=("qwen_test",),
        sample_limit=2,
        client_factory=lambda _config: first_client,
    )
    first_records = load_pilot_predictions(predictions_path)
    assert [record.status for record in first_records] == [INVALID_OUTPUT, API_STATUS_FAILURE]

    resumed_client = QueueClient(["ENV_DEPENDENCY"])
    audit = run_pilot_stage_a(
        input_path=input_path,
        predictions_path=predictions_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        model_keys=("qwen_test",),
        sample_limit=2,
        client_factory=lambda _config: resumed_client,
    )

    assert len(resumed_client.prompts) == 1
    assert audit.invalid_output == 1
    assert audit.success == 1
    final_records = load_pilot_predictions(predictions_path)
    assert final_records[0] == first_records[0]
    assert final_records[1].normalized_output == "ENV_DEPENDENCY"


def test_audit_reports_missing_duplicate_and_status_counts():
    records = [
        make_prediction("P001", "model-a", "CODE_RUNTIME"),
        make_prediction("P001", "model-a", "CODE_RUNTIME"),
        make_prediction("P001", "model-b", INVALID_OUTPUT, status=INVALID_OUTPUT),
    ]

    audit = audit_pilot_predictions(
        records,
        sample_ids=("P001", "P002"),
        api_models=("model-a", "model-b"),
    )

    assert audit.expected_combinations == 4
    assert audit.observed_rows == 3
    assert audit.missing_combinations == ("P002/model-a", "P002/model-b")
    assert audit.duplicate_combinations == ("P001/model-a",)
    assert audit.success == 2
    assert audit.invalid_output == 1


def test_stage_b_generates_matrix_summary_and_report_after_complete_stage_a(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    ground_truth_path = tmp_path / "ground_truth.csv"
    truth = write_ground_truth(ground_truth_path)
    records: list[PilotPrediction] = []
    for sample_id, label in truth.items():
        for model in PILOT_API_MODELS:
            records.append(make_prediction(sample_id, model, label))
    records[1] = make_prediction("P001", PILOT_API_MODELS[1], "NETWORK_API")
    records[5] = make_prediction("P002", PILOT_API_MODELS[2], INVALID_OUTPUT, status=INVALID_OUTPUT)
    records[6] = make_prediction("P003", PILOT_API_MODELS[0], "", status=API_STATUS_FAILURE)
    predictions_path = tmp_path / "pilot_predictions_v1.csv"
    write_predictions(predictions_path, records)
    smoke_predictions_path = tmp_path / "smoke_predictions.csv"
    write_predictions(
        smoke_predictions_path,
        [make_prediction("P001", PILOT_API_MODELS[0], truth["P001"])],
    )

    matrix_path = tmp_path / "pilot_item_matrix_v1.csv"
    summary_path = tmp_path / "pilot_summary_v1.csv"
    report_path = tmp_path / "pilot_execution_report_v1.md"
    audit = run_pilot_stage_b(
        input_path=input_path,
        predictions_path=predictions_path,
        ground_truth_path=ground_truth_path,
        item_matrix_path=matrix_path,
        summary_path=summary_path,
        report_path=report_path,
        smoke_predictions_path=smoke_predictions_path,
    )

    assert audit.expected_combinations == 72
    assert audit.observed_rows == 72
    assert audit.success == 70
    assert audit.invalid_output == 1
    assert audit.api_failure == 1
    with matrix_path.open(encoding="utf-8", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    assert len(matrix) == 24
    assert matrix[0]["model_disagreement"] == "true"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    by_scope = defaultdict(list)
    for row in summary:
        by_scope[row["scope"]].append(row)
    assert len(by_scope["overall"]) == 1
    assert len(by_scope["model"]) == 3
    assert len(by_scope["difficulty"]) == 2
    assert len(by_scope["model_difficulty"]) == 6
    report = report_path.read_text(encoding="utf-8")
    assert "Smoke Test: COMPLETE (1/1)" in report
    assert "Stage A: COMPLETE" in report
    assert "Missing combinations: 0" in report
    assert "API_FAILURE: 1" in report
    assert "empty raw_output: 0" in report


def test_stage_b_refuses_to_read_incomplete_predictions(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    predictions_path = tmp_path / "pilot_predictions_v1.csv"
    write_predictions(
        predictions_path,
        [make_prediction("P001", PILOT_API_MODELS[0], "CODE_RUNTIME")],
    )

    with pytest.raises(PilotDataError, match="Missing Pilot combinations"):
        run_pilot_stage_b(
            input_path=input_path,
            predictions_path=predictions_path,
            ground_truth_path=tmp_path / "must-not-be-read.csv",
            item_matrix_path=tmp_path / "matrix.csv",
            summary_path=tmp_path / "summary.csv",
            report_path=tmp_path / "report.md",
        )
