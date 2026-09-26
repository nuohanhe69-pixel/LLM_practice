from __future__ import annotations

import csv
from collections import deque

import pytest

from llm_experiment.api_client import CompletionResult, ProviderFatalError, ProviderRequestError
from llm_experiment.config import ModelConfig
from llm_experiment.constants import API_STATUS_FAILURE, API_STATUS_SUCCESS, INVALID_OUTPUT
from llm_experiment.dataset import load_dataset
from llm_experiment.prediction import (
    PREDICTION_FIELDS,
    PredictionStateError,
    load_prediction_records,
    normalize_prediction,
    run_predictions,
)
from tests.helpers import write_protocol_dataset


class SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = deque(outcomes)
        self.prompts: list[str] = []

    def complete_with_metadata(self, prompt: str) -> CompletionResult:
        self.prompts.append(prompt)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, CompletionResult):
            return outcome
        return CompletionResult(outcome, None, None, None, None)


def model_config() -> ModelConfig:
    return ModelConfig(
        name="qwen_test",
        api_model="qwen-test",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        temperature=0.0,
        max_tokens=8,
        max_completion_tokens=None,
        timeout_seconds=5.0,
        max_retries=2,
    )


def test_normalize_prediction_only_strips_whitespace_newlines_and_changes_case():
    assert normalize_prediction("  network_api\n") == "NETWORK_API"
    assert normalize_prediction("CONTEXT_\nLIMIT") == "CONTEXT_LIMIT"
    assert normalize_prediction("`NETWORK_API`") == INVALID_OUTPUT
    assert normalize_prediction("The answer is NETWORK_API") == INVALID_OUTPUT
    assert normalize_prediction("NETWORK API") == INVALID_OUTPUT


def test_run_predictions_distinguishes_invalid_output_from_api_failure(tmp_path):
    bundle = load_dataset(write_protocol_dataset(tmp_path / "dataset_v1.csv"))
    predictions_path = tmp_path / "predictions.csv"
    client = SequenceClient(
        [ProviderRequestError("TimeoutError: timed out"), "network_api\n", "not-a-label"]
    )

    records = run_predictions(
        samples=bundle.tests[:3],
        demos=bundle.demos,
        model_config=model_config(),
        prompt_type="zero_shot",
        run_id=1,
        predictions_path=predictions_path,
        prompt_dir="prompts",
        client=client,
    )

    assert [record.api_status for record in records] == [
        API_STATUS_FAILURE,
        API_STATUS_SUCCESS,
        API_STATUS_SUCCESS,
    ]
    assert [record.prediction for record in records] == ["", "NETWORK_API", INVALID_OUTPUT]
    assert records[0].raw_output == ""
    assert records[0].api_error == "ProviderRequestError: TimeoutError: timed out"
    assert records[1].raw_output == "network_api\n"
    assert all(record.latency_seconds >= 0 for record in records)

    with predictions_path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in saved] == ["T001", "T002", "T003"]
    assert saved[0]["api_status"] == API_STATUS_FAILURE
    assert all(
        saved[0][field] == ""
        for field in (
            "finish_reason",
            "completion_tokens",
            "reasoning_tokens",
        )
    )
    assert saved[2]["prediction"] == INVALID_OUTPUT
    assert load_prediction_records(predictions_path) == records


def test_resume_skips_successful_rows_without_overwriting_them(tmp_path):
    bundle = load_dataset(write_protocol_dataset(tmp_path / "dataset_v1.csv"))
    predictions_path = tmp_path / "predictions.csv"
    first_client = SequenceClient([CompletionResult("NETWORK_API", "private", "stop", 12, 8)])
    run_predictions(
        samples=bundle.tests[:1],
        demos=bundle.demos,
        model_config=model_config(),
        prompt_type="few_shot",
        run_id=1,
        predictions_path=predictions_path,
        prompt_dir="prompts",
        client=first_client,
    )
    original_bytes = predictions_path.read_bytes()

    second_client = SequenceClient(["CODE_RUNTIME"])
    records = run_predictions(
        samples=bundle.tests[:2],
        demos=bundle.demos,
        model_config=model_config(),
        prompt_type="few_shot",
        run_id=1,
        predictions_path=predictions_path,
        prompt_dir="prompts",
        client=second_client,
    )

    assert len(second_client.prompts) == 1
    assert [record.sample_id for record in records] == ["T001", "T002"]
    assert records[0].raw_output == "NETWORK_API"
    assert (
        records[0].finish_reason,
        records[0].completion_tokens,
        records[0].reasoning_tokens,
    ) == (
        "stop",
        12,
        8,
    )
    assert original_bytes in predictions_path.read_bytes()


def test_resume_retries_api_failure_and_replaces_it_without_duplicate_ids(tmp_path):
    bundle = load_dataset(write_protocol_dataset(tmp_path / "dataset_v1.csv"))
    predictions_path = tmp_path / "predictions.csv"
    first_client = SequenceClient(
        [ProviderRequestError("TimeoutError: temporary outage"), "not-a-label", "NETWORK_API"]
    )
    first_records = run_predictions(
        samples=bundle.tests[:3],
        demos=bundle.demos,
        model_config=model_config(),
        prompt_type="zero_shot",
        run_id=1,
        predictions_path=predictions_path,
        prompt_dir="prompts",
        client=first_client,
    )
    assert [record.api_status for record in first_records] == [
        API_STATUS_FAILURE,
        API_STATUS_SUCCESS,
        API_STATUS_SUCCESS,
    ]
    assert first_records[1].prediction == INVALID_OUTPUT

    resumed_client = SequenceClient([CompletionResult("CODE_RUNTIME", "private", "stop", 7, 3)])
    final_records = run_predictions(
        samples=bundle.tests[:3],
        demos=bundle.demos,
        model_config=model_config(),
        prompt_type="zero_shot",
        run_id=1,
        predictions_path=predictions_path,
        prompt_dir="prompts",
        client=resumed_client,
    )

    assert len(resumed_client.prompts) == 1
    assert [record.sample_id for record in final_records] == ["T001", "T002", "T003"]
    assert final_records[0].api_status == API_STATUS_SUCCESS
    assert final_records[0].prediction == "CODE_RUNTIME"
    assert final_records[0].api_error == ""
    assert final_records[0].completion_tokens == 7
    assert final_records[0].reasoning_tokens == 3
    assert final_records[1].prediction == INVALID_OUTPUT
    assert final_records[2].prediction == "NETWORK_API"

    with predictions_path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in saved] == ["T001", "T002", "T003"]
    assert sum(row["sample_id"] == "T001" for row in saved) == 1


@pytest.mark.parametrize(
    "content,finish,completion,reasoning,expected",
    [
        ("NETWORK_API", "stop", 12, 8, "NETWORK_API"),
        ("CONTEXT_LIMIT", "stop", 20, 16, "CONTEXT_LIMIT"),
        ("", "length", 19, 16, INVALID_OUTPUT),
        ("not-a-label", "length", 21, 16, INVALID_OUTPUT),
        ("NETWORK_API", None, None, None, "NETWORK_API"),
        ("NETWORK_API", "stop", 0, 0, "NETWORK_API"),
    ],
)
def test_metadata_roundtrip_uses_only_visible_content(
    tmp_path,
    content,
    finish,
    completion,
    reasoning,
    expected,
):
    bundle = load_dataset(write_protocol_dataset(tmp_path / "dataset.csv"))
    path = tmp_path / "predictions.csv"
    records = run_predictions(
        samples=bundle.tests[:1],
        demos=bundle.demos,
        model_config=model_config(),
        prompt_type="zero_shot",
        run_id=9002,
        predictions_path=path,
        prompt_dir="prompts",
        client=SequenceClient(
            [
                CompletionResult(
                    content, "private reasoning: ENV_DEPENDENCY", finish, completion, reasoning
                ),
            ]
        ),
    )
    record = records[0]
    assert record.raw_output == content
    assert record.prediction == expected
    assert record.api_status == API_STATUS_SUCCESS
    assert record.finish_reason == (finish or "")
    assert record.completion_tokens == completion
    assert record.reasoning_tokens == reasoning
    assert load_prediction_records(path) == records
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == PREDICTION_FIELDS
        row = next(reader)
    assert row["completion_tokens"] == ("" if completion is None else str(completion))
    assert row["reasoning_tokens"] == ("" if reasoning is None else str(reasoning))
    assert "private reasoning" not in path.read_text()
    assert "reasoning_content" not in row


@pytest.mark.parametrize("token_value", ["bad", "-1", "1.5"])
def test_checkpoint_rejects_invalid_token_counts(tmp_path, token_value):
    path = tmp_path / "predictions.csv"
    row = {
        "sample_id": "T001",
        "model": "qwen_test",
        "api_model": "qwen-test",
        "prompt_type": "zero_shot",
        "run_id": 1,
        "temperature": 0,
        "raw_output": "NETWORK_API",
        "prediction": "NETWORK_API",
        "api_status": API_STATUS_SUCCESS,
        "api_error": "",
        "latency": 0.1,
        "finish_reason": "stop",
        "completion_tokens": token_value,
        "reasoning_tokens": "",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(PredictionStateError, match="Invalid prediction checkpoint row"):
        load_prediction_records(path)


def test_run_predictions_propagates_unexpected_program_error_without_checkpoint(tmp_path):
    bundle = load_dataset(write_protocol_dataset(tmp_path / "dataset_v1.csv"))
    predictions_path = tmp_path / "predictions.csv"
    client = SequenceClient([RuntimeError("program bug")])

    with pytest.raises(RuntimeError, match="program bug"):
        run_predictions(
            samples=bundle.tests[:1],
            demos=bundle.demos,
            model_config=model_config(),
            prompt_type="zero_shot",
            run_id=1,
            predictions_path=predictions_path,
            prompt_dir="prompts",
            client=client,
        )

    assert not predictions_path.exists()


@pytest.mark.parametrize("status_code", [400, 401, 404])
def test_run_predictions_stops_on_fatal_provider_error(status_code, tmp_path):
    bundle = load_dataset(write_protocol_dataset(tmp_path / "dataset_v1.csv"))
    predictions_path = tmp_path / "predictions.csv"
    client = SequenceClient([ProviderFatalError(f"HTTP {status_code}"), "NETWORK_API"])

    with pytest.raises(ProviderFatalError, match=f"HTTP {status_code}"):
        run_predictions(
            samples=bundle.tests[:2],
            demos=bundle.demos,
            model_config=model_config(),
            prompt_type="zero_shot",
            run_id=1,
            predictions_path=predictions_path,
            prompt_dir="prompts",
            client=client,
        )

    assert len(client.prompts) == 1
    assert not predictions_path.exists()
