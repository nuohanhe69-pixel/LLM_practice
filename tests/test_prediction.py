from __future__ import annotations

import csv
from collections import deque

import pytest

from llm_experiment.api_client import ProviderFatalError, ProviderRequestError
from llm_experiment.config import ModelConfig
from llm_experiment.constants import API_STATUS_FAILURE, API_STATUS_SUCCESS, INVALID_OUTPUT
from llm_experiment.dataset import load_dataset
from llm_experiment.prediction import normalize_prediction, run_predictions
from tests.helpers import write_protocol_dataset


class SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = deque(outcomes)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def model_config() -> ModelConfig:
    return ModelConfig(
        name="qwen_test",
        api_model="qwen-test",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        temperature=0.0,
        max_tokens=8,
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
    assert saved[2]["prediction"] == INVALID_OUTPUT


def test_resume_skips_successful_rows_without_overwriting_them(tmp_path):
    bundle = load_dataset(write_protocol_dataset(tmp_path / "dataset_v1.csv"))
    predictions_path = tmp_path / "predictions.csv"
    first_client = SequenceClient(["NETWORK_API"])
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

    resumed_client = SequenceClient(["CODE_RUNTIME"])
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
    assert final_records[1].prediction == INVALID_OUTPUT
    assert final_records[2].prediction == "NETWORK_API"

    with predictions_path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in saved] == ["T001", "T002", "T003"]
    assert sum(row["sample_id"] == "T001" for row in saved) == 1


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
