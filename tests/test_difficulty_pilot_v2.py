from __future__ import annotations

import csv
from collections import Counter, defaultdict

import pytest

from llm_experiment.api_client import CompletionResult
from llm_experiment.constants import INVALID_OUTPUT
from llm_experiment.difficulty_pilot import PILOT_API_MODELS, PilotDataError
from llm_experiment.difficulty_pilot_v2 import (
    DEEPSEEK_API_MODEL,
    DEEPSEEK_MAX_COMPLETION_TOKENS,
    INHERITED_MODELS,
    load_pilot_v2_predictions,
    run_pilot_v2_smoke,
    run_pilot_v2_stage_a,
    run_pilot_v2_stage_b,
)
from tests.test_difficulty_pilot import (
    make_prediction,
    write_ground_truth,
    write_pilot_input,
    write_predictions,
)


class MetadataQueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete_with_metadata(self, prompt: str) -> CompletionResult:
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def metadata_result(
    content: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 80,
    reasoning_tokens: int = 70,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning_content="reasoning trace",
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def write_v2_model_config(path):
    path.write_text(
        """{
  "models": {
    "deepseek_v4_pro": {
      "api_model": "deepseek-v4-pro",
      "base_url": "https://example.invalid/v1",
      "api_key_env": "TEST_API_KEY",
      "temperature": 0.0,
      "max_completion_tokens": 2048,
      "timeout_seconds": 5.0,
      "max_retries": 2
    }
  }
}
""",
        encoding="utf-8",
    )
    return path


def write_v1_predictions(path, *, invalid_inherited: bool = False):
    rows = []
    for index in range(24):
        sample_id = f"P{index + 1:03d}"
        for model in PILOT_API_MODELS:
            is_invalid = model == DEEPSEEK_API_MODEL or (
                invalid_inherited and sample_id == "P001" and model == INHERITED_MODELS[0]
            )
            if is_invalid:
                rows.append(
                    make_prediction(sample_id, model, INVALID_OUTPUT, status=INVALID_OUTPUT)
                )
            else:
                rows.append(make_prediction(sample_id, model, "CODE_RUNTIME"))
    write_predictions(path, rows)
    return path


def test_v2_smoke_records_reasoning_metadata_and_validates_final_content(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    config_path = write_v2_model_config(tmp_path / "models.json")
    predictions_path = tmp_path / "pilot_predictions_v2.csv"
    client = MetadataQueueClient([metadata_result("CONTEXT_LIMIT")])

    record = run_pilot_v2_smoke(
        input_path=input_path,
        predictions_path=predictions_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        client_factory=lambda config: client,
    )

    assert record.status == "SUCCESS"
    assert record.normalized_output == "CONTEXT_LIMIT"
    assert record.finish_reason == "stop"
    assert record.completion_tokens == 80
    assert record.reasoning_tokens == 70
    assert record.reasoning_content_present is True
    assert record.reasoning_content_length == len("reasoning trace")
    assert record.max_completion_tokens == DEEPSEEK_MAX_COMPLETION_TOKENS
    assert len(client.prompts) == 1


def test_v2_smoke_stops_when_reasoning_exhausts_budget(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    config_path = write_v2_model_config(tmp_path / "models.json")
    client = MetadataQueueClient(
        [
            metadata_result(
                "",
                finish_reason="length",
                completion_tokens=2048,
                reasoning_tokens=2048,
            )
        ]
    )

    with pytest.raises(PilotDataError, match="smoke failed"):
        run_pilot_v2_smoke(
            input_path=input_path,
            predictions_path=tmp_path / "pilot_predictions_v2.csv",
            model_config_path=config_path,
            prompt_dir="prompts",
            client_factory=lambda config: client,
        )


def test_v2_stage_a_inherits_qwen_glm_and_only_calls_remaining_deepseek(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    config_path = write_v2_model_config(tmp_path / "models.json")
    v1_path = write_v1_predictions(tmp_path / "pilot_predictions_v1.csv")
    v2_path = tmp_path / "pilot_predictions_v2.csv"
    smoke_client = MetadataQueueClient([metadata_result("CODE_RUNTIME")])
    run_pilot_v2_smoke(
        input_path=input_path,
        predictions_path=v2_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        client_factory=lambda config: smoke_client,
    )
    stage_client = MetadataQueueClient([metadata_result("NETWORK_API") for _ in range(23)])
    factory_models: list[str] = []

    def client_factory(config):
        factory_models.append(config.api_model)
        return stage_client

    audit = run_pilot_v2_stage_a(
        input_path=input_path,
        v1_predictions_path=v1_path,
        v2_predictions_path=v2_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        client_factory=client_factory,
    )

    assert audit.expected_combinations == 72
    assert audit.observed_rows == 72
    assert audit.success == 72
    assert len(stage_client.prompts) == 23
    assert factory_models == [DEEPSEEK_API_MODEL]
    records = load_pilot_v2_predictions(v2_path)
    counts = Counter(record.model for record in records)
    assert counts == {model: 24 for model in PILOT_API_MODELS}
    inherited = [record for record in records if record.model in INHERITED_MODELS]
    assert all(record.source == "inherited_v1" for record in inherited)
    assert all(record.finish_reason is None for record in inherited)
    deepseek = [record for record in records if record.model == DEEPSEEK_API_MODEL]
    assert all(record.source == "deepseek_v2_api" for record in deepseek)
    assert all(record.reasoning_tokens == 70 for record in deepseek)


def test_v2_stage_a_rejects_non_success_v1_inheritance_before_api_call(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    config_path = write_v2_model_config(tmp_path / "models.json")
    v1_path = write_v1_predictions(
        tmp_path / "pilot_predictions_v1.csv",
        invalid_inherited=True,
    )

    with pytest.raises(PilotDataError, match="not SUCCESS"):
        run_pilot_v2_stage_a(
            input_path=input_path,
            v1_predictions_path=v1_path,
            v2_predictions_path=tmp_path / "pilot_predictions_v2.csv",
            model_config_path=config_path,
            prompt_dir="prompts",
            client_factory=lambda config: pytest.fail("API client must not be created"),
        )


def test_v2_stage_b_generates_all_required_summary_scopes(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    ground_truth_path = tmp_path / "ground_truth.csv"
    write_ground_truth(ground_truth_path)
    config_path = write_v2_model_config(tmp_path / "models.json")
    v1_path = write_v1_predictions(tmp_path / "pilot_predictions_v1.csv")
    v2_path = tmp_path / "pilot_predictions_v2.csv"
    smoke_client = MetadataQueueClient([metadata_result("CODE_RUNTIME")])
    run_pilot_v2_smoke(
        input_path=input_path,
        predictions_path=v2_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        client_factory=lambda config: smoke_client,
    )
    stage_client = MetadataQueueClient([metadata_result("CODE_RUNTIME") for _ in range(23)])
    run_pilot_v2_stage_a(
        input_path=input_path,
        v1_predictions_path=v1_path,
        v2_predictions_path=v2_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        client_factory=lambda config: stage_client,
    )

    summary_path = tmp_path / "pilot_summary_v2.csv"
    report_path = tmp_path / "pilot_execution_report_v2.md"
    audit = run_pilot_v2_stage_b(
        input_path=input_path,
        predictions_path=v2_path,
        ground_truth_path=ground_truth_path,
        item_matrix_path=tmp_path / "pilot_item_matrix_v2.csv",
        summary_path=summary_path,
        report_path=report_path,
    )

    assert audit.observed_rows == 72
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    by_scope = defaultdict(list)
    for row in summary:
        by_scope[row["scope"]].append(row)
    assert {scope: len(rows) for scope, rows in by_scope.items()} == {
        "overall": 1,
        "model": 3,
        "difficulty": 2,
        "class": 4,
        "model_difficulty": 6,
        "model_class": 12,
    }
    report = report_path.read_text(encoding="utf-8")
    assert "Qwen inherited from V1: 24 SUCCESS, 0 API calls" in report
    assert "max_completion_tokens: 2048" in report


def test_v2_stage_b_checks_72_combinations_before_ground_truth(tmp_path):
    input_path = write_pilot_input(tmp_path / "pilot.csv")
    config_path = write_v2_model_config(tmp_path / "models.json")
    v2_path = tmp_path / "pilot_predictions_v2.csv"
    client = MetadataQueueClient([metadata_result("CODE_RUNTIME")])
    run_pilot_v2_smoke(
        input_path=input_path,
        predictions_path=v2_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        client_factory=lambda config: client,
    )

    with pytest.raises(PilotDataError, match="Missing Pilot combinations"):
        run_pilot_v2_stage_b(
            input_path=input_path,
            predictions_path=v2_path,
            ground_truth_path=tmp_path / "must-not-be-read.csv",
            item_matrix_path=tmp_path / "matrix.csv",
            summary_path=tmp_path / "summary.csv",
            report_path=tmp_path / "report.md",
        )
