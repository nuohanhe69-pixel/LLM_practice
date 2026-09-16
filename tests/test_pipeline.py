from __future__ import annotations

import json

from llm_experiment.pipeline import resolve_result_directory, run_experiment
from tests.helpers import write_protocol_dataset
from tests.test_prediction import SequenceClient


def write_model_config(path):
    path.write_text(
        json.dumps(
            {
                "models": {
                    "qwen_test": {
                        "api_model": "qwen-test",
                        "base_url": "https://example.invalid/v1",
                        "api_key_env": "TEST_API_KEY",
                        "temperature": 0,
                        "max_tokens": 8,
                        "timeout_seconds": 5,
                        "max_retries": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_smoke_run_uses_full_pipeline_and_isolated_result_directory(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    config_path = write_model_config(tmp_path / "models.json")
    results_root = tmp_path / "results"
    client = SequenceClient(["CODE_RUNTIME", "CONTEXT_LIMIT", "invalid"])

    result = run_experiment(
        model_name="qwen_test",
        prompt_type="few_shot",
        run_id=1,
        limit=3,
        dataset_path=dataset_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        results_root=results_root,
        client=client,
    )

    expected_dir = results_root / "qwen_test" / "few_shot" / "run_1" / "smoke_limit_3"
    assert result.output_dir == expected_dir
    assert result.predictions_path.is_file()
    assert result.metrics_path.is_file()
    assert result.error_cases_path.is_file()
    assert result.metrics["total_samples"] == 3
    assert result.metrics["invalid_outputs"] == 1
    assert len(client.prompts) == 3
    assert all("固定示例" in prompt for prompt in client.prompts)
    assert not (results_root / "qwen_test" / "few_shot" / "run_1" / "predictions.csv").exists()

    resumed_client = SequenceClient([])
    resumed = run_experiment(
        model_name="qwen_test",
        prompt_type="few_shot",
        run_id=1,
        limit=3,
        dataset_path=dataset_path,
        model_config_path=config_path,
        prompt_dir="prompts",
        results_root=results_root,
        client=resumed_client,
    )
    assert resumed.metrics == result.metrics
    assert resumed_client.prompts == []


def test_formal_and_limited_runs_resolve_to_different_directories(tmp_path):
    formal = resolve_result_directory(
        results_root=tmp_path,
        model_name="qwen_plus",
        prompt_type="zero_shot",
        run_id=2,
        limit=None,
    )
    smoke = resolve_result_directory(
        results_root=tmp_path,
        model_name="qwen_plus",
        prompt_type="zero_shot",
        run_id=2,
        limit=10,
    )

    assert formal == tmp_path / "qwen_plus" / "zero_shot" / "run_2"
    assert smoke == formal / "smoke_limit_10"
