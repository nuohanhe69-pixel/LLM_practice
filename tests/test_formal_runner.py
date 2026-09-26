from types import SimpleNamespace

import pytest

import run_formal_experiment as runner
from llm_experiment.api_client import ProviderFatalError


def install_fake_pipeline(monkeypatch, *, anomaly=None):
    calls = []
    loaded_paths = []

    def run_experiment(**kwargs):
        index = len(calls)
        calls.append(kwargs)
        if index == 0 and anomaly == "fatal":
            raise ProviderFatalError("HTTP 403: test provider failure")
        return SimpleNamespace(
            predictions_path=f"configuration_{index}/predictions.csv",
            metrics={
                "api_failures": int(index == 0 and anomaly == "api_failure"),
                "invalid_outputs": int(anomaly == "invalid_output"),
            },
        )

    def load_records(path):
        loaded_paths.append(path)
        first = path == "configuration_0/predictions.csv"
        return [
            SimpleNamespace(
                finish_reason="length" if first and anomaly == "length" else "stop",
                prediction="INVALID_OUTPUT" if anomaly == "invalid_output" else "NETWORK_API",
            )
        ]

    monkeypatch.setattr(runner, "run_experiment", run_experiment)
    monkeypatch.setattr(runner, "load_prediction_records", load_records)
    return calls, loaded_paths


def test_formal_matrix_is_exactly_twenty_full_configurations(monkeypatch, capsys):
    calls, loaded_paths = install_fake_pipeline(monkeypatch)
    assert runner.main() == 0
    expected = {
        (model, prompt, run_id)
        for model in (
            "qwen3_7_plus",
            "glm_5",
            "deepseek_v4_pro",
            "deepseek_v4_1_flash",
            "kimi_k3",
        )
        for prompt in ("zero_shot", "few_shot")
        for run_id in (1, 2)
    }
    assert len(calls) == len(runner.FORMAL_MATRIX) == 20
    assert set(runner.FORMAL_MATRIX) == expected
    assert [(c["model_name"], c["prompt_type"], c["run_id"]) for c in calls] == list(
        runner.FORMAL_MATRIX
    )
    assert all(c["limit"] is None for c in calls)
    assert all(set(c) == {"model_name", "prompt_type", "run_id", "limit"} for c in calls)
    assert len(loaded_paths) == 20
    assert capsys.readouterr().out.count("FORMAL EXPERIMENT COMPLETED") == 1


@pytest.mark.parametrize("anomaly", ["api_failure", "length", "fatal"])
def test_anomalies_return_nonzero_and_continue_all_configurations(monkeypatch, capsys, anomaly):
    calls, loaded_paths = install_fake_pipeline(monkeypatch, anomaly=anomaly)
    assert runner.main() == 1
    assert len(calls) == 20
    assert len(loaded_paths) == (19 if anomaly == "fatal" else 20)
    output = capsys.readouterr().out
    assert "qwen3_7_plus/zero_shot/run_1:" in output
    assert "kimi_k3/few_shot/run_2: OK" in output
    assert "FORMAL EXPERIMENT INCOMPLETE: 1 configuration(s)" in output
    assert "FORMAL EXPERIMENT COMPLETED" not in output


def test_invalid_output_alone_is_not_runner_failure(monkeypatch, capsys):
    calls, _ = install_fake_pipeline(monkeypatch, anomaly="invalid_output")
    assert runner.main() == 0
    assert len(calls) == 20
    assert "FORMAL EXPERIMENT COMPLETED" in capsys.readouterr().out


def test_rerun_delegates_same_matrix_to_resumable_pipeline(monkeypatch):
    calls, _ = install_fake_pipeline(monkeypatch)
    assert runner.main() == 0
    assert runner.main() == 0
    assert calls[:20] == calls[20:]
