from __future__ import annotations

from types import SimpleNamespace

from llm_experiment.cli import main


def experiment_result(tmp_path, *, api_failures):
    return SimpleNamespace(
        predictions_path=tmp_path / "predictions.csv",
        metrics_path=tmp_path / "metrics.json",
        error_cases_path=tmp_path / "error_cases.csv",
        metrics={"api_failures": api_failures},
    )


def cli_args():
    return ["--model", "qwen_test", "--prompt-type", "zero_shot", "--run-id", "1"]


def test_cli_reports_incomplete_experiment_when_api_failures_remain(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "llm_experiment.cli.run_experiment",
        lambda **kwargs: experiment_result(tmp_path, api_failures=2),
    )

    exit_code = main(cli_args())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Experiment completed." not in output
    assert "Experiment incomplete: 2 API failure(s) remain." in output
    assert "Re-run the same command to retry failed samples." in output


def test_cli_reports_completed_experiment_when_no_api_failures_remain(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        "llm_experiment.cli.run_experiment",
        lambda **kwargs: experiment_result(tmp_path, api_failures=0),
    )

    exit_code = main(cli_args())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Experiment completed." in output
    assert "Experiment incomplete:" not in output
