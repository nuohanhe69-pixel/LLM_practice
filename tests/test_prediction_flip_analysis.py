import json
from itertools import product

import pytest

from experiments.formal_analysis_v1.analyze_results import (
    MODELS,
    ROOT,
    RUNS,
    AnalysisValidationError,
    read_csv,
    validate_metadata,
    validate_predictions,
)
from experiments.prediction_flip_analysis_v1.analyze_results import (
    DETAIL_FIELDS,
    SUMMARY_FIELDS,
    make_report,
    pair_predictions,
    summarize,
    verify_formal_delta,
)


def test_real_formal_inputs_pair_to_600_and_match_existing_analysis():
    dataset = read_csv(ROOT / "dataset_v2.csv")
    path = ROOT / "experiments/challenge_audit_v1/v4/candidates_full_v4.jsonl"
    v4 = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    labels, distractors, boundaries = validate_metadata(dataset, v4)
    assert len(boundaries) == 12
    predictions = validate_predictions(ROOT / "results", labels)
    assert len(predictions) == 20
    assert sum(map(len, predictions.values())) == 1200
    details = pair_predictions(predictions, labels, distractors)
    summary = summarize(details)
    assert len(details) == 600
    assert len(summary) == 15
    assert set(details[0]) == set(DETAIL_FIELDS)
    assert set(summary[0]) == set(SUMMARY_FIELDS)
    verify_formal_delta(
        summary,
        read_csv(ROOT / "experiments/formal_analysis_v1/artifacts/overall_results.csv"),
    )
    for row in summary:
        denominator = 120 if row["run_id"] == "combined" else 60
        assert (
            sum(
                row[state]
                for state in (
                    "correct_to_correct",
                    "correct_to_wrong",
                    "wrong_to_correct",
                    "wrong_to_wrong",
                )
            )
            == denominator
        )
        assert row["accuracy_delta"] == pytest.approx(
            (row["wrong_to_correct"] - row["correct_to_wrong"]) / denominator
        )


def test_pairing_and_flip_states_on_synthetic_predictions():
    labels = {f"T{i:03}": "NETWORK_API" for i in range(1, 61)}
    distractors = {item_id: "CODE_RUNTIME" for item_id in labels}
    predictions = {}
    for model, run in product(MODELS, RUNS):
        zero = {item_id: {"prediction": "NETWORK_API"} for item_id in labels}
        few = {item_id: {"prediction": "NETWORK_API"} for item_id in labels}
        zero["T003"]["prediction"] = "CODE_RUNTIME"
        zero["T004"]["prediction"] = "CODE_RUNTIME"
        few["T002"]["prediction"] = "CODE_RUNTIME"
        few["T004"]["prediction"] = "CODE_RUNTIME"
        predictions[model, "zero_shot", run] = zero
        predictions[model, "few_shot", run] = few
    details = pair_predictions(predictions, labels, distractors)
    summary = summarize(details)
    one_run = next(row for row in summary if row["model"] == MODELS[0] and row["run_id"] == "1")
    assert [
        one_run[state]
        for state in (
            "correct_to_correct",
            "correct_to_wrong",
            "wrong_to_correct",
            "wrong_to_wrong",
        )
    ] == [57, 1, 1, 1]
    assert one_run["accuracy_delta"] == 0
    combined = next(
        row for row in summary if row["model"] == MODELS[0] and row["run_id"] == "combined"
    )
    assert combined["correct_to_wrong"] == 2
    assert combined["wrong_to_correct"] == 2
    assert len(details) == 600
    report = make_report(summary, details)
    assert "NETWORK_API→CODE_RUNTIME=2" in report
    assert "Wrong prediction destination: CODE_RUNTIME=2" in report


def test_formal_delta_mismatch_is_rejected():
    summary = [
        {
            "model": MODELS[0],
            "run_id": "1",
            "zero_shot_accuracy": 0.5,
            "few_shot_accuracy": 0.4,
            "accuracy_delta": -0.1,
        }
    ]
    overall = [
        {"model": MODELS[0], "prompt_type": "zero_shot", "run1_accuracy": "0.6"},
        {"model": MODELS[0], "prompt_type": "few_shot", "run1_accuracy": "0.4"},
    ]
    with pytest.raises(AnalysisValidationError, match="ten unique rows"):
        verify_formal_delta(summary, overall)
    # Use a valid-sized baseline, then change one number.
    baseline = read_csv(ROOT / "experiments/formal_analysis_v1/artifacts/overall_results.csv")
    with pytest.raises(AnalysisValidationError, match="disagrees"):
        verify_formal_delta(summary, baseline)
