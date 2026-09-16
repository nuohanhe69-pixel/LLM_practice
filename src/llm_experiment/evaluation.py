from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_experiment.constants import (
    ALLOWED_LABELS,
    API_STATUS_FAILURE,
    API_STATUS_SUCCESS,
    INVALID_OUTPUT,
)
from llm_experiment.dataset import load_dataset
from llm_experiment.prediction import PredictionRecord, load_prediction_records

ERROR_CASE_FIELDS = (
    "sample_id",
    "error_text",
    "ground_truth",
    "prediction",
    "raw_output",
    "error_type",
    "api_status",
    "api_error",
)


class EvaluationError(ValueError):
    """Raised when predictions cannot be matched to the frozen ground truth."""


@dataclass(frozen=True, slots=True)
class GroundTruth:
    sample_id: str
    error_text: str
    label: str


def evaluate_predictions(
    *,
    dataset_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    expected_sample_ids: Sequence[str],
) -> dict[str, object]:
    ordered_ids = list(expected_sample_ids)
    if not ordered_ids:
        raise EvaluationError("At least one expected sample is required")
    if len(ordered_ids) != len(set(ordered_ids)):
        raise EvaluationError("Expected sample IDs contain duplicates")

    ground_truth = _load_ground_truth(dataset_path)
    unknown_ids = [sample_id for sample_id in ordered_ids if sample_id not in ground_truth]
    if unknown_ids:
        raise EvaluationError(f"Unknown test sample IDs: {', '.join(unknown_ids)}")

    records = load_prediction_records(predictions_path)
    records_by_id: dict[str, PredictionRecord] = {}
    for record in records:
        if record.sample_id in records_by_id:
            raise EvaluationError(f"Duplicate prediction: {record.sample_id}")
        records_by_id[record.sample_id] = record

    missing = [sample_id for sample_id in ordered_ids if sample_id not in records_by_id]
    if missing:
        raise EvaluationError(f"Missing predictions: {', '.join(missing)}")
    unexpected = [sample_id for sample_id in records_by_id if sample_id not in set(ordered_ids)]
    if unexpected:
        raise EvaluationError(f"Unexpected predictions: {', '.join(unexpected)}")

    ordered_records = [records_by_id[sample_id] for sample_id in ordered_ids]
    _validate_prediction_records(ordered_records)
    metrics, error_cases = _calculate_metrics(ordered_records, ground_truth)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(destination / "metrics.json", metrics)
    _write_csv_atomically(destination / "error_cases.csv", ERROR_CASE_FIELDS, error_cases)
    return metrics


def _load_ground_truth(path: str | Path) -> dict[str, GroundTruth]:
    dataset_path = Path(path)
    load_dataset(dataset_path)
    ground_truth: dict[str, GroundTruth] = {}
    try:
        handle = dataset_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise EvaluationError(f"Cannot read ground truth: {dataset_path}") from exc

    with handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if row["split"] != "test":
                continue
            sample_id = row["id"]
            label = row["label"]
            if label not in ALLOWED_LABELS:
                raise EvaluationError(f"Row {row_number} has unknown Ground Truth label {label!r}")
            ground_truth[sample_id] = GroundTruth(
                sample_id=sample_id,
                error_text=row["error_text"],
                label=label,
            )
    return ground_truth


def _validate_prediction_records(records: Sequence[PredictionRecord]) -> None:
    first = records[0]
    for record in records:
        if (
            record.model != first.model
            or record.api_model != first.api_model
            or record.prompt_type != first.prompt_type
            or record.run_id != first.run_id
            or record.temperature != first.temperature
        ):
            raise EvaluationError("Prediction file mixes multiple experiment configurations")
        if record.api_status == API_STATUS_SUCCESS:
            if record.prediction not in ALLOWED_LABELS | {INVALID_OUTPUT}:
                raise EvaluationError(f"Invalid prediction value for {record.sample_id}")
        elif record.api_status == API_STATUS_FAILURE:
            if record.prediction or not record.api_error:
                raise EvaluationError(f"Invalid API failure row for {record.sample_id}")
        else:
            raise EvaluationError(f"Unknown API status for {record.sample_id}")


def _calculate_metrics(
    records: Sequence[PredictionRecord],
    ground_truth: dict[str, GroundTruth],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    successful = [record for record in records if record.api_status == API_STATUS_SUCCESS]
    api_failures = [record for record in records if record.api_status == API_STATUS_FAILURE]
    invalid_outputs = [record for record in successful if record.prediction == INVALID_OUTPUT]
    correct = [
        record for record in successful if record.prediction == ground_truth[record.sample_id].label
    ]

    per_class: dict[str, dict[str, int | float | None]] = {}
    for label in sorted(ALLOWED_LABELS):
        class_records = [
            record for record in records if ground_truth[record.sample_id].label == label
        ]
        class_successful = [
            record for record in class_records if record.api_status == API_STATUS_SUCCESS
        ]
        class_correct = [record for record in class_successful if record.prediction == label]
        per_class[label] = {
            "total_samples": len(class_records),
            "successful_predictions": len(class_successful),
            "correct_predictions": len(class_correct),
            "accuracy": (len(class_correct) / len(class_successful) if class_successful else None),
        }

    first = records[0]
    metrics: dict[str, object] = {
        "model": first.model,
        "api_model": first.api_model,
        "prompt_type": first.prompt_type,
        "run_id": first.run_id,
        "temperature": first.temperature,
        "total_samples": len(records),
        "successful_predictions": len(successful),
        "invalid_outputs": len(invalid_outputs),
        "api_failures": len(api_failures),
        "correct_predictions": len(correct),
        "accuracy": len(correct) / len(successful) if successful else None,
        "per_class": per_class,
    }

    error_cases: list[dict[str, str]] = []
    for record in records:
        truth = ground_truth[record.sample_id]
        if record.api_status == API_STATUS_FAILURE:
            error_type = API_STATUS_FAILURE
        elif record.prediction == INVALID_OUTPUT:
            error_type = INVALID_OUTPUT
        elif record.prediction != truth.label:
            error_type = "CLASSIFICATION_ERROR"
        else:
            continue
        error_cases.append(
            {
                "sample_id": record.sample_id,
                "error_text": truth.error_text,
                "ground_truth": truth.label,
                "prediction": record.prediction,
                "raw_output": record.raw_output,
                "error_type": error_type,
                "api_status": record.api_status,
                "api_error": record.api_error,
            }
        )
    return metrics, error_cases


def _write_json_atomically(path: Path, data: dict[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise EvaluationError(f"Cannot write metrics: {path}") from exc


def _write_csv_atomically(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise EvaluationError(f"Cannot write error cases: {path}") from exc
