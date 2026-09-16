from __future__ import annotations

import csv
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from llm_experiment.config import ModelConfig
from llm_experiment.constants import (
    ALLOWED_LABELS,
    API_STATUS_FAILURE,
    API_STATUS_SUCCESS,
    INVALID_OUTPUT,
)
from llm_experiment.dataset import DemoSample, TestSample
from llm_experiment.prompts import render_prompt

PREDICTION_FIELDS = (
    "sample_id",
    "model",
    "api_model",
    "prompt_type",
    "run_id",
    "temperature",
    "raw_output",
    "prediction",
    "api_status",
    "api_error",
    "latency",
)


class CompletionClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class PredictionStateError(ValueError):
    """Raised when an existing prediction checkpoint is incompatible or corrupt."""


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    sample_id: str
    model: str
    api_model: str
    prompt_type: str
    run_id: int
    temperature: float
    raw_output: str
    prediction: str
    api_status: str
    api_error: str
    latency_seconds: float

    def to_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["latency"] = row.pop("latency_seconds")
        return row


def normalize_prediction(raw_output: str) -> str:
    normalized = raw_output.strip().replace("\r", "").replace("\n", "").upper()
    return normalized if normalized in ALLOWED_LABELS else INVALID_OUTPUT


def run_predictions(
    *,
    samples: Sequence[TestSample],
    demos: Sequence[DemoSample],
    model_config: ModelConfig,
    prompt_type: str,
    run_id: int,
    predictions_path: str | Path,
    prompt_dir: str | Path,
    client: CompletionClient,
) -> list[PredictionRecord]:
    if run_id < 1:
        raise ValueError("run_id must be at least 1")
    sample_ids = [sample.id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Prediction samples contain duplicate IDs")

    output_path = Path(predictions_path)
    records = load_prediction_records(output_path)
    _validate_checkpoint(
        records,
        target_ids=set(sample_ids),
        model_config=model_config,
        prompt_type=prompt_type,
        run_id=run_id,
    )
    by_sample_id = {record.sample_id: record for record in records}

    for sample in samples:
        if sample.id in by_sample_id:
            continue

        prompt = render_prompt(prompt_type, sample, demos, prompt_dir=prompt_dir)
        started_at = time.perf_counter()
        try:
            raw_output = client.complete(prompt)
            record = PredictionRecord(
                sample_id=sample.id,
                model=model_config.name,
                api_model=model_config.api_model,
                prompt_type=prompt_type,
                run_id=run_id,
                temperature=model_config.temperature,
                raw_output=raw_output,
                prediction=normalize_prediction(raw_output),
                api_status=API_STATUS_SUCCESS,
                api_error="",
                latency_seconds=time.perf_counter() - started_at,
            )
        except Exception as exc:  # The provider boundary may raise SDK or transport errors.
            record = PredictionRecord(
                sample_id=sample.id,
                model=model_config.name,
                api_model=model_config.api_model,
                prompt_type=prompt_type,
                run_id=run_id,
                temperature=model_config.temperature,
                raw_output="",
                prediction="",
                api_status=API_STATUS_FAILURE,
                api_error=f"{type(exc).__name__}: {exc}",
                latency_seconds=time.perf_counter() - started_at,
            )

        records.append(record)
        by_sample_id[record.sample_id] = record
        _write_records_atomically(output_path, records)

    return [by_sample_id[sample_id] for sample_id in sample_ids]


def load_prediction_records(path: str | Path) -> list[PredictionRecord]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise PredictionStateError(f"Cannot read prediction checkpoint: {path}") from exc

    records: list[PredictionRecord] = []
    with handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != PREDICTION_FIELDS:
            raise PredictionStateError(
                f"Prediction columns must be exactly {PREDICTION_FIELDS}; got {fields}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                records.append(
                    PredictionRecord(
                        sample_id=row["sample_id"],
                        model=row["model"],
                        api_model=row["api_model"],
                        prompt_type=row["prompt_type"],
                        run_id=int(row["run_id"]),
                        temperature=float(row["temperature"]),
                        raw_output=row["raw_output"],
                        prediction=row["prediction"],
                        api_status=row["api_status"],
                        api_error=row["api_error"],
                        latency_seconds=float(row["latency"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PredictionStateError(
                    f"Invalid prediction checkpoint row {row_number}"
                ) from exc
    return records


def _validate_checkpoint(
    records: Sequence[PredictionRecord],
    *,
    target_ids: set[str],
    model_config: ModelConfig,
    prompt_type: str,
    run_id: int,
) -> None:
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise PredictionStateError(f"Duplicate prediction for {record.sample_id}")
        seen.add(record.sample_id)
        if record.sample_id not in target_ids:
            raise PredictionStateError(
                f"Checkpoint contains sample outside this experiment: {record.sample_id}"
            )
        if (
            record.model != model_config.name
            or record.api_model != model_config.api_model
            or record.prompt_type != prompt_type
            or record.run_id != run_id
            or record.temperature != model_config.temperature
        ):
            raise PredictionStateError(
                f"Checkpoint metadata does not match this experiment: {record.sample_id}"
            )
        if record.api_status not in {API_STATUS_SUCCESS, API_STATUS_FAILURE}:
            raise PredictionStateError(f"Invalid API status for {record.sample_id}")
        if record.api_status == API_STATUS_SUCCESS:
            if record.prediction not in ALLOWED_LABELS | {INVALID_OUTPUT}:
                raise PredictionStateError(f"Invalid prediction for {record.sample_id}")
        elif record.prediction or not record.api_error:
            raise PredictionStateError(f"Invalid API failure record for {record.sample_id}")


def _write_records_atomically(path: Path, records: Sequence[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(record.to_csv_row() for record in records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PredictionStateError(f"Cannot persist prediction checkpoint: {path}") from exc
