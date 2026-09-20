from __future__ import annotations

import csv
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_experiment.api_client import ProviderRequestError, build_openai_client
from llm_experiment.config import ModelConfig, load_model_config
from llm_experiment.constants import ALLOWED_LABELS, API_STATUS_FAILURE, INVALID_OUTPUT
from llm_experiment.dataset import TestSample
from llm_experiment.prediction import CompletionClient, normalize_prediction
from llm_experiment.prompts import render_prompt

PILOT_MODEL_KEYS = ("qwen3_7_plus", "glm_5", "deepseek_v4_pro")
PILOT_API_MODELS = ("qwen3.7-plus", "glm-5", "deepseek-v4-pro")
PILOT_SAMPLE_COUNT = 24
PILOT_COMBINATION_COUNT = PILOT_SAMPLE_COUNT * len(PILOT_MODEL_KEYS)

PILOT_INPUT_FIELDS = ("id", "error_text")
PILOT_PREDICTION_FIELDS = (
    "id",
    "model",
    "run",
    "prompt_type",
    "raw_output",
    "normalized_output",
    "status",
    "error_type",
    "latency_ms",
    "actual_temperature",
    "timestamp",
)
PILOT_GROUND_TRUTH_FIELDS = (
    "id",
    "candidate_id",
    "label",
    "difficulty",
    "difficulty_score",
    "title",
    "candidate_source_plan",
    "source_file",
    "split",
)
PILOT_MATRIX_FIELDS = (
    "id",
    "candidate_id",
    "difficulty",
    "ground_truth",
    *PILOT_API_MODELS,
    "correct_count",
    "all_models_correct",
    "model_disagreement",
)
PILOT_SUMMARY_FIELDS = (
    "scope",
    "model",
    "difficulty",
    "total",
    "success",
    "invalid_output",
    "api_failure",
    "correct",
    "accuracy",
)

FINAL_STATUSES = frozenset({"SUCCESS", INVALID_OUTPUT, API_STATUS_FAILURE})


class PilotDataError(ValueError):
    """Raised when Pilot inputs, checkpoints, or Ground Truth are invalid."""


@dataclass(frozen=True, slots=True)
class PilotPrediction:
    id: str
    model: str
    run: int
    prompt_type: str
    raw_output: str
    normalized_output: str
    status: str
    error_type: str
    latency_ms: float
    actual_temperature: float
    timestamp: str

    def to_csv_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PilotGroundTruth:
    id: str
    candidate_id: str
    label: str
    difficulty: str


@dataclass(frozen=True, slots=True)
class PilotAudit:
    expected_combinations: int
    observed_rows: int
    missing_combinations: tuple[str, ...]
    duplicate_combinations: tuple[str, ...]
    unexpected_combinations: tuple[str, ...]
    success: int
    invalid_output: int
    api_failure: int


ClientFactory = Callable[[ModelConfig], CompletionClient]


def load_pilot_input(path: str | Path) -> tuple[TestSample, ...]:
    """Stage A loader: accepts exactly id and error_text, and nothing else."""
    input_path = Path(path)
    try:
        handle = input_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise PilotDataError(f"Cannot read Pilot input: {input_path}") from exc

    samples: list[TestSample] = []
    seen_ids: set[str] = set()
    with handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != PILOT_INPUT_FIELDS:
            raise PilotDataError(
                f"Stage A input columns must be exactly {PILOT_INPUT_FIELDS}; got {actual_fields}"
            )
        for row_number, row in enumerate(reader, start=2):
            sample_id = row["id"].strip()
            error_text = row["error_text"].strip()
            if not sample_id or not error_text:
                raise PilotDataError(f"Pilot input row {row_number} has an empty id or error_text")
            if sample_id in seen_ids:
                raise PilotDataError(f"Duplicate Pilot input id: {sample_id}")
            seen_ids.add(sample_id)
            samples.append(TestSample(id=sample_id, error_text=error_text))

    if len(samples) != PILOT_SAMPLE_COUNT:
        raise PilotDataError(
            f"Pilot input must contain exactly {PILOT_SAMPLE_COUNT} rows; got {len(samples)}"
        )
    return tuple(samples)


def run_pilot_stage_a(
    *,
    input_path: str | Path,
    predictions_path: str | Path,
    model_config_path: str | Path = "configs/models.json",
    prompt_dir: str | Path = "prompts",
    model_keys: Sequence[str] = PILOT_MODEL_KEYS,
    sample_limit: int | None = None,
    client_factory: ClientFactory = build_openai_client,
) -> PilotAudit:
    samples = load_pilot_input(input_path)
    if sample_limit is not None:
        if not 1 <= sample_limit <= len(samples):
            raise PilotDataError(f"sample_limit must be between 1 and {len(samples)}")
        samples = samples[:sample_limit]
    if not model_keys:
        raise PilotDataError("At least one Pilot model is required")
    if len(model_keys) != len(set(model_keys)):
        raise PilotDataError("Pilot model keys contain duplicates")

    configs = [load_model_config(model_config_path, key) for key in model_keys]
    api_models = [config.api_model for config in configs]
    if len(api_models) != len(set(api_models)):
        raise PilotDataError("Pilot API model names contain duplicates")

    output_path = Path(predictions_path)
    records = load_pilot_predictions(output_path)
    _validate_stage_a_checkpoint(records, samples=samples, configs=configs)
    record_indexes = {(record.id, record.model): index for index, record in enumerate(records)}
    clients: dict[str, CompletionClient] = {}

    for sample in samples:
        for config in configs:
            pair = (sample.id, config.api_model)
            existing = records[record_indexes[pair]] if pair in record_indexes else None
            if existing is not None and existing.status in {"SUCCESS", INVALID_OUTPUT}:
                continue

            client = clients.get(config.api_model)
            if client is None:
                client = client_factory(config)
                clients[config.api_model] = client
            prompt = render_prompt("zero_shot", sample, (), prompt_dir=prompt_dir)
            started_at = time.perf_counter()
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                raw_output = client.complete(prompt)
                normalized = normalize_prediction(raw_output)
                status = INVALID_OUTPUT if normalized == INVALID_OUTPUT else "SUCCESS"
                error_type = ""
            except ProviderRequestError as exc:
                raw_output = ""
                normalized = ""
                status = API_STATUS_FAILURE
                error_type = f"{type(exc).__name__}: {exc}"

            record = PilotPrediction(
                id=sample.id,
                model=config.api_model,
                run=1,
                prompt_type="zero_shot",
                raw_output=raw_output,
                normalized_output=normalized,
                status=status,
                error_type=error_type,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
                actual_temperature=config.temperature,
                timestamp=timestamp,
            )
            if pair in record_indexes:
                records[record_indexes[pair]] = record
            else:
                record_indexes[pair] = len(records)
                records.append(record)
            _write_pilot_predictions(output_path, records, samples=samples, api_models=api_models)

    audit = audit_pilot_predictions(
        records,
        sample_ids=[sample.id for sample in samples],
        api_models=api_models,
    )
    _require_complete_audit(audit)
    return audit


def load_pilot_predictions(path: str | Path) -> list[PilotPrediction]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        return []
    try:
        handle = prediction_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise PilotDataError(f"Cannot read Pilot predictions: {prediction_path}") from exc

    records: list[PilotPrediction] = []
    with handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != PILOT_PREDICTION_FIELDS:
            raise PilotDataError(
                "Pilot prediction columns must be exactly "
                f"{PILOT_PREDICTION_FIELDS}; got {actual_fields}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                record = PilotPrediction(
                    id=row["id"],
                    model=row["model"],
                    run=int(row["run"]),
                    prompt_type=row["prompt_type"],
                    raw_output=row["raw_output"],
                    normalized_output=row["normalized_output"],
                    status=row["status"],
                    error_type=row["error_type"],
                    latency_ms=float(row["latency_ms"]),
                    actual_temperature=float(row["actual_temperature"]),
                    timestamp=row["timestamp"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PilotDataError(f"Invalid Pilot prediction row {row_number}") from exc
            _validate_prediction_record(record, row_number=row_number)
            records.append(record)
    return records


def audit_pilot_predictions(
    records: Sequence[PilotPrediction],
    *,
    sample_ids: Sequence[str],
    api_models: Sequence[str],
) -> PilotAudit:
    expected_pairs = {(sample_id, model) for sample_id in sample_ids for model in api_models}
    pair_counts = Counter((record.id, record.model) for record in records)
    observed_pairs = set(pair_counts)
    missing = sorted(expected_pairs - observed_pairs)
    unexpected = sorted(observed_pairs - expected_pairs)
    duplicates = sorted(pair for pair, count in pair_counts.items() if count > 1)
    status_counts = Counter(record.status for record in records)
    return PilotAudit(
        expected_combinations=len(expected_pairs),
        observed_rows=len(records),
        missing_combinations=tuple(_format_pair(pair) for pair in missing),
        duplicate_combinations=tuple(_format_pair(pair) for pair in duplicates),
        unexpected_combinations=tuple(_format_pair(pair) for pair in unexpected),
        success=status_counts["SUCCESS"],
        invalid_output=status_counts[INVALID_OUTPUT],
        api_failure=status_counts[API_STATUS_FAILURE],
    )


def run_pilot_stage_b(
    *,
    input_path: str | Path,
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    item_matrix_path: str | Path,
    summary_path: str | Path,
    report_path: str | Path,
    smoke_predictions_path: str | Path | None = None,
) -> PilotAudit:
    records = load_pilot_predictions(predictions_path)
    samples = load_pilot_input(input_path)
    ordered_ids = [sample.id for sample in samples]
    audit = audit_pilot_predictions(
        records,
        sample_ids=ordered_ids,
        api_models=PILOT_API_MODELS,
    )
    if audit.expected_combinations != PILOT_COMBINATION_COUNT:
        raise PilotDataError(
            f"Expected Pilot combination count must be {PILOT_COMBINATION_COUNT}; "
            f"got {audit.expected_combinations}"
        )
    _require_complete_audit(audit)

    smoke_audit = None
    if smoke_predictions_path is not None:
        smoke_records = load_pilot_predictions(smoke_predictions_path)
        smoke_audit = audit_pilot_predictions(
            smoke_records,
            sample_ids=ordered_ids[:1],
            api_models=PILOT_API_MODELS[:1],
        )
        _require_complete_audit(smoke_audit)

    # Ground Truth is deliberately opened only after the Stage A completeness audit passes.
    ground_truth = _load_pilot_ground_truth(ground_truth_path)
    if list(ground_truth) != ordered_ids:
        raise PilotDataError("Pilot Ground Truth IDs or ordering do not match Stage A input")

    records_by_pair = {(record.id, record.model): record for record in records}
    matrix_rows = _build_item_matrix(ordered_ids, ground_truth, records_by_pair)
    summary_rows = _build_summary_rows(records, ground_truth)
    _write_csv_atomically(Path(item_matrix_path), PILOT_MATRIX_FIELDS, matrix_rows)
    _write_csv_atomically(Path(summary_path), PILOT_SUMMARY_FIELDS, summary_rows)
    _write_report_atomically(
        Path(report_path),
        audit=audit,
        smoke_audit=smoke_audit,
        records=records,
        matrix_rows=matrix_rows,
        summary_rows=summary_rows,
    )
    return audit


def _validate_stage_a_checkpoint(
    records: Sequence[PilotPrediction],
    *,
    samples: Sequence[TestSample],
    configs: Sequence[ModelConfig],
) -> None:
    sample_ids = [sample.id for sample in samples]
    api_models = [config.api_model for config in configs]
    audit = audit_pilot_predictions(records, sample_ids=sample_ids, api_models=api_models)
    if audit.duplicate_combinations:
        raise PilotDataError(
            f"Duplicate Pilot combinations: {', '.join(audit.duplicate_combinations)}"
        )
    if audit.unexpected_combinations:
        raise PilotDataError(
            f"Unexpected Pilot combinations: {', '.join(audit.unexpected_combinations)}"
        )
    configs_by_model = {config.api_model: config for config in configs}
    for record in records:
        config = configs_by_model[record.model]
        if (
            record.run != 1
            or record.prompt_type != "zero_shot"
            or record.actual_temperature != config.temperature
        ):
            pair = f"{record.id}/{record.model}"
            raise PilotDataError(f"Pilot checkpoint metadata mismatch: {pair}")


def _validate_prediction_record(record: PilotPrediction, *, row_number: int) -> None:
    if not record.id or not record.model or not record.timestamp:
        raise PilotDataError(f"Pilot prediction row {row_number} has an empty identity field")
    if record.status not in FINAL_STATUSES:
        raise PilotDataError(
            f"Pilot prediction row {row_number} has invalid status {record.status!r}"
        )
    if record.status == "SUCCESS":
        if record.normalized_output not in ALLOWED_LABELS or record.error_type:
            raise PilotDataError(f"Invalid SUCCESS prediction row {row_number}")
    elif record.status == INVALID_OUTPUT:
        if record.normalized_output != INVALID_OUTPUT or record.error_type:
            raise PilotDataError(f"Invalid INVALID_OUTPUT prediction row {row_number}")
    elif record.raw_output or record.normalized_output or not record.error_type:
        raise PilotDataError(f"Invalid API_FAILURE prediction row {row_number}")


def _load_pilot_ground_truth(path: str | Path) -> dict[str, PilotGroundTruth]:
    ground_truth_path = Path(path)
    try:
        handle = ground_truth_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise PilotDataError(f"Cannot read Pilot Ground Truth: {ground_truth_path}") from exc

    ground_truth: dict[str, PilotGroundTruth] = {}
    with handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != PILOT_GROUND_TRUTH_FIELDS:
            raise PilotDataError(
                "Pilot Ground Truth columns must be exactly "
                f"{PILOT_GROUND_TRUTH_FIELDS}; got {actual_fields}"
            )
        for row_number, row in enumerate(reader, start=2):
            sample_id = row["id"].strip()
            label = row["label"].strip()
            difficulty = row["difficulty"].strip()
            if sample_id in ground_truth:
                raise PilotDataError(f"Duplicate Pilot Ground Truth id: {sample_id}")
            if label not in ALLOWED_LABELS:
                raise PilotDataError(f"Ground Truth row {row_number} has unknown label {label!r}")
            if difficulty not in {"Medium", "Hard"}:
                raise PilotDataError(
                    f"Ground Truth row {row_number} has unknown difficulty {difficulty!r}"
                )
            if row["split"] != "pilot":
                raise PilotDataError(f"Ground Truth row {row_number} is not in the pilot split")
            ground_truth[sample_id] = PilotGroundTruth(
                id=sample_id,
                candidate_id=row["candidate_id"],
                label=label,
                difficulty=difficulty,
            )
    if len(ground_truth) != PILOT_SAMPLE_COUNT:
        raise PilotDataError(
            f"Pilot Ground Truth must contain {PILOT_SAMPLE_COUNT} rows; got {len(ground_truth)}"
        )
    return ground_truth


def _build_item_matrix(
    ordered_ids: Sequence[str],
    ground_truth: dict[str, PilotGroundTruth],
    records_by_pair: dict[tuple[str, str], PilotPrediction],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id in ordered_ids:
        truth = ground_truth[sample_id]
        predictions = [records_by_pair[(sample_id, model)] for model in PILOT_API_MODELS]
        displayed = [_display_prediction(record) for record in predictions]
        correct_count = sum(value == truth.label for value in displayed)
        rows.append(
            {
                "id": sample_id,
                "candidate_id": truth.candidate_id,
                "difficulty": truth.difficulty,
                "ground_truth": truth.label,
                **dict(zip(PILOT_API_MODELS, displayed, strict=True)),
                "correct_count": correct_count,
                "all_models_correct": str(correct_count == len(PILOT_API_MODELS)).lower(),
                "model_disagreement": str(len(set(displayed)) > 1).lower(),
            }
        )
    return rows


def _build_summary_rows(
    records: Sequence[PilotPrediction],
    ground_truth: dict[str, PilotGroundTruth],
) -> list[dict[str, object]]:
    groups: list[tuple[str, str, str, list[PilotPrediction]]] = [
        ("overall", "ALL", "ALL", list(records))
    ]
    groups.extend(
        ("model", model, "ALL", [record for record in records if record.model == model])
        for model in PILOT_API_MODELS
    )
    groups.extend(
        (
            "difficulty",
            "ALL",
            difficulty,
            [record for record in records if ground_truth[record.id].difficulty == difficulty],
        )
        for difficulty in ("Medium", "Hard")
    )
    groups.extend(
        (
            "model_difficulty",
            model,
            difficulty,
            [
                record
                for record in records
                if record.model == model and ground_truth[record.id].difficulty == difficulty
            ],
        )
        for model in PILOT_API_MODELS
        for difficulty in ("Medium", "Hard")
    )

    rows: list[dict[str, object]] = []
    for scope, model, difficulty, group_records in groups:
        statuses = Counter(record.status for record in group_records)
        successful = statuses["SUCCESS"] + statuses[INVALID_OUTPUT]
        correct = sum(
            record.status == "SUCCESS" and record.normalized_output == ground_truth[record.id].label
            for record in group_records
        )
        rows.append(
            {
                "scope": scope,
                "model": model,
                "difficulty": difficulty,
                "total": len(group_records),
                "success": statuses["SUCCESS"],
                "invalid_output": statuses[INVALID_OUTPUT],
                "api_failure": statuses[API_STATUS_FAILURE],
                "correct": correct,
                "accuracy": f"{correct / successful:.6f}" if successful else "",
            }
        )
    return rows


def _write_pilot_predictions(
    path: Path,
    records: Sequence[PilotPrediction],
    *,
    samples: Sequence[TestSample],
    api_models: Sequence[str],
) -> None:
    order = {
        (sample.id, model): index
        for index, (sample, model) in enumerate(
            (sample, model) for sample in samples for model in api_models
        )
    }
    ordered = sorted(records, key=lambda record: order[(record.id, record.model)])
    _write_csv_atomically(
        path,
        PILOT_PREDICTION_FIELDS,
        [record.to_csv_row() for record in ordered],
    )


def _write_csv_atomically(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PilotDataError(f"Cannot write Pilot artifact: {path}") from exc


def _write_report_atomically(
    path: Path,
    *,
    audit: PilotAudit,
    smoke_audit: PilotAudit | None,
    records: Sequence[PilotPrediction],
    matrix_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
) -> None:
    model_rows = [row for row in summary_rows if row["scope"] == "model"]
    disagreement_ids = [
        str(row["id"]) for row in matrix_rows if row["model_disagreement"] == "true"
    ]
    all_correct_ids = [str(row["id"]) for row in matrix_rows if row["all_models_correct"] == "true"]
    lines = [
        "# Difficulty Pilot V1 Execution Report",
        "",
        "## Execution status",
        "",
        "- Smoke Test: "
        + (
            f"COMPLETE ({smoke_audit.observed_rows}/{smoke_audit.expected_combinations})"
            if smoke_audit is not None
            else "NOT CHECKED"
        ),
        "- Stage A: COMPLETE",
        "- Stage B: COMPLETE",
        f"- Expected `(sample, model)` combinations: {audit.expected_combinations}",
        f"- Observed rows: {audit.observed_rows}",
        f"- Missing combinations: {len(audit.missing_combinations)}",
        f"- Duplicate combinations: {len(audit.duplicate_combinations)}",
        f"- Unexpected combinations: {len(audit.unexpected_combinations)}",
        f"- SUCCESS: {audit.success}",
        f"- INVALID_OUTPUT: {audit.invalid_output}",
        f"- API_FAILURE: {audit.api_failure}",
        "",
        "## Per-model results",
        "",
        "| Model | Total | SUCCESS | INVALID_OUTPUT | API_FAILURE | Correct | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| {model} | {total} | {success} | {invalid_output} | {api_failure} | "
        "{correct} | {accuracy} |".format(**row)
        for row in model_rows
    )
    lines.extend(
        [
            "",
            "## Item-level observations",
            "",
            f"- Model-disagreement items ({len(disagreement_ids)}): "
            + (", ".join(disagreement_ids) if disagreement_ids else "none"),
            f"- Three-model-correct items ({len(all_correct_ids)}): "
            + (", ".join(all_correct_ids) if all_correct_ids else "none"),
            "- These are execution observations only. No sample text, Ground Truth, difficulty, "
            "Prompt, taxonomy, or formal experiment configuration was modified.",
            "- Disagreement and unanimous correctness are review signals, not automatic evidence "
            "of ambiguity, shortcut, or a required rewrite.",
        ]
    )
    invalid_records = [record for record in records if record.status == INVALID_OUTPUT]
    if invalid_records:
        lines.extend(["", "## Invalid-output observations", ""])
        for model in PILOT_API_MODELS:
            model_invalid = [record for record in invalid_records if record.model == model]
            if not model_invalid:
                continue
            empty_raw = sum(not record.raw_output for record in model_invalid)
            lines.append(
                f"- {model}: {len(model_invalid)} INVALID_OUTPUT "
                f"(empty raw_output: {empty_raw}, non-empty raw_output: "
                f"{len(model_invalid) - empty_raw})"
            )
        lines.append("- These successful API responses remain frozen and were not retried.")
    lines.append("")
    _write_text_atomically(path, "\n".join(lines))


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PilotDataError(f"Cannot write Pilot report: {path}") from exc


def _display_prediction(record: PilotPrediction) -> str:
    return API_STATUS_FAILURE if record.status == API_STATUS_FAILURE else record.normalized_output


def _require_complete_audit(audit: PilotAudit) -> None:
    if audit.missing_combinations:
        raise PilotDataError(f"Missing Pilot combinations: {', '.join(audit.missing_combinations)}")
    if audit.duplicate_combinations:
        raise PilotDataError(
            f"Duplicate Pilot combinations: {', '.join(audit.duplicate_combinations)}"
        )
    if audit.unexpected_combinations:
        raise PilotDataError(
            f"Unexpected Pilot combinations: {', '.join(audit.unexpected_combinations)}"
        )
    status_total = audit.success + audit.invalid_output + audit.api_failure
    if status_total != audit.observed_rows:
        raise PilotDataError(
            f"Pilot status counts total {status_total}, but observed rows are {audit.observed_rows}"
        )


def _format_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"
