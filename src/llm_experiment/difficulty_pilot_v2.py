from __future__ import annotations

import csv
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from llm_experiment.api_client import (
    CompletionResult,
    ProviderRequestError,
    build_openai_client,
)
from llm_experiment.config import ModelConfig, load_model_config
from llm_experiment.constants import ALLOWED_LABELS, API_STATUS_FAILURE, INVALID_OUTPUT
from llm_experiment.dataset import TestSample
from llm_experiment.difficulty_pilot import (
    PILOT_API_MODELS,
    PILOT_COMBINATION_COUNT,
    PILOT_MATRIX_FIELDS,
    PilotAudit,
    PilotDataError,
    PilotGroundTruth,
    PilotPrediction,
    _load_pilot_ground_truth,
    _require_complete_audit,
    _write_csv_atomically,
    _write_text_atomically,
    load_pilot_input,
    load_pilot_predictions,
)
from llm_experiment.prediction import normalize_prediction
from llm_experiment.prompts import render_prompt

DEEPSEEK_MODEL_KEY = "deepseek_v4_pro"
DEEPSEEK_API_MODEL = "deepseek-v4-pro"
# Alibaba Model Studio recommends max_completion_tokens for thinking models:
# https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
DEEPSEEK_MAX_COMPLETION_TOKENS = 2048
INHERITED_MODELS = PILOT_API_MODELS[:2]

PILOT_V2_PREDICTION_FIELDS = (
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
    "finish_reason",
    "completion_tokens",
    "reasoning_tokens",
    "reasoning_content_present",
    "reasoning_content_length",
    "max_completion_tokens",
    "source",
)
PILOT_V2_SUMMARY_FIELDS = (
    "scope",
    "model",
    "difficulty",
    "ground_truth",
    "total",
    "success",
    "invalid_output",
    "api_failure",
    "correct",
    "accuracy",
)


class MetadataCompletionClient(Protocol):
    def complete_with_metadata(self, prompt: str) -> CompletionResult: ...


MetadataClientFactory = Callable[[ModelConfig], MetadataCompletionClient]


@dataclass(frozen=True, slots=True)
class PilotV2Prediction:
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
    finish_reason: str | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    reasoning_content_present: bool | None
    reasoning_content_length: int | None
    max_completion_tokens: int | None
    source: str

    def to_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        for key, value in tuple(row.items()):
            if value is None:
                row[key] = ""
            elif isinstance(value, bool):
                row[key] = str(value).lower()
        return row


def run_pilot_v2_smoke(
    *,
    input_path: str | Path,
    predictions_path: str | Path,
    model_config_path: str | Path = "configs/models.json",
    prompt_dir: str | Path = "prompts",
    client_factory: MetadataClientFactory = build_openai_client,
) -> PilotV2Prediction:
    samples = load_pilot_input(input_path)
    config = _load_deepseek_v2_config(model_config_path)
    records = load_pilot_v2_predictions(predictions_path)
    _validate_v2_pairs(records, sample_ids=[sample.id for sample in samples])
    by_pair = {(record.id, record.model): record for record in records}
    pair = (samples[0].id, DEEPSEEK_API_MODEL)
    existing = by_pair.get(pair)
    if existing is None or existing.status == API_STATUS_FAILURE:
        client = client_factory(config)
        record = _call_deepseek(
            sample=samples[0],
            config=config,
            prompt_dir=prompt_dir,
            client=client,
        )
        by_pair[pair] = record
        _write_v2_predictions(Path(predictions_path), list(by_pair.values()), samples=samples)
    else:
        record = existing

    _validate_smoke_result(record)
    return record


def run_pilot_v2_stage_a(
    *,
    input_path: str | Path,
    v1_predictions_path: str | Path,
    v2_predictions_path: str | Path,
    model_config_path: str | Path = "configs/models.json",
    prompt_dir: str | Path = "prompts",
    client_factory: MetadataClientFactory = build_openai_client,
) -> PilotAudit:
    samples = load_pilot_input(input_path)
    sample_ids = [sample.id for sample in samples]
    inherited = inherit_v1_predictions(
        v1_predictions_path=v1_predictions_path,
        sample_ids=sample_ids,
    )
    inherited_by_pair = {(record.id, record.model): record for record in inherited}
    config = _load_deepseek_v2_config(model_config_path)
    existing = load_pilot_v2_predictions(v2_predictions_path)
    _validate_v2_pairs(existing, sample_ids=sample_ids)
    by_pair = {(record.id, record.model): record for record in existing}

    for pair, inherited_record in inherited_by_pair.items():
        current = by_pair.get(pair)
        if current is not None and current != inherited_record:
            raise PilotDataError(f"V2 inherited record does not match V1: {pair[0]}/{pair[1]}")
        by_pair[pair] = inherited_record

    smoke_pair = (sample_ids[0], DEEPSEEK_API_MODEL)
    if smoke_pair not in by_pair:
        raise PilotDataError("DeepSeek V2 smoke must complete before the 24-sample run")
    _validate_smoke_result(by_pair[smoke_pair])

    client: MetadataCompletionClient | None = None
    for sample in samples:
        pair = (sample.id, DEEPSEEK_API_MODEL)
        current = by_pair.get(pair)
        if current is not None and current.status != API_STATUS_FAILURE:
            continue
        if client is None:
            client = client_factory(config)
        by_pair[pair] = _call_deepseek(
            sample=sample,
            config=config,
            prompt_dir=prompt_dir,
            client=client,
        )
        _write_v2_predictions(Path(v2_predictions_path), list(by_pair.values()), samples=samples)

    records = list(by_pair.values())
    _write_v2_predictions(Path(v2_predictions_path), records, samples=samples)
    audit = audit_pilot_v2_predictions(records, sample_ids=sample_ids)
    _require_complete_audit(audit)
    return audit


def inherit_v1_predictions(
    *,
    v1_predictions_path: str | Path,
    sample_ids: Sequence[str],
) -> list[PilotV2Prediction]:
    records = load_pilot_predictions(v1_predictions_path)
    pair_counts = Counter(
        (record.id, record.model) for record in records if record.model in INHERITED_MODELS
    )
    expected_pairs = {(sample_id, model) for sample_id in sample_ids for model in INHERITED_MODELS}
    missing = sorted(expected_pairs - set(pair_counts))
    duplicates = sorted(pair for pair, count in pair_counts.items() if count > 1)
    unexpected = sorted(set(pair_counts) - expected_pairs)
    if missing:
        raise PilotDataError(f"Missing V1 inherited predictions: {_format_pairs(missing)}")
    if duplicates:
        raise PilotDataError(f"Duplicate V1 inherited predictions: {_format_pairs(duplicates)}")
    if unexpected:
        raise PilotDataError(f"Unexpected V1 inherited predictions: {_format_pairs(unexpected)}")

    by_pair = {(record.id, record.model): record for record in records}
    inherited: list[PilotV2Prediction] = []
    for sample_id in sample_ids:
        for model in INHERITED_MODELS:
            record = by_pair[(sample_id, model)]
            if record.status != "SUCCESS":
                raise PilotDataError(f"V1 inherited prediction is not SUCCESS: {sample_id}/{model}")
            inherited.append(_inherit_v1_record(record))
    return inherited


def load_pilot_v2_predictions(path: str | Path) -> list[PilotV2Prediction]:
    prediction_path = Path(path)
    if not prediction_path.exists():
        return []
    try:
        handle = prediction_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise PilotDataError(f"Cannot read Pilot V2 predictions: {prediction_path}") from exc

    records: list[PilotV2Prediction] = []
    with handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != PILOT_V2_PREDICTION_FIELDS:
            raise PilotDataError(
                "Pilot V2 prediction columns must be exactly "
                f"{PILOT_V2_PREDICTION_FIELDS}; got {actual_fields}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                record = PilotV2Prediction(
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
                    finish_reason=row["finish_reason"] or None,
                    completion_tokens=_parse_optional_int(row["completion_tokens"]),
                    reasoning_tokens=_parse_optional_int(row["reasoning_tokens"]),
                    reasoning_content_present=_parse_optional_bool(
                        row["reasoning_content_present"]
                    ),
                    reasoning_content_length=_parse_optional_int(row["reasoning_content_length"]),
                    max_completion_tokens=_parse_optional_int(row["max_completion_tokens"]),
                    source=row["source"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PilotDataError(f"Invalid Pilot V2 prediction row {row_number}") from exc
            _validate_v2_record(record, row_number=row_number)
            records.append(record)
    return records


def audit_pilot_v2_predictions(
    records: Sequence[PilotV2Prediction],
    *,
    sample_ids: Sequence[str],
) -> PilotAudit:
    expected_pairs = {(sample_id, model) for sample_id in sample_ids for model in PILOT_API_MODELS}
    pair_counts = Counter((record.id, record.model) for record in records)
    observed_pairs = set(pair_counts)
    status_counts = Counter(record.status for record in records)
    return PilotAudit(
        expected_combinations=len(expected_pairs),
        observed_rows=len(records),
        missing_combinations=tuple(
            _format_pair(pair) for pair in sorted(expected_pairs - observed_pairs)
        ),
        duplicate_combinations=tuple(
            _format_pair(pair) for pair, count in sorted(pair_counts.items()) if count > 1
        ),
        unexpected_combinations=tuple(
            _format_pair(pair) for pair in sorted(observed_pairs - expected_pairs)
        ),
        success=status_counts["SUCCESS"],
        invalid_output=status_counts[INVALID_OUTPUT],
        api_failure=status_counts[API_STATUS_FAILURE],
    )


def run_pilot_v2_stage_b(
    *,
    input_path: str | Path,
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    item_matrix_path: str | Path,
    summary_path: str | Path,
    report_path: str | Path,
) -> PilotAudit:
    samples = load_pilot_input(input_path)
    ordered_ids = [sample.id for sample in samples]
    records = load_pilot_v2_predictions(predictions_path)
    audit = audit_pilot_v2_predictions(records, sample_ids=ordered_ids)
    if audit.expected_combinations != PILOT_COMBINATION_COUNT:
        raise PilotDataError(
            f"Expected Pilot V2 combination count must be {PILOT_COMBINATION_COUNT}"
        )
    _require_complete_audit(audit)
    _validate_inherited_v2_records(records, ordered_ids)

    # Ground Truth remains inaccessible until all 72 Stage A combinations pass audit.
    ground_truth = _load_pilot_ground_truth(ground_truth_path)
    if list(ground_truth) != ordered_ids:
        raise PilotDataError("Pilot Ground Truth IDs or ordering do not match Stage A input")

    by_pair = {(record.id, record.model): record for record in records}
    matrix_rows = _build_v2_item_matrix(ordered_ids, ground_truth, by_pair)
    summary_rows = _build_v2_summary_rows(records, ground_truth)
    _write_csv_atomically(Path(item_matrix_path), PILOT_MATRIX_FIELDS, matrix_rows)
    _write_csv_atomically(Path(summary_path), PILOT_V2_SUMMARY_FIELDS, summary_rows)
    _write_v2_report(
        Path(report_path),
        audit=audit,
        records=records,
        matrix_rows=matrix_rows,
        summary_rows=summary_rows,
    )
    return audit


def _load_deepseek_v2_config(path: str | Path) -> ModelConfig:
    config = load_model_config(path, DEEPSEEK_MODEL_KEY)
    if config.api_model != DEEPSEEK_API_MODEL:
        raise PilotDataError(f"Unexpected DeepSeek API model: {config.api_model}")
    if config.max_tokens is not None:
        raise PilotDataError("DeepSeek V2 must not send max_tokens")
    if config.max_completion_tokens != DEEPSEEK_MAX_COMPLETION_TOKENS:
        raise PilotDataError(
            f"DeepSeek V2 max_completion_tokens must be {DEEPSEEK_MAX_COMPLETION_TOKENS}"
        )
    return config


def _call_deepseek(
    *,
    sample: TestSample,
    config: ModelConfig,
    prompt_dir: str | Path,
    client: MetadataCompletionClient,
) -> PilotV2Prediction:
    prompt = render_prompt("zero_shot", sample, (), prompt_dir=prompt_dir)
    started_at = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        result = client.complete_with_metadata(prompt)
        normalized = normalize_prediction(result.content)
        status = (
            "SUCCESS"
            if normalized in ALLOWED_LABELS and result.finish_reason != "length"
            else INVALID_OUTPUT
        )
        error_type = ""
    except ProviderRequestError as exc:
        result = CompletionResult(
            content="",
            reasoning_content=None,
            finish_reason=None,
            completion_tokens=None,
            reasoning_tokens=None,
        )
        normalized = ""
        status = API_STATUS_FAILURE
        error_type = f"{type(exc).__name__}: {exc}"

    reasoning_content = result.reasoning_content
    return PilotV2Prediction(
        id=sample.id,
        model=config.api_model,
        run=1,
        prompt_type="zero_shot",
        raw_output=result.content,
        normalized_output=normalized,
        status=status,
        error_type=error_type,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
        actual_temperature=config.temperature,
        timestamp=timestamp,
        finish_reason=result.finish_reason,
        completion_tokens=result.completion_tokens,
        reasoning_tokens=result.reasoning_tokens,
        reasoning_content_present=(
            reasoning_content is not None if status != API_STATUS_FAILURE else None
        ),
        reasoning_content_length=(
            len(reasoning_content) if reasoning_content is not None else None
        ),
        max_completion_tokens=config.max_completion_tokens,
        source="deepseek_v2_api",
    )


def _validate_smoke_result(record: PilotV2Prediction) -> None:
    if record.status != "SUCCESS":
        raise PilotDataError(
            "DeepSeek V2 smoke failed: "
            f"status={record.status}, finish_reason={record.finish_reason}, "
            f"content_empty={not bool(record.raw_output)}"
        )
    if record.normalized_output not in ALLOWED_LABELS or not record.raw_output:
        raise PilotDataError("DeepSeek V2 smoke did not return a valid final label")
    if record.finish_reason == "length":
        raise PilotDataError("DeepSeek V2 smoke exhausted the generation budget")
    if record.reasoning_tokens is None or record.reasoning_tokens <= 0:
        raise PilotDataError("DeepSeek V2 smoke did not record positive reasoning_tokens")


def _inherit_v1_record(record: PilotPrediction) -> PilotV2Prediction:
    return PilotV2Prediction(
        id=record.id,
        model=record.model,
        run=record.run,
        prompt_type=record.prompt_type,
        raw_output=record.raw_output,
        normalized_output=record.normalized_output,
        status=record.status,
        error_type=record.error_type,
        latency_ms=record.latency_ms,
        actual_temperature=record.actual_temperature,
        timestamp=record.timestamp,
        finish_reason=None,
        completion_tokens=None,
        reasoning_tokens=None,
        reasoning_content_present=None,
        reasoning_content_length=None,
        max_completion_tokens=None,
        source="inherited_v1",
    )


def _validate_v2_record(record: PilotV2Prediction, *, row_number: int) -> None:
    if record.model not in PILOT_API_MODELS:
        raise PilotDataError(f"Pilot V2 row {row_number} has unknown model {record.model!r}")
    if record.run != 1 or record.prompt_type != "zero_shot":
        raise PilotDataError(f"Pilot V2 row {row_number} has invalid experiment metadata")
    if record.source == "inherited_v1":
        if record.model not in INHERITED_MODELS or record.status != "SUCCESS":
            raise PilotDataError(f"Invalid inherited V2 row {row_number}")
        metadata = (
            record.finish_reason,
            record.completion_tokens,
            record.reasoning_tokens,
            record.reasoning_content_present,
            record.reasoning_content_length,
            record.max_completion_tokens,
        )
        if any(value is not None for value in metadata):
            raise PilotDataError(f"Inherited V2 row {row_number} fabricates unavailable metadata")
        return
    if record.source != "deepseek_v2_api" or record.model != DEEPSEEK_API_MODEL:
        raise PilotDataError(f"Pilot V2 row {row_number} has invalid source")
    if record.max_completion_tokens != DEEPSEEK_MAX_COMPLETION_TOKENS:
        raise PilotDataError(f"Pilot V2 row {row_number} has invalid DeepSeek budget")
    if record.status == "SUCCESS":
        if record.normalized_output not in ALLOWED_LABELS or not record.raw_output:
            raise PilotDataError(f"Invalid SUCCESS V2 row {row_number}")
    elif record.status == INVALID_OUTPUT:
        if not record.raw_output and record.finish_reason != "length":
            raise PilotDataError(f"Invalid INVALID_OUTPUT V2 row {row_number}")
    elif record.status == API_STATUS_FAILURE:
        if record.raw_output or record.normalized_output or not record.error_type:
            raise PilotDataError(f"Invalid API_FAILURE V2 row {row_number}")
    else:
        raise PilotDataError(f"Pilot V2 row {row_number} has invalid status {record.status!r}")


def _validate_v2_pairs(
    records: Sequence[PilotV2Prediction],
    *,
    sample_ids: Sequence[str],
) -> None:
    allowed_pairs = {(sample_id, model) for sample_id in sample_ids for model in PILOT_API_MODELS}
    counts = Counter((record.id, record.model) for record in records)
    duplicates = sorted(pair for pair, count in counts.items() if count > 1)
    unexpected = sorted(set(counts) - allowed_pairs)
    if duplicates:
        raise PilotDataError(f"Duplicate Pilot V2 combinations: {_format_pairs(duplicates)}")
    if unexpected:
        raise PilotDataError(f"Unexpected Pilot V2 combinations: {_format_pairs(unexpected)}")


def _validate_inherited_v2_records(
    records: Sequence[PilotV2Prediction],
    sample_ids: Sequence[str],
) -> None:
    inherited = [record for record in records if record.model in INHERITED_MODELS]
    expected = len(sample_ids) * len(INHERITED_MODELS)
    if len(inherited) != expected or any(
        record.source != "inherited_v1" or record.status != "SUCCESS" for record in inherited
    ):
        raise PilotDataError("Pilot V2 Qwen/GLM inheritance validation failed")


def _write_v2_predictions(
    path: Path,
    records: Sequence[PilotV2Prediction],
    *,
    samples: Sequence[TestSample],
) -> None:
    order = {
        (sample.id, model): index
        for index, (sample, model) in enumerate(
            (sample, model) for sample in samples for model in PILOT_API_MODELS
        )
    }
    ordered = sorted(records, key=lambda record: order[(record.id, record.model)])
    _write_csv_atomically(
        path,
        PILOT_V2_PREDICTION_FIELDS,
        [record.to_csv_row() for record in ordered],
    )


def _build_v2_item_matrix(
    ordered_ids: Sequence[str],
    ground_truth: dict[str, PilotGroundTruth],
    by_pair: dict[tuple[str, str], PilotV2Prediction],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id in ordered_ids:
        truth = ground_truth[sample_id]
        displayed = [_display_prediction(by_pair[(sample_id, model)]) for model in PILOT_API_MODELS]
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


def _build_v2_summary_rows(
    records: Sequence[PilotV2Prediction],
    ground_truth: dict[str, PilotGroundTruth],
) -> list[dict[str, object]]:
    groups: list[tuple[str, str, str, str, list[PilotV2Prediction]]] = [
        ("overall", "ALL", "ALL", "ALL", list(records))
    ]
    groups.extend(
        ("model", model, "ALL", "ALL", [r for r in records if r.model == model])
        for model in PILOT_API_MODELS
    )
    groups.extend(
        (
            "difficulty",
            "ALL",
            difficulty,
            "ALL",
            [r for r in records if ground_truth[r.id].difficulty == difficulty],
        )
        for difficulty in ("Medium", "Hard")
    )
    groups.extend(
        (
            "class",
            "ALL",
            "ALL",
            label,
            [r for r in records if ground_truth[r.id].label == label],
        )
        for label in sorted(ALLOWED_LABELS)
    )
    groups.extend(
        (
            "model_difficulty",
            model,
            difficulty,
            "ALL",
            [
                r
                for r in records
                if r.model == model and ground_truth[r.id].difficulty == difficulty
            ],
        )
        for model in PILOT_API_MODELS
        for difficulty in ("Medium", "Hard")
    )
    groups.extend(
        (
            "model_class",
            model,
            "ALL",
            label,
            [r for r in records if r.model == model and ground_truth[r.id].label == label],
        )
        for model in PILOT_API_MODELS
        for label in sorted(ALLOWED_LABELS)
    )

    rows: list[dict[str, object]] = []
    for scope, model, difficulty, label, group in groups:
        counts = Counter(record.status for record in group)
        denominator = counts["SUCCESS"] + counts[INVALID_OUTPUT]
        correct = sum(
            record.status == "SUCCESS" and record.normalized_output == ground_truth[record.id].label
            for record in group
        )
        rows.append(
            {
                "scope": scope,
                "model": model,
                "difficulty": difficulty,
                "ground_truth": label,
                "total": len(group),
                "success": counts["SUCCESS"],
                "invalid_output": counts[INVALID_OUTPUT],
                "api_failure": counts[API_STATUS_FAILURE],
                "correct": correct,
                "accuracy": f"{correct / denominator:.6f}" if denominator else "",
            }
        )
    return rows


def _write_v2_report(
    path: Path,
    *,
    audit: PilotAudit,
    records: Sequence[PilotV2Prediction],
    matrix_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
) -> None:
    model_rows = [row for row in summary_rows if row["scope"] == "model"]
    disagreement_ids = [
        str(row["id"]) for row in matrix_rows if row["model_disagreement"] == "true"
    ]
    deepseek = [record for record in records if record.model == DEEPSEEK_API_MODEL]
    finish_reasons = Counter(record.finish_reason or "<missing>" for record in deepseek)
    completion_values = [
        record.completion_tokens for record in deepseek if record.completion_tokens is not None
    ]
    reasoning_values = [
        record.reasoning_tokens for record in deepseek if record.reasoning_tokens is not None
    ]
    smoke = next(record for record in deepseek if record.id == "P001")
    lines = [
        "# Difficulty Pilot V2 Execution Report",
        "",
        "## Execution status",
        "",
        "- Smoke Test: COMPLETE (P001 × deepseek-v4-pro)",
        "- Stage A: COMPLETE",
        "- Stage B: COMPLETE",
        f"- Expected `(sample, model)` combinations: {audit.expected_combinations}",
        f"- Observed rows: {audit.observed_rows}",
        f"- Missing combinations: {len(audit.missing_combinations)}",
        f"- Duplicate combinations: {len(audit.duplicate_combinations)}",
        f"- SUCCESS: {audit.success}",
        f"- INVALID_OUTPUT: {audit.invalid_output}",
        f"- API_FAILURE: {audit.api_failure}",
        "- Qwen inherited from V1: 24 SUCCESS, 0 API calls",
        "- GLM inherited from V1: 24 SUCCESS, 0 API calls",
        "- DeepSeek V2 API calls represented: 24",
        "",
        "## DeepSeek Smoke Test",
        "",
        f"- normalized_output: {smoke.normalized_output}",
        f"- finish_reason: {smoke.finish_reason}",
        f"- completion_tokens: {smoke.completion_tokens}",
        f"- reasoning_tokens: {smoke.reasoning_tokens}",
        f"- reasoning_content_present: {str(smoke.reasoning_content_present).lower()}",
        f"- reasoning_content_length: {smoke.reasoning_content_length}",
        f"- max_completion_tokens: {smoke.max_completion_tokens}",
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
            "## DeepSeek response metadata",
            "",
            f"- finish_reason counts: {dict(sorted(finish_reasons.items()))}",
            f"- completion_tokens range: {_range_text(completion_values)}",
            f"- reasoning_tokens range: {_range_text(reasoning_values)}",
            "- reasoning_content was diagnostic metadata only; classification used content.",
            "",
            "## Item-level observations",
            "",
            f"- Model-disagreement items ({len(disagreement_ids)}): "
            + (", ".join(disagreement_ids) if disagreement_ids else "none"),
            "- No sample text, Ground Truth, difficulty, Prompt, taxonomy, Qwen prediction, "
            "or GLM prediction was modified.",
            "",
        ]
    )
    _write_text_atomically(path, "\n".join(lines))


def _display_prediction(record: PilotV2Prediction) -> str:
    return API_STATUS_FAILURE if record.status == API_STATUS_FAILURE else record.normalized_output


def _parse_optional_int(value: str) -> int | None:
    return int(value) if value else None


def _parse_optional_bool(value: str) -> bool | None:
    if not value:
        return None
    if value not in {"true", "false"}:
        raise ValueError(f"Invalid optional boolean: {value!r}")
    return value == "true"


def _format_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _format_pairs(pairs: Sequence[tuple[str, str]]) -> str:
    return ", ".join(_format_pair(pair) for pair in pairs)


def _range_text(values: Sequence[int]) -> str:
    return f"{min(values)}–{max(values)}" if values else "unavailable"
