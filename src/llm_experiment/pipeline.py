from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_experiment.api_client import build_openai_client
from llm_experiment.config import SAFE_NAME_PATTERN, load_model_config
from llm_experiment.dataset import load_frozen_dataset
from llm_experiment.evaluation import evaluate_predictions
from llm_experiment.prediction import CompletionClient, run_predictions
from llm_experiment.prompts import PROMPT_FILES


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    output_dir: Path
    predictions_path: Path
    metrics_path: Path
    error_cases_path: Path
    metrics: dict[str, object]


def run_experiment(
    *,
    model_name: str,
    prompt_type: str,
    run_id: int,
    dataset_path: str | Path = "dataset_v1.csv",
    model_config_path: str | Path = "configs/models.json",
    prompt_dir: str | Path = "prompts",
    results_root: str | Path = "results",
    limit: int | None = None,
    client: CompletionClient | None = None,
) -> ExperimentResult:
    model_config = load_model_config(model_config_path, model_name)
    bundle = load_frozen_dataset(dataset_path)
    if prompt_type not in PROMPT_FILES:
        raise ValueError(f"Unsupported prompt type: {prompt_type!r}")
    if run_id < 1:
        raise ValueError("run_id must be at least 1")
    if limit is not None and not 1 <= limit <= len(bundle.tests):
        raise ValueError(f"limit must be between 1 and {len(bundle.tests)}")

    samples = bundle.tests if limit is None else bundle.tests[:limit]
    output_dir = resolve_result_directory(
        results_root=results_root,
        model_name=model_config.name,
        prompt_type=prompt_type,
        run_id=run_id,
        limit=limit,
    )
    predictions_path = output_dir / "predictions.csv"
    completion_client = client if client is not None else build_openai_client(model_config)

    run_predictions(
        samples=samples,
        demos=bundle.demos,
        model_config=model_config,
        prompt_type=prompt_type,
        run_id=run_id,
        predictions_path=predictions_path,
        prompt_dir=prompt_dir,
        client=completion_client,
    )
    metrics = evaluate_predictions(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        output_dir=output_dir,
        expected_sample_ids=[sample.id for sample in samples],
    )
    return ExperimentResult(
        output_dir=output_dir,
        predictions_path=predictions_path,
        metrics_path=output_dir / "metrics.json",
        error_cases_path=output_dir / "error_cases.csv",
        metrics=metrics,
    )


def resolve_result_directory(
    *,
    results_root: str | Path,
    model_name: str,
    prompt_type: str,
    run_id: int,
    limit: int | None,
) -> Path:
    if not SAFE_NAME_PATTERN.fullmatch(model_name):
        raise ValueError("Model name must be a safe identifier")
    if prompt_type not in PROMPT_FILES:
        raise ValueError(f"Unsupported prompt type: {prompt_type!r}")
    if run_id < 1:
        raise ValueError("run_id must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    result = Path(results_root) / model_name / prompt_type / f"run_{run_id}"
    return result if limit is None else result / f"smoke_limit_{limit}"
