from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from llm_experiment.api_client import ProviderConfigurationError
from llm_experiment.config import ConfigurationError
from llm_experiment.dataset import DatasetValidationError
from llm_experiment.evaluation import EvaluationError
from llm_experiment.pipeline import run_experiment
from llm_experiment.prediction import PredictionStateError
from llm_experiment.prompts import PromptTemplateError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prediction and automatic evaluation for one experiment configuration."
    )
    parser.add_argument("--model", required=True, help="Model key from configs/models.json")
    parser.add_argument(
        "--prompt-type",
        required=True,
        choices=("zero_shot", "few_shot"),
        help="Prompt protocol to run",
    )
    parser.add_argument("--run-id", required=True, type=_positive_int, help="Positive run number")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Run the same pipeline on the first N test samples in an isolated smoke directory",
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset_v2.csv"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_experiment(
            model_name=args.model,
            prompt_type=args.prompt_type,
            run_id=args.run_id,
            limit=args.limit,
            dataset_path=args.dataset,
            model_config_path=args.model_config,
            prompt_dir=args.prompt_dir,
            results_root=args.results_dir,
        )
    except (
        ConfigurationError,
        DatasetValidationError,
        EvaluationError,
        PredictionStateError,
        PromptTemplateError,
        ProviderConfigurationError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    api_failures = result.metrics["api_failures"]
    if api_failures:
        print(f"Experiment incomplete: {api_failures} API failure(s) remain.")
        print("Re-run the same command to retry failed samples.")
    else:
        print("Experiment completed.")
    print(f"predictions: {result.predictions_path.resolve()}")
    print(f"metrics: {result.metrics_path.resolve()}")
    print(f"error cases: {result.error_cases_path.resolve()}")
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
