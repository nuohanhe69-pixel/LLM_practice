from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from llm_experiment.difficulty_pilot import (
    PILOT_MODEL_KEYS,
    run_pilot_stage_a,
    run_pilot_stage_b,
)

PILOT_ROOT = Path("experiments/difficulty_pilot_v1")
DEFAULT_INPUT = PILOT_ROOT / "data/difficulty_pilot_input_v1.csv"
DEFAULT_GROUND_TRUTH = PILOT_ROOT / "data/difficulty_pilot_ground_truth_v1.csv"
DEFAULT_ARTIFACTS = PILOT_ROOT / "artifacts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Difficulty Pilot V1")
    parser.add_argument("command", choices=("smoke", "stage-a", "stage-b"))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--model-config", default="configs/models.json")
    parser.add_argument("--prompt-dir", default="prompts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir)
    if args.command == "smoke":
        audit = run_pilot_stage_a(
            input_path=args.input,
            predictions_path=artifacts_dir / "smoke/pilot_predictions_v1.csv",
            model_config_path=args.model_config,
            prompt_dir=args.prompt_dir,
            model_keys=PILOT_MODEL_KEYS[:1],
            sample_limit=1,
        )
    elif args.command == "stage-a":
        audit = run_pilot_stage_a(
            input_path=args.input,
            predictions_path=artifacts_dir / "pilot_predictions_v1.csv",
            model_config_path=args.model_config,
            prompt_dir=args.prompt_dir,
        )
    else:
        audit = run_pilot_stage_b(
            input_path=args.input,
            predictions_path=artifacts_dir / "pilot_predictions_v1.csv",
            ground_truth_path=args.ground_truth,
            item_matrix_path=artifacts_dir / "pilot_item_matrix_v1.csv",
            summary_path=artifacts_dir / "pilot_summary_v1.csv",
            report_path=artifacts_dir / "pilot_execution_report_v1.md",
            smoke_predictions_path=artifacts_dir / "smoke/pilot_predictions_v1.csv",
        )
    print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
