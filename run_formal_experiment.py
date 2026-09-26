"""Run the frozen formal matrix serially using the existing resumable pipeline."""

from itertools import product

from llm_experiment.api_client import ProviderFatalError
from llm_experiment.pipeline import run_experiment
from llm_experiment.prediction import load_prediction_records

MODELS = (
    "qwen3_7_plus",
    "glm_5",
    "deepseek_v4_pro",
    "deepseek_v4_1_flash",
    "kimi_k3",
)
PROMPTS = ("zero_shot", "few_shot")
RUNS = (1, 2)
FORMAL_MATRIX = tuple(product(MODELS, PROMPTS, RUNS))


def main() -> int:
    anomalies: list[tuple[str, str]] = []
    for model, prompt, run_id in FORMAL_MATRIX:
        configuration = f"{model}/{prompt}/run_{run_id}"
        try:
            result = run_experiment(
                model_name=model,
                prompt_type=prompt,
                run_id=run_id,
                limit=None,
            )
        except ProviderFatalError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            anomalies.append((configuration, detail))
            print(f"{configuration}: ERROR {detail}", flush=True)
            continue

        records = load_prediction_records(result.predictions_path)
        length_count = sum(record.finish_reason == "length" for record in records)
        api_failures = result.metrics["api_failures"]
        detail = f"api_failures={api_failures}, length={length_count}"
        if api_failures or length_count:
            anomalies.append((configuration, detail))
            status = "WARNING"
        else:
            status = "OK"
        print(f"{configuration}: {status} {detail}", flush=True)

    if anomalies:
        print(f"FORMAL EXPERIMENT INCOMPLETE: {len(anomalies)} configuration(s) with anomalies")
        for configuration, detail in anomalies:
            print(f"  {configuration}: {detail}")
        return 1
    print("FORMAL EXPERIMENT COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
