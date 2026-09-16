# Spec: Reproducible LLM Classification Experiment Pipeline

## Objective

Build a reproducible Python pipeline that reads the frozen `dataset_v1.csv`, calls Alibaba Cloud Model Studio through its OpenAI-compatible API for zero-shot or few-shot classification, saves resumable predictions, and automatically evaluates them against ground truth. The implementation must not change the dataset, labels, or experiment protocol and must not run the formal 60-sample experiment in this phase.

## Tech Stack

- Python 3.10+
- `openai==3.14.0` for the OpenAI-compatible client
- `pytest==9.1.1` for tests
- `ruff==0.16.7` for linting and formatting checks
- Python standard-library CSV and JSON support for experiment artifacts

## Commands

- Install: `python -m pip install -e ".[dev]"`
- Test: `python -m pytest`
- Lint: `python -m ruff check .`
- Format check: `python -m ruff format --check .`
- Smoke run: `python run_experiment.py --model qwen_plus --prompt-type zero_shot --run-id 1 --limit 10`
- Formal run: `python run_experiment.py --model qwen_plus --prompt-type zero_shot --run-id 1`

## Project Structure

- `src/llm_experiment/`: dataset, prompt, API, prediction, evaluation, and orchestration modules
- `prompts/`: external prompt templates
- `configs/`: model/provider configuration
- `tests/`: unit and local integration tests with generated fixtures
- `docs/`: concise implementation and operating documentation
- `tasks/`: implementation plan and checklist
- `results/`: generated artifacts, excluded from Git

## Code Style

Use typed, small functions with explicit inputs and outputs. Keep provider calls behind a narrow client boundary so tests can use a fake without network access.

```python
def normalize_prediction(raw_output: str) -> str:
    normalized = " ".join(raw_output.strip().splitlines()).strip().upper()
    return normalized if normalized in ALLOWED_LABELS else "INVALID_OUTPUT"
```

## Testing Strategy

- Unit tests cover dataset validation, prompt isolation, normalization, retry result handling, evaluation metrics, and result-path isolation.
- A local integration test runs the same prediction/evaluation pipeline with a fake API client and a generated 8-demo/60-test dataset.
- Tests must not call an external API or modify the real dataset.
- The full suite, lint, format check, and CLI help command must pass before push.

## Boundaries

- Always: read API keys from environment variables; keep prompt templates external; validate the frozen protocol; write prediction state after each finalized sample; evaluate automatically after prediction.
- Ask first: change label definitions, ground truth, dataset rows, sample counts, or accuracy semantics.
- Never: commit secrets; put test ground truth fields into a test-sample prompt; infer labels from malformed model output; count API failures as classification errors; run the formal experiment in this phase.

## Success Criteria

- Zero-shot and few-shot use the same ordered test subset and few-shot uses exactly eight demo rows.
- Only `id` and `error_text` from each test row can reach prediction/prompt code.
- Outputs preserve raw model text and only normalize whitespace/newlines/case; all other malformed results become `INVALID_OUTPUT`.
- Bounded SDK retry and timeout settings produce `SUCCESS` or `API_FAILURE` rows with error details.
- Restarting an experiment skips `SUCCESS` rows, retries `API_FAILURE` rows in place, and never overwrites successful rows or creates duplicate sample IDs.
- Evaluation joins on `sample_id`, excludes API failures from accuracy denominators, counts invalid outputs as unsuccessful classifications, and writes all required artifacts.
- Smoke outputs cannot overwrite formal outputs.
- Documentation explains structure, data flow, isolation, resume behavior, model extension, and commands.

## Open Questions

- None.
