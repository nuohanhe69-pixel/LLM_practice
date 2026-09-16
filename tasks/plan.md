# Implementation Plan: LLM Classification Experiment Pipeline

## Overview

Implement the approved experiment protocol in four dependency-ordered slices: validated inputs, resumable prediction, automatic evaluation, and CLI/documentation. Each slice is verified before moving on and committed as an independent save point.

## Architecture Decisions

- Pass immutable `TestSample(id, error_text)` objects into prediction so test ground-truth fields cannot enter prompts by accident.
- Store model/provider settings in JSON and prompt text in template files so adding a model or revising a prompt does not change prediction control flow.
- Persist the complete CSV atomically after every attempt; `SUCCESS` rows are terminal checkpoints while `API_FAILURE` rows are retried and replaced in place on resume.
- Define accuracy as `correct_predictions / successful_predictions`, where API failures are excluded and invalid outputs remain incorrect successful API responses.
- Place limited runs under `smoke_limit_<N>/` beneath the run directory to prevent overlap with formal artifacts.

## Task List

### Phase 1: Inputs and prompts

- Task 1: Add project configuration and strict dataset/model loaders.
- Task 2: Add external zero-shot/few-shot templates and isolation tests.

### Checkpoint: Inputs

- Focused input and prompt tests pass.
- No real API calls occur.

### Phase 2: Prediction and evaluation

- Task 3: Add the OpenAI-compatible client boundary and resumable prediction writer.
- Task 4: Add automatic evaluation and artifact generation.

### Checkpoint: Core pipeline

- Focused prediction and evaluation tests pass.
- API failures and invalid outputs are distinguished.

### Phase 3: Orchestration and delivery

- Task 5: Add unified CLI and local end-to-end fake-client test.
- Task 6: Add README and required implementation report.
- Task 7: Run full verification, commit, and push.

### Checkpoint: Complete

- Full tests, lint, formatting, compile, and CLI help checks pass.
- No secrets or generated results are committed.
- Formal experiments remain unexecuted.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Ground truth leaks into test prompts | High | Narrow test sample type plus prompt-content tests |
| Interrupted CSV write corrupts resume state | High | Atomic temporary-file replacement after each row |
| API outages get counted as model errors | High | Separate `api_status`, `api_error`, and denominator rules |
| Smoke run overwrites a formal run | Medium | Dedicated `smoke_limit_<N>` directory |
| Accidental frozen dataset modification | High | Byte-for-byte import plus protocol and regression checks |

## Open Questions

- None.
