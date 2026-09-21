# Difficulty Pilot V2 Execution Report

## Execution status

- Smoke Test: COMPLETE (P001 × deepseek-v4-pro)
- Stage A: COMPLETE
- Stage B: COMPLETE
- Expected `(sample, model)` combinations: 72
- Observed rows: 72
- Missing combinations: 0
- Duplicate combinations: 0
- SUCCESS: 72
- INVALID_OUTPUT: 0
- API_FAILURE: 0
- Qwen inherited from V1: 24 SUCCESS, 0 API calls
- GLM inherited from V1: 24 SUCCESS, 0 API calls
- DeepSeek V2 API calls represented: 24

## DeepSeek Smoke Test

- normalized_output: CODE_RUNTIME
- finish_reason: stop
- completion_tokens: 409
- reasoning_tokens: 404
- reasoning_content_present: true
- reasoning_content_length: 1175
- max_completion_tokens: 2048

## Per-model results

| Model | Total | SUCCESS | INVALID_OUTPUT | API_FAILURE | Correct | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.7-plus | 24 | 24 | 0 | 0 | 22 | 0.916667 |
| glm-5 | 24 | 24 | 0 | 0 | 23 | 0.958333 |
| deepseek-v4-pro | 24 | 24 | 0 | 0 | 24 | 1.000000 |

## DeepSeek response metadata

- finish_reason counts: {'stop': 24}
- completion_tokens range: 53–503
- reasoning_tokens range: 46–497
- reasoning_content was diagnostic metadata only; classification used content.

## Item-level observations

- Model-disagreement items (3): P002, P019, P023
- No sample text, Ground Truth, difficulty, Prompt, taxonomy, Qwen prediction, or GLM prediction was modified.
