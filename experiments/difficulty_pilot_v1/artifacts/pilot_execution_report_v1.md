# Difficulty Pilot V1 Execution Report

## Execution status

- Smoke Test: COMPLETE (1/1)
- Stage A: COMPLETE
- Stage B: COMPLETE
- Expected `(sample, model)` combinations: 72
- Observed rows: 72
- Missing combinations: 0
- Duplicate combinations: 0
- Unexpected combinations: 0
- SUCCESS: 48
- INVALID_OUTPUT: 24
- API_FAILURE: 0

## Per-model results

| Model | Total | SUCCESS | INVALID_OUTPUT | API_FAILURE | Correct | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.7-plus | 24 | 24 | 0 | 0 | 22 | 0.916667 |
| glm-5 | 24 | 24 | 0 | 0 | 23 | 0.958333 |
| deepseek-v4-pro | 24 | 0 | 24 | 0 | 0 | 0.000000 |

## Item-level observations

- Model-disagreement items (24): P001, P002, P003, P004, P005, P006, P007, P008, P009, P010, P011, P012, P013, P014, P015, P016, P017, P018, P019, P020, P021, P022, P023, P024
- Three-model-correct items (0): none
- These are execution observations only. No sample text, Ground Truth, difficulty, Prompt, taxonomy, or formal experiment configuration was modified.
- Disagreement and unanimous correctness are review signals, not automatic evidence of ambiguity, shortcut, or a required rewrite.

## Invalid-output observations

- deepseek-v4-pro: 24 INVALID_OUTPUT (empty raw_output: 24, non-empty raw_output: 0)
- These successful API responses remain frozen and were not retried.
