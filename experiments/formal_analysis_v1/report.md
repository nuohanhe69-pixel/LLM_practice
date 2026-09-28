# Formal Analysis V1

Validated: 20 configurations × 60 Test items = 1200 predictions; API_FAILURE=0; finish_reason=length=0; 12 directed boundaries × 5 items.

All accuracies are proportions. Two-run means are arithmetic means; run gaps are absolute differences. No model ranking or significance testing is implied.

## Overall performance

| Model | Prompt | Run 1 | Run 2 | Mean | Run gap | Invalid outputs |
|---|---|---:|---:|---:|---:|---:|
| qwen3_7_plus | zero_shot | 0.650 | 0.600 | 0.625 | 0.050 | 0 |
| qwen3_7_plus | few_shot | 0.517 | 0.533 | 0.525 | 0.017 | 0 |
| glm_5 | zero_shot | 0.567 | 0.567 | 0.567 | 0.000 | 0 |
| glm_5 | few_shot | 0.583 | 0.517 | 0.550 | 0.067 | 0 |
| deepseek_v4_pro | zero_shot | 0.450 | 0.483 | 0.467 | 0.033 | 0 |
| deepseek_v4_pro | few_shot | 0.433 | 0.400 | 0.417 | 0.033 | 0 |
| deepseek_v4_1_flash | zero_shot | 0.817 | 0.800 | 0.808 | 0.017 | 0 |
| deepseek_v4_1_flash | few_shot | 0.833 | 0.800 | 0.817 | 0.033 | 0 |
| kimi_k3 | zero_shot | 0.917 | 0.917 | 0.917 | 0.000 | 0 |
| kimi_k3 | few_shot | 0.800 | 0.833 | 0.817 | 0.033 | 0 |

## Four GT classes

Two-run mean accuracy for each model × prompt × GT class; separate Run 1 and Run 2 values are in `artifacts/per_class_results.csv`.

| Model | Prompt | NETWORK_API | CONTEXT_LIMIT | ENV_DEPENDENCY | CODE_RUNTIME |
|---|---|---:|---:|---:|---:|
| qwen3_7_plus | zero_shot | 0.867 | 0.867 | 0.233 | 0.533 |
| qwen3_7_plus | few_shot | 0.900 | 0.600 | 0.200 | 0.400 |
| glm_5 | zero_shot | 0.900 | 0.833 | 0.233 | 0.300 |
| glm_5 | few_shot | 0.900 | 0.767 | 0.200 | 0.333 |
| deepseek_v4_pro | zero_shot | 0.867 | 0.533 | 0.233 | 0.233 |
| deepseek_v4_pro | few_shot | 0.900 | 0.433 | 0.133 | 0.200 |
| deepseek_v4_1_flash | zero_shot | 0.933 | 1.000 | 0.533 | 0.767 |
| deepseek_v4_1_flash | few_shot | 0.933 | 1.000 | 0.533 | 0.800 |
| kimi_k3 | zero_shot | 0.967 | 1.000 | 0.733 | 0.967 |
| kimi_k3 | few_shot | 0.967 | 1.000 | 0.433 | 0.867 |

## Directed boundaries

The table gives mean boundary accuracy across the ten model × prompt groups (both runs in each group). Lower values mean more errors on that directed boundary. Per-group Run 1, Run 2, mean and distractor hit rates are in `artifacts/boundary_results.csv`; blank hit rate means no errors.

| GT → primary distractor | Mean boundary accuracy |
|---|---:|
| NETWORK_API → CONTEXT_LIMIT | 0.910 |
| NETWORK_API → ENV_DEPENDENCY | 1.000 |
| NETWORK_API → CODE_RUNTIME | 0.830 |
| CONTEXT_LIMIT → NETWORK_API | 0.680 |
| CONTEXT_LIMIT → ENV_DEPENDENCY | 0.940 |
| CONTEXT_LIMIT → CODE_RUNTIME | 0.790 |
| ENV_DEPENDENCY → NETWORK_API | 0.140 |
| ENV_DEPENDENCY → CONTEXT_LIMIT | 0.580 |
| ENV_DEPENDENCY → CODE_RUNTIME | 0.320 |
| CODE_RUNTIME → NETWORK_API | 0.690 |
| CODE_RUNTIME → CONTEXT_LIMIT | 0.370 |
| CODE_RUNTIME → ENV_DEPENDENCY | 0.560 |

## Prompt change and run stability

| Model | Zero-shot mean | Few-shot mean | Few-shot delta | Zero-shot gap | Few-shot gap |
|---|---:|---:|---:|---:|---:|
| qwen3_7_plus | 0.625 | 0.525 | -0.100 | 0.050 | 0.017 |
| glm_5 | 0.567 | 0.550 | -0.017 | 0.000 | 0.067 |
| deepseek_v4_pro | 0.467 | 0.417 | -0.050 | 0.033 | 0.033 |
| deepseek_v4_1_flash | 0.808 | 0.817 | +0.008 | 0.017 | 0.033 |
| kimi_k3 | 0.917 | 0.817 | -0.100 | 0.000 | 0.033 |

## Confusion matrices and heatmaps

Ten 4×4 confusion matrices combine both runs and count only predictions in the four legal labels; INVALID_OUTPUT counts remain in overall_results.csv.
The Zero-shot and Few-shot heatmaps show two-run mean boundary accuracy (five models × twelve directed boundaries).
