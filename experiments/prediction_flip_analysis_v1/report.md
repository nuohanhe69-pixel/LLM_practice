# Prediction Flip Analysis V1

20 formal configurations pair into 600 model × run × Test-sample Zero-shot/Few-shot comparisons. All input validations passed; per-run and combined accuracy deltas match Formal Analysis V1.

| Model | Run | C→C | C→W | W→C | W→W | Accuracy Δ (Few−Zero) |
|---|---:|---:|---:|---:|---:|---:|
| qwen3_7_plus | 1 | 30 | 9 | 1 | 20 | -0.133 |
| qwen3_7_plus | 2 | 29 | 7 | 3 | 21 | -0.067 |
| qwen3_7_plus | combined | 59 | 16 | 4 | 41 | -0.100 |
| glm_5 | 1 | 31 | 3 | 4 | 22 | +0.017 |
| glm_5 | 2 | 29 | 5 | 2 | 24 | -0.050 |
| glm_5 | combined | 60 | 8 | 6 | 46 | -0.017 |
| deepseek_v4_pro | 1 | 23 | 4 | 3 | 30 | -0.017 |
| deepseek_v4_pro | 2 | 21 | 8 | 3 | 28 | -0.083 |
| deepseek_v4_pro | combined | 44 | 12 | 6 | 58 | -0.050 |
| deepseek_v4_1_flash | 1 | 46 | 3 | 4 | 7 | +0.017 |
| deepseek_v4_1_flash | 2 | 45 | 3 | 3 | 9 | +0.000 |
| deepseek_v4_1_flash | combined | 91 | 6 | 7 | 16 | +0.008 |
| kimi_k3 | 1 | 48 | 7 | 0 | 5 | -0.117 |
| kimi_k3 | 2 | 50 | 5 | 0 | 5 | -0.083 |
| kimi_k3 | combined | 98 | 12 | 0 | 10 | -0.100 |

## Correct → wrong cases

Counts below include both runs. Predicted destination is the Few-shot prediction. A directed boundary denotes the item's GT → primary distractor, not necessarily the predicted destination.

### qwen3_7_plus (16 cases)

- GT: CODE_RUNTIME=6, CONTEXT_LIMIT=8, ENV_DEPENDENCY=2
- Primary distractor: CODE_RUNTIME=4, CONTEXT_LIMIT=2, ENV_DEPENDENCY=4, NETWORK_API=6
- Wrong prediction destination: CODE_RUNTIME=4, CONTEXT_LIMIT=2, ENV_DEPENDENCY=3, NETWORK_API=7
- Directed boundary: CODE_RUNTIME→ENV_DEPENDENCY=3, CODE_RUNTIME→NETWORK_API=3, CONTEXT_LIMIT→CODE_RUNTIME=4, CONTEXT_LIMIT→ENV_DEPENDENCY=1, CONTEXT_LIMIT→NETWORK_API=3, ENV_DEPENDENCY→CONTEXT_LIMIT=2

### glm_5 (8 cases)

- GT: CODE_RUNTIME=2, CONTEXT_LIMIT=4, ENV_DEPENDENCY=1, NETWORK_API=1
- Primary distractor: CODE_RUNTIME=3, CONTEXT_LIMIT=1, NETWORK_API=4
- Wrong prediction destination: CODE_RUNTIME=2, CONTEXT_LIMIT=1, NETWORK_API=5
- Directed boundary: CODE_RUNTIME→NETWORK_API=2, CONTEXT_LIMIT→CODE_RUNTIME=2, CONTEXT_LIMIT→NETWORK_API=2, ENV_DEPENDENCY→CODE_RUNTIME=1, NETWORK_API→CONTEXT_LIMIT=1

### deepseek_v4_pro (12 cases)

- GT: CODE_RUNTIME=4, CONTEXT_LIMIT=5, ENV_DEPENDENCY=3
- Primary distractor: CODE_RUNTIME=5, CONTEXT_LIMIT=2, ENV_DEPENDENCY=3, NETWORK_API=2
- Wrong prediction destination: CODE_RUNTIME=4, CONTEXT_LIMIT=2, ENV_DEPENDENCY=3, NETWORK_API=3
- Directed boundary: CODE_RUNTIME→CONTEXT_LIMIT=1, CODE_RUNTIME→ENV_DEPENDENCY=2, CODE_RUNTIME→NETWORK_API=1, CONTEXT_LIMIT→CODE_RUNTIME=3, CONTEXT_LIMIT→ENV_DEPENDENCY=1, CONTEXT_LIMIT→NETWORK_API=1, ENV_DEPENDENCY→CODE_RUNTIME=2, ENV_DEPENDENCY→CONTEXT_LIMIT=1

### deepseek_v4_1_flash (6 cases)

- GT: CODE_RUNTIME=2, ENV_DEPENDENCY=4
- Primary distractor: CODE_RUNTIME=2, CONTEXT_LIMIT=2, ENV_DEPENDENCY=1, NETWORK_API=1
- Wrong prediction destination: CODE_RUNTIME=3, CONTEXT_LIMIT=1, ENV_DEPENDENCY=1, NETWORK_API=1
- Directed boundary: CODE_RUNTIME→CONTEXT_LIMIT=1, CODE_RUNTIME→ENV_DEPENDENCY=1, ENV_DEPENDENCY→CODE_RUNTIME=2, ENV_DEPENDENCY→CONTEXT_LIMIT=1, ENV_DEPENDENCY→NETWORK_API=1

### kimi_k3 (12 cases)

- GT: CODE_RUNTIME=3, ENV_DEPENDENCY=9
- Primary distractor: CODE_RUNTIME=3, CONTEXT_LIMIT=6, ENV_DEPENDENCY=1, NETWORK_API=2
- Wrong prediction destination: CODE_RUNTIME=9, CONTEXT_LIMIT=2, ENV_DEPENDENCY=1
- Directed boundary: CODE_RUNTIME→CONTEXT_LIMIT=2, CODE_RUNTIME→ENV_DEPENDENCY=1, ENV_DEPENDENCY→CODE_RUNTIME=3, ENV_DEPENDENCY→CONTEXT_LIMIT=4, ENV_DEPENDENCY→NETWORK_API=2

These are descriptive prediction changes on fixed Test items; they do not by themselves establish a mechanism for the prompt effect.
