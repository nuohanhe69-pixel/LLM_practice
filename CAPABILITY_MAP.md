# Capability Map: LLM Classification Experiment Pipeline

| Module id | Responsibility | Depends on |
|---|---|---|
| experiment-inputs | Validate the frozen dataset, model configuration, and prompt templates | — |
| prediction | Render zero/few-shot prompts, call Model Studio, normalize outputs, and persist resumable predictions | experiment-inputs |
| evaluation | Join predictions to ground truth and generate metrics and error cases | experiment-inputs, prediction |
| orchestration | Provide one CLI that runs prediction and then evaluation in isolated result directories | prediction, evaluation |

Build order: `experiment-inputs` → `prediction` → `evaluation` → `orchestration`.
