# Formal Analysis V1

Input is read-only: repository-root `dataset_v2.csv`, the 20 full `results/<model>/<prompt>/run_{1,2}/predictions.csv` files, and `experiments/challenge_audit_v1/v4/candidates_full_v4.jsonl`. The script checks the full 1200-row matrix, metadata agreement, status, finish reasons, and 12 directed boundaries before writing analysis outputs.

Run from the repository root (Python 3.10+ and `uv`):

```bash
uv run --no-project --with-requirements experiments/formal_analysis_v1/requirements.txt python experiments/formal_analysis_v1/analyze_results.py
```

`--no-project` keeps Matplotlib out of the formal model-calling environment. Outputs are in this directory: four summary CSVs, ten 4×4 confusion-matrix CSVs, two directed-boundary heatmaps (PDF and PNG), and `report.md`. Accuracy values are proportions. Confusion matrices combine both runs; their four columns count legal-label predictions only. INVALID_OUTPUT counts are retained in `overall_results.csv` and count as errors for accuracy and distractor-hit denominators. A blank distractor hit rate means no errors occurred in that group.
