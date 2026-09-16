from __future__ import annotations

import csv
from pathlib import Path

from llm_experiment.constants import ALLOWED_LABELS, DATASET_FIELDS


def write_protocol_dataset(path: Path) -> Path:
    labels = sorted(ALLOWED_LABELS)
    rows: list[dict[str, str]] = []
    for index in range(8):
        rows.append(
            {
                "id": f"D{index + 1:03d}",
                "error_text": f"demo error {index + 1}",
                "label": labels[index % len(labels)],
                "reason": f"DEMO_REASON_{index + 1}",
                "source_type": "fixture",
                "source_reference": f"demo-source-{index + 1}",
                "split": "demo",
            }
        )
    for index in range(60):
        rows.append(
            {
                "id": f"T{index + 1:03d}",
                "error_text": f"test error {index + 1}",
                "label": labels[index % len(labels)],
                "reason": f"TEST_REASON_{index + 1}",
                "source_type": "fixture",
                "source_reference": f"test-source-{index + 1}",
                "split": "test",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
