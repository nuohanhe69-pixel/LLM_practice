from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from llm_experiment.constants import (
    ALLOWED_LABELS,
    DATASET_FIELDS,
    EXPECTED_DEMO_COUNT,
    EXPECTED_TEST_COUNT,
)


class DatasetValidationError(ValueError):
    """Raised when the frozen dataset does not match the experiment protocol."""


@dataclass(frozen=True, slots=True)
class DemoSample:
    id: str
    error_text: str
    label: str


@dataclass(frozen=True, slots=True)
class TestSample:
    id: str
    error_text: str


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    demos: tuple[DemoSample, ...]
    tests: tuple[TestSample, ...]


def load_dataset(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    try:
        handle = dataset_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DatasetValidationError(f"Cannot read dataset: {dataset_path}") from exc

    demos: list[DemoSample] = []
    tests: list[TestSample] = []
    seen_ids: set[str] = set()

    with handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != DATASET_FIELDS:
            raise DatasetValidationError(
                f"Dataset columns must be exactly {DATASET_FIELDS}; got {actual_fields}"
            )

        for row_number, row in enumerate(reader, start=2):
            sample_id = (row["id"] or "").strip()
            error_text = (row["error_text"] or "").strip()
            label = (row["label"] or "").strip()
            split = (row["split"] or "").strip()

            if not sample_id or not error_text:
                raise DatasetValidationError(f"Row {row_number} has an empty id or error_text")
            if sample_id in seen_ids:
                raise DatasetValidationError(f"Duplicate sample id: {sample_id}")
            seen_ids.add(sample_id)
            if label not in ALLOWED_LABELS:
                raise DatasetValidationError(f"Row {row_number} has unknown label {label!r}")

            if split == "demo":
                demos.append(DemoSample(id=sample_id, error_text=error_text, label=label))
            elif split == "test":
                tests.append(TestSample(id=sample_id, error_text=error_text))
            else:
                raise DatasetValidationError(f"Row {row_number} has unsupported split {split!r}")

    if len(demos) != EXPECTED_DEMO_COUNT or len(tests) != EXPECTED_TEST_COUNT:
        raise DatasetValidationError(
            "Dataset must contain exactly "
            f"{EXPECTED_DEMO_COUNT} demo and {EXPECTED_TEST_COUNT} test rows; "
            f"got {len(demos)} demo and {len(tests)} test rows"
        )

    return DatasetBundle(demos=tuple(demos), tests=tuple(tests))
