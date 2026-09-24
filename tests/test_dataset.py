from __future__ import annotations

import csv
import hashlib
from dataclasses import fields
from pathlib import Path

import pytest

from llm_experiment.constants import DATASET_FIELDS
from llm_experiment.dataset import (
    DatasetValidationError,
    dataset_fingerprint,
    load_dataset,
    load_frozen_dataset,
)
from llm_experiment.dataset import TestSample as DatasetTestSample
from tests.helpers import write_protocol_dataset


def write_modified_frozen_dataset(path, *, field, value):
    with Path("dataset_v2.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_frozen_dataset_fingerprint_accepts_reviewed_dataset():
    dataset_path = Path("dataset_v2.csv")
    bundle = load_frozen_dataset(dataset_path)

    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == (
        "3471f8572b9953f4de7e8a547fad88893265c3f304d7c2fd5fdf5b38afd4876e"
    )
    assert dataset_fingerprint(dataset_path) == (
        "7abb151f8a42bca101465011726ffcf6376fb20523e0dd9cdfc559dacf0610f0"
    )
    assert len(bundle.demos) == 8
    assert len(bundle.tests) == 60
    assert [demo.id for demo in bundle.demos] == [f"D{index:03d}" for index in range(1, 9)]
    assert [sample.id for sample in bundle.tests] == [f"T{index:03d}" for index in range(1, 61)]
    assert tuple(field.name for field in fields(DatasetTestSample)) == ("id", "error_text")


def test_previous_dataset_is_not_the_active_frozen_dataset():
    with pytest.raises(DatasetValidationError, match="fingerprint"):
        load_frozen_dataset("dataset_v1.csv")


def test_frozen_dataset_fingerprint_rejects_modified_error_text(tmp_path):
    dataset_path = write_modified_frozen_dataset(
        tmp_path / "dataset_v2.csv",
        field="error_text",
        value="modified error text",
    )

    with pytest.raises(DatasetValidationError, match="fingerprint"):
        load_frozen_dataset(dataset_path)


def test_frozen_dataset_fingerprint_rejects_modified_label(tmp_path):
    dataset_path = write_modified_frozen_dataset(
        tmp_path / "dataset_v2.csv",
        field="label",
        value="CONTEXT_LIMIT",
    )

    with pytest.raises(DatasetValidationError, match="fingerprint"):
        load_frozen_dataset(dataset_path)


def test_load_dataset_exposes_only_safe_test_fields(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")

    bundle = load_dataset(dataset_path)

    assert len(bundle.demos) == 8
    assert len(bundle.tests) == 60
    assert bundle.tests[0].id == "T001"
    assert bundle.tests[0].error_text == "test error 1"
    assert not hasattr(bundle.tests[0], "label")
    assert not hasattr(bundle.tests[0], "reason")
    assert not hasattr(bundle.tests[0], "source_type")
    assert not hasattr(bundle.tests[0], "source_reference")


def test_load_dataset_rejects_a_missing_required_column(tmp_path):
    dataset_path = tmp_path / "dataset_v1.csv"
    fields = [field for field in DATASET_FIELDS if field != "label"]
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

    with pytest.raises(DatasetValidationError, match="columns"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_an_unknown_label(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    content = dataset_path.read_text(encoding="utf-8")
    dataset_path.write_text(content.replace("CODE_RUNTIME", "UNKNOWN", 1), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="unknown label"):
        load_dataset(dataset_path)


def test_prediction_dataset_loader_does_not_use_test_ground_truth(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    content = dataset_path.read_text(encoding="utf-8")
    dataset_path.write_text(
        content.replace("T001,test error 1,CODE_RUNTIME", "T001,test error 1,LEAK_SENTINEL"),
        encoding="utf-8",
    )

    bundle = load_dataset(dataset_path)

    assert bundle.tests[0].id == "T001"
    assert not hasattr(bundle.tests[0], "label")


def test_load_dataset_preserves_error_text_exactly(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    content = dataset_path.read_text(encoding="utf-8")
    dataset_path.write_text(
        content.replace("test error 1", "  test error 1  ", 1),
        encoding="utf-8",
    )

    bundle = load_dataset(dataset_path)

    assert bundle.tests[0].error_text == "  test error 1  "


def test_load_dataset_rejects_wrong_protocol_counts(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    rows = dataset_path.read_text(encoding="utf-8").splitlines()
    dataset_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="60 test"):
        load_dataset(dataset_path)
