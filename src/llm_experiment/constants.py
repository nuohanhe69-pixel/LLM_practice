from __future__ import annotations

ALLOWED_LABELS = frozenset(
    {
        "NETWORK_API",
        "CONTEXT_LIMIT",
        "ENV_DEPENDENCY",
        "CODE_RUNTIME",
    }
)

INVALID_OUTPUT = "INVALID_OUTPUT"
API_STATUS_SUCCESS = "SUCCESS"
API_STATUS_FAILURE = "API_FAILURE"

DATASET_FIELDS = (
    "id",
    "error_text",
    "label",
    "reason",
    "source_type",
    "source_reference",
    "split",
)

EXPECTED_DEMO_COUNT = 8
EXPECTED_TEST_COUNT = 60
