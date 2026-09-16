from __future__ import annotations

import pytest

from llm_experiment.dataset import load_dataset
from llm_experiment.prompts import PromptTemplateError, render_prompt
from tests.helpers import write_protocol_dataset


def test_zero_shot_prompt_contains_definitions_and_only_current_test_text(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    bundle = load_dataset(dataset_path)

    prompt = render_prompt("zero_shot", bundle.tests[0], bundle.demos, prompt_dir="prompts")

    assert "NETWORK_API：" in prompt
    assert "test error 1" in prompt
    assert "demo error 1" not in prompt
    assert "TEST_REASON_1" not in prompt
    assert "test-source-1" not in prompt


def test_few_shot_prompt_contains_all_eight_demo_pairs_without_metadata(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    bundle = load_dataset(dataset_path)

    prompt = render_prompt("few_shot", bundle.tests[0], bundle.demos, prompt_dir="prompts")

    assert prompt.count("示例 ") == 8
    for demo in bundle.demos:
        assert demo.error_text in prompt
        assert f"标签：{demo.label}" in prompt
    assert "DEMO_REASON_1" not in prompt
    assert "demo-source-1" not in prompt
    assert "TEST_REASON_1" not in prompt


def test_prompt_renderer_rejects_unknown_prompt_type(tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    bundle = load_dataset(dataset_path)

    with pytest.raises(PromptTemplateError, match="prompt type"):
        render_prompt("unknown", bundle.tests[0], bundle.demos, prompt_dir="prompts")
