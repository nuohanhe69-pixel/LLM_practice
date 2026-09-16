from __future__ import annotations

from pathlib import Path

import pytest

from llm_experiment.dataset import load_dataset
from llm_experiment.prompts import PromptTemplateError, render_prompt
from tests.helpers import write_protocol_dataset

FROZEN_LABEL_DEFINITIONS = "\n".join(
    (
        "NETWORK_API：网络连接、远程 API/服务通信、网关/端点可用性、连接中断或超时。",
        (
            "CONTEXT_LIMIT：请求体大小、输入/Token 长度、序列长度、"
            "上下文窗口容量或明确的输入/请求大小上限。"
        ),
        "ENV_DEPENDENCY：缺少包/模块/系统库、依赖版本不兼容、解释器/运行时版本不匹配或依赖解析失败。",
        "CODE_RUNTIME：在运行环境已可用的前提下，由程序自身运行逻辑、参数或值处理导致的运行时错误。",
    )
)


def test_label_definitions_match_frozen_taxonomy():
    definitions = Path("prompts/label_definitions.txt").read_text(encoding="utf-8").strip()

    assert definitions == FROZEN_LABEL_DEFINITIONS


@pytest.mark.parametrize("prompt_type", ["zero_shot", "few_shot"])
def test_prompt_uses_frozen_label_taxonomy(prompt_type, tmp_path):
    dataset_path = write_protocol_dataset(tmp_path / "dataset_v1.csv")
    bundle = load_dataset(dataset_path)

    prompt = render_prompt(prompt_type, bundle.tests[0], bundle.demos, prompt_dir="prompts")

    assert FROZEN_LABEL_DEFINITIONS in prompt


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
