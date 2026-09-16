from __future__ import annotations

import pytest

from llm_experiment.dataset import load_dataset
from llm_experiment.prompts import PromptTemplateError, render_prompt
from tests.helpers import write_protocol_dataset

FROZEN_LABEL_DEFINITIONS = """NETWORK_API：网络连接、DNS、HTTP/API 请求或远端服务可用性问题。
CONTEXT_LIMIT：上下文窗口、Token 数量、输入长度或模型容量限制问题。
ENV_DEPENDENCY：运行环境、依赖包、模块、版本、设备或安装问题。
CODE_RUNTIME：代码语法、类型、取值、索引、逻辑或执行期异常，且不属于以上类别。"""


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
