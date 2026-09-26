from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_experiment.config import ConfigurationError, load_model_config


def test_load_model_config_resolves_environment_base_url(tmp_path):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "qwen_test": {
                        "api_model": "qwen-test",
                        "base_url": "https://default.example/v1",
                        "base_url_env": "TEST_BASE_URL",
                        "api_key_env": "TEST_API_KEY",
                        "temperature": 0,
                        "max_tokens": 8,
                        "timeout_seconds": 5,
                        "max_retries": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_model_config(
        config_path,
        "qwen_test",
        environ={"TEST_BASE_URL": "https://workspace.example/v1"},
    )

    assert config.name == "qwen_test"
    assert config.api_model == "qwen-test"
    assert config.base_url == "https://workspace.example/v1"
    assert config.temperature == 0.0
    assert config.max_tokens == 8
    assert config.max_completion_tokens is None
    assert config.max_retries == 1


def test_load_model_config_accepts_max_completion_tokens(tmp_path):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "deepseek_test": {
                        "api_model": "deepseek-test",
                        "base_url": "https://default.example/v1",
                        "api_key_env": "TEST_API_KEY",
                        "temperature": 0,
                        "max_completion_tokens": 2048,
                        "timeout_seconds": 5,
                        "max_retries": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_model_config(config_path, "deepseek_test", environ={})

    assert config.max_tokens is None
    assert config.max_completion_tokens == 2048


@pytest.mark.parametrize(
    "budget_fields",
    [{"max_tokens": 16, "max_completion_tokens": 2048}],
)
def test_load_model_config_rejects_two_generation_budgets(tmp_path, budget_fields):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "test": {
                        "api_model": "test",
                        "base_url": "https://default.example/v1",
                        "api_key_env": "TEST_API_KEY",
                        "temperature": 0,
                        "timeout_seconds": 5,
                        "max_retries": 1,
                        **budget_fields,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="At most one"):
        load_model_config(config_path, "test", environ={})


@pytest.mark.parametrize("thinking", [None, True, False])
def test_load_model_config_accepts_no_generation_cap(tmp_path, thinking):
    raw = {
        "api_model": "test",
        "base_url": "https://default.example/v1",
        "api_key_env": "TEST_API_KEY",
        "temperature": 0,
        "timeout_seconds": 5,
        "max_retries": 1,
    }
    if thinking is not None:
        raw["enable_thinking"] = thinking
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"test": raw}}), encoding="utf-8")
    config = load_model_config(config_path, "test", environ={})
    assert config.max_tokens is None
    assert config.max_completion_tokens is None
    assert config.enable_thinking is thinking


@pytest.mark.parametrize("thinking", [None, "true", 1, 0])
def test_load_model_config_rejects_non_boolean_thinking(tmp_path, thinking):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "test": {
                        "api_model": "test",
                        "base_url": "https://default.example/v1",
                        "api_key_env": "TEST_API_KEY",
                        "temperature": 0,
                        "timeout_seconds": 5,
                        "max_retries": 1,
                        "enable_thinking": thinking,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="enable_thinking must be a boolean"):
        load_model_config(config_path, "test", environ={})


@pytest.mark.parametrize(
    "key,api_model",
    [
        ("qwen3_7_plus", "qwen3.7-plus-2026-05-26"),
        ("glm_5", "glm-5"),
        ("deepseek_v4_pro", "deepseek-v4-pro"),
        ("deepseek_v4_1_flash", "deepseek-v4.1-flash"),
        ("kimi_k3", "kimi-k3"),
    ],
)
def test_formal_model_protocol(key, api_model):
    path = Path(__file__).resolve().parents[1] / "configs/models.json"
    config = load_model_config(path, key, environ={})
    assert config.api_model == api_model
    assert config.enable_thinking is True
    assert config.max_tokens is None
    assert config.max_completion_tokens is None


def test_formal_model_set_contains_exactly_the_five_final_models():
    path = Path(__file__).resolve().parents[1] / "configs/models.json"
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    assert set(models) == {
        "qwen3_7_plus",
        "glm_5",
        "deepseek_v4_pro",
        "deepseek_v4_1_flash",
        "kimi_k3",
    }
    with pytest.raises(ConfigurationError, match="Unknown model"):
        load_model_config(path, "deepseek_v4_flash_0731", environ={})


def test_load_model_config_rejects_an_unknown_model(tmp_path):
    config_path = tmp_path / "models.json"
    config_path.write_text('{"models": {}}', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unknown model"):
        load_model_config(config_path, "missing")


def test_load_model_config_rejects_unsafe_model_name(tmp_path):
    config_path = tmp_path / "models.json"
    config_path.write_text('{"models": {}}', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="safe identifier"):
        load_model_config(config_path, "../outside")
