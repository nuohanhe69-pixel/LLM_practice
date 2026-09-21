from __future__ import annotations

import json

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
    [{}, {"max_tokens": 16, "max_completion_tokens": 2048}],
)
def test_load_model_config_requires_exactly_one_generation_budget(tmp_path, budget_fields):
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

    with pytest.raises(ConfigurationError, match="Exactly one"):
        load_model_config(config_path, "test", environ={})


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
