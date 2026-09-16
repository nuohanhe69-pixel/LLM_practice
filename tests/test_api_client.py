from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_experiment.api_client import (
    OpenAICompatibleClient,
    ProviderConfigurationError,
    build_openai_client,
)
from llm_experiment.config import ModelConfig


def model_config() -> ModelConfig:
    return ModelConfig(
        name="qwen_test",
        api_model="qwen-test",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        temperature=0.0,
        max_tokens=8,
        timeout_seconds=12.5,
        max_retries=2,
    )


def test_build_openai_client_uses_configured_credentials_timeout_and_retries(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSDKClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("llm_experiment.api_client.OpenAI", FakeSDKClient)

    client = build_openai_client(model_config(), environ={"TEST_API_KEY": "secret-value"})

    assert isinstance(client, OpenAICompatibleClient)
    assert captured == {
        "api_key": "secret-value",
        "base_url": "https://example.invalid/v1",
        "timeout": 12.5,
        "max_retries": 2,
    }


def test_build_openai_client_requires_environment_api_key():
    with pytest.raises(ProviderConfigurationError, match="TEST_API_KEY"):
        build_openai_client(model_config(), environ={})


def test_openai_compatible_client_sends_configured_generation_parameters():
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="NETWORK_API"))]
            )

    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleClient(model_config(), sdk_client)

    result = client.complete("classify this")

    assert result == "NETWORK_API"
    assert captured == {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "classify this"}],
        "temperature": 0.0,
        "max_tokens": 8,
    }
