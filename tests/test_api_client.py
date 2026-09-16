from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest
from openai import APIStatusError, APITimeoutError

from llm_experiment.api_client import (
    OpenAICompatibleClient,
    ProviderConfigurationError,
    ProviderFatalError,
    ProviderRequestError,
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


def api_status_error(status_code: int) -> APIStatusError:
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx2.Response(status_code, request=request)
    return APIStatusError(f"HTTP {status_code}", response=response, body=None)


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


def test_openai_compatible_client_wraps_provider_timeout():
    class FakeCompletions:
        def create(self, **kwargs):
            raise APITimeoutError(request=None)

    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleClient(model_config(), sdk_client)

    with pytest.raises(ProviderRequestError, match="APITimeoutError: Request timed out"):
        client.complete("classify this")


@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
def test_openai_compatible_client_wraps_transient_status(status_code):
    class FakeCompletions:
        def create(self, **kwargs):
            raise api_status_error(status_code)

    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleClient(model_config(), sdk_client)

    with pytest.raises(ProviderRequestError, match=f"APIStatusError: HTTP {status_code}"):
        client.complete("classify this")


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_openai_compatible_client_raises_fatal_status(status_code):
    class FakeCompletions:
        def create(self, **kwargs):
            raise api_status_error(status_code)

    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleClient(model_config(), sdk_client)

    with pytest.raises(ProviderFatalError, match=f"APIStatusError: HTTP {status_code}"):
        client.complete("classify this")


def test_openai_compatible_client_propagates_unexpected_program_error():
    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("program bug")

    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = OpenAICompatibleClient(model_config(), sdk_client)

    with pytest.raises(RuntimeError, match="program bug"):
        client.complete("classify this")
