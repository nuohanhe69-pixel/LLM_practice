from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from openai import APIStatusError, APITimeoutError

from llm_experiment.api_client import (
    CompletionResult,
    OpenAICompatibleClient,
    ProviderConfigurationError,
    ProviderFatalError,
    ProviderRequestError,
    build_openai_client,
)
from llm_experiment.config import ModelConfig, load_model_config


def model_config() -> ModelConfig:
    return ModelConfig(
        name="qwen_test",
        api_model="qwen-test",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        temperature=0.0,
        max_tokens=8,
        max_completion_tokens=None,
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
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="NETWORK_API"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
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


@pytest.mark.parametrize("thinking", [None, True, False])
def test_openai_compatible_client_omits_budget_and_respects_thinking(thinking):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="NETWORK_API"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    config = replace(model_config(), max_tokens=None, enable_thinking=thinking)
    client = OpenAICompatibleClient(
        config,
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    assert client.complete("classify this") == "NETWORK_API"
    expected = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "classify this"}],
        "temperature": 0.0,
    }
    if thinking is not None:
        expected["extra_body"] = {"enable_thinking": thinking}
    assert captured == expected


@pytest.mark.parametrize(
    "key",
    [
        "qwen3_7_plus",
        "glm_5",
        "deepseek_v4_pro",
        "deepseek_v4_flash_0731",
        "kimi_k3",
    ],
)
def test_formal_models_send_thinking_without_generation_cap(key):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="NETWORK_API"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    config = load_model_config(
        Path(__file__).resolve().parents[1] / "configs/models.json",
        key,
        environ={},
    )
    client = OpenAICompatibleClient(
        config,
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    client.complete("classify this")
    assert captured == {
        "model": config.api_model,
        "messages": [{"role": "user", "content": "classify this"}],
        "temperature": config.temperature,
        "extra_body": {"enable_thinking": True},
    }


def test_openai_compatible_client_returns_thinking_metadata():
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            details = SimpleNamespace(reasoning_tokens=37)
            usage = SimpleNamespace(completion_tokens=41, completion_tokens_details=details)
            message = SimpleNamespace(
                content="CONTEXT_LIMIT",
                reasoning_content="diagnostic reasoning",
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=usage,
            )

    config = ModelConfig(
        name="deepseek_test",
        api_model="deepseek-test",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        temperature=0.0,
        max_tokens=None,
        max_completion_tokens=2048,
        timeout_seconds=12.5,
        max_retries=2,
    )
    client = OpenAICompatibleClient(
        config,
        SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )

    result = client.complete_with_metadata("classify this")

    assert result == CompletionResult(
        content="CONTEXT_LIMIT",
        reasoning_content="diagnostic reasoning",
        finish_reason="stop",
        completion_tokens=41,
        reasoning_tokens=37,
    )
    assert captured["max_completion_tokens"] == 2048
    assert "max_tokens" not in captured


def test_openai_compatible_client_preserves_length_finish_reason():
    class FakeCompletions:
        def create(self, **kwargs):
            details = SimpleNamespace(reasoning_tokens=16)
            usage = SimpleNamespace(completion_tokens=16, completion_tokens_details=details)
            message = SimpleNamespace(content="", reasoning_content="still thinking")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="length")],
                usage=usage,
            )

    client = OpenAICompatibleClient(
        model_config(),
        SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )

    result = client.complete_with_metadata("classify this")

    assert result.content == ""
    assert result.finish_reason == "length"
    assert result.completion_tokens == 16
    assert result.reasoning_tokens == 16


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
