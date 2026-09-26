from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from llm_experiment.config import ModelConfig


class ProviderConfigurationError(ValueError):
    """Raised when the provider cannot be initialized safely."""


class ProviderRequestError(RuntimeError):
    """Raised when an expected provider request failure exhausts SDK retries."""


class ProviderFatalError(RuntimeError):
    """Raised when a provider error cannot recover by moving to another sample."""


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Provider response fields needed for classification and run diagnostics."""

    content: str
    reasoning_content: str | None
    finish_reason: str | None
    completion_tokens: int | None
    reasoning_tokens: int | None


class OpenAICompatibleClient:
    """Small provider boundary around an OpenAI-compatible chat client."""

    def __init__(self, config: ModelConfig, sdk_client: Any) -> None:
        self._config = config
        self._sdk_client = sdk_client

    def complete(self, prompt: str) -> str:
        return self.complete_with_metadata(prompt).content

    def complete_with_metadata(self, prompt: str) -> CompletionResult:
        # max_completion_tokens includes visible and reasoning tokens:
        # https://developers.openai.com/api/docs/guides/token-counting
        request_options: dict[str, Any] = {}
        if self._config.max_tokens is not None:
            request_options["max_tokens"] = self._config.max_tokens
        elif self._config.max_completion_tokens is not None:
            request_options["max_completion_tokens"] = self._config.max_completion_tokens
        if self._config.enable_thinking is not None:
            request_options["extra_body"] = {"enable_thinking": self._config.enable_thinking}
        try:
            response = self._sdk_client.chat.completions.create(
                model=self._config.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.temperature,
                **request_options,
            )
        except APIConnectionError as exc:
            raise ProviderRequestError(f"{type(exc).__name__}: {exc}") from exc
        except APIStatusError as exc:
            error_type = (
                ProviderRequestError
                if exc.status_code in {408, 429} or 500 <= exc.status_code < 600
                else ProviderFatalError
            )
            raise error_type(f"{type(exc).__name__}: {exc}") from exc
        except APIError as exc:
            raise ProviderFatalError(f"{type(exc).__name__}: {exc}") from exc
        except (TimeoutError, ConnectionError) as exc:
            raise ProviderRequestError(f"{type(exc).__name__}: {exc}") from exc
        choice = response.choices[0]
        message = choice.message
        content = message.content if isinstance(message.content, str) else ""
        reasoning_content = getattr(message, "reasoning_content", None)
        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        return CompletionResult(
            content=content,
            reasoning_content=(reasoning_content if isinstance(reasoning_content, str) else None),
            finish_reason=(choice.finish_reason if isinstance(choice.finish_reason, str) else None),
            completion_tokens=_optional_int(getattr(usage, "completion_tokens", None)),
            reasoning_tokens=_optional_int(getattr(completion_details, "reasoning_tokens", None)),
        )


def build_openai_client(
    config: ModelConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> OpenAICompatibleClient:
    environment = os.environ if environ is None else environ
    api_key = environment.get(config.api_key_env, "").strip()
    if not api_key:
        raise ProviderConfigurationError(
            f"Required API key environment variable is not set: {config.api_key_env}"
        )

    # Alibaba Cloud documents OpenAI SDK use with api_key and a regional base_url:
    # https://help.aliyun.com/en/model-studio/what-is-model-studio
    # The OpenAI SDK documents bounded max_retries and client-level timeout here:
    # https://github.com/openai/openai-python/blob/main/README.md#retries
    sdk_client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    return OpenAICompatibleClient(config, sdk_client)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
