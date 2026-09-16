from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from openai import APIError, OpenAI

from llm_experiment.config import ModelConfig


class ProviderConfigurationError(ValueError):
    """Raised when the provider cannot be initialized safely."""


class ProviderRequestError(RuntimeError):
    """Raised when an expected provider request failure exhausts SDK retries."""


class OpenAICompatibleClient:
    """Small provider boundary around an OpenAI-compatible chat client."""

    def __init__(self, config: ModelConfig, sdk_client: Any) -> None:
        self._config = config
        self._sdk_client = sdk_client

    def complete(self, prompt: str) -> str:
        try:
            response = self._sdk_client.chat.completions.create(
                model=self._config.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except (APIError, TimeoutError, ConnectionError) as exc:
            raise ProviderRequestError(f"{type(exc).__name__}: {exc}") from exc
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""


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
