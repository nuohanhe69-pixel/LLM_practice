from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigurationError(ValueError):
    """Raised when experiment configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    api_model: str
    base_url: str
    api_key_env: str
    temperature: float
    max_tokens: int | None
    max_completion_tokens: int | None
    timeout_seconds: float
    max_retries: int


def load_model_config(
    path: str | Path,
    model_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelConfig:
    if not SAFE_NAME_PATTERN.fullmatch(model_name):
        raise ConfigurationError("Model name must be a safe identifier")

    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot load model configuration: {config_path}") from exc

    models = document.get("models") if isinstance(document, dict) else None
    if not isinstance(models, dict) or model_name not in models:
        available = ", ".join(sorted(models)) if isinstance(models, dict) else ""
        suffix = f" Available models: {available}" if available else ""
        raise ConfigurationError(f"Unknown model {model_name!r}.{suffix}")

    raw = models[model_name]
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration for {model_name!r} must be an object")

    environment = os.environ if environ is None else environ
    base_url = _required_string(raw, "base_url")
    base_url_env = raw.get("base_url_env")
    if base_url_env is not None:
        if not isinstance(base_url_env, str) or not base_url_env.strip():
            raise ConfigurationError("base_url_env must be a non-empty string")
        base_url = environment.get(base_url_env, "").strip() or base_url
    if not base_url.startswith(("https://", "http://")):
        raise ConfigurationError("base_url must use http:// or https://")

    temperature = _required_number(raw, "temperature")
    max_tokens = _optional_integer(raw, "max_tokens", minimum=1)
    max_completion_tokens = _optional_integer(raw, "max_completion_tokens", minimum=1)
    if (max_tokens is None) == (max_completion_tokens is None):
        raise ConfigurationError(
            "Exactly one of max_tokens or max_completion_tokens must be configured"
        )
    timeout_seconds = _required_number(raw, "timeout_seconds")
    max_retries = _required_integer(raw, "max_retries", minimum=0)
    if not 0 <= temperature <= 2:
        raise ConfigurationError("temperature must be between 0 and 2")
    if timeout_seconds <= 0:
        raise ConfigurationError("timeout_seconds must be greater than zero")

    return ModelConfig(
        name=model_name,
        api_model=_required_string(raw, "api_model"),
        base_url=base_url,
        api_key_env=_required_string(raw, "api_key_env"),
        temperature=float(temperature),
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        timeout_seconds=float(timeout_seconds),
        max_retries=max_retries,
    )


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value.strip()


def _required_number(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key} must be a number")
    return float(value)


def _required_integer(raw: dict[str, Any], key: str, *, minimum: int) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{key} must be an integer >= {minimum}")
    return value


def _optional_integer(raw: dict[str, Any], key: str, *, minimum: int) -> int | None:
    if key not in raw:
        return None
    return _required_integer(raw, key, minimum=minimum)
