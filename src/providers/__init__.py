"""Provider adapters. Map a config `provider:` string to an adapter class."""

from __future__ import annotations

from typing import Any

from .base import Provider, ProviderResult
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .openai_compat import OpenAICompatProvider

_REGISTRY = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "openai_compat": OpenAICompatProvider,
}


def build_provider(model_cfg: dict[str, Any], run_cfg: dict[str, Any]) -> Provider:
    name = model_cfg["provider"]
    if name not in _REGISTRY:
        raise KeyError(f"unknown provider {name!r}; known: {list(_REGISTRY)}")
    return _REGISTRY[name](model_cfg, run_cfg)


__all__ = ["Provider", "ProviderResult", "build_provider"]
