"""Generic OpenAI-compatible adapter.

Many free-tier APIs speak the OpenAI Chat Completions protocol — Groq, DeepSeek,
OpenRouter, Together, etc. One adapter covers all of them: set `provider:
openai_compat`, a `base_url`, and the `env_key` holding the API key in config.yml.
No new code needed to add a provider from this family.
"""

from __future__ import annotations

import time

from .base import Provider, ProviderResult


class OpenAICompatProvider(Provider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = None
        self.base_url = self.model_cfg.get("base_url")  # None => api.openai.com

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        client = self._get_client()
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(self.run_cfg.get("temperature", 0)),
            max_tokens=int(self.run_cfg.get("max_tokens", 512)),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return ProviderResult(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
