"""Anthropic (Claude) adapter.

Determinism note: the current Claude models split on sampling params. Haiku 4.5
still accepts `temperature`; Opus 4.x / Sonnet 5 / Fable 5 reject it with a 400.
We send temperature=0 optimistically and, if the model rejects it, retry once
without sampling params (and remember that for the rest of the run). That keeps
determinism where it's available without crashing on models that dropped it.
"""

from __future__ import annotations

import time

from .base import Provider, ProviderResult

# Model ids whose family is known to reject `temperature`. Not required for
# correctness (the runtime fallback below handles any model), just avoids a
# wasted first request for models we already know about.
_KNOWN_NO_SAMPLING = ("opus-4", "sonnet-5", "fable-5", "mythos-5")


class AnthropicProvider(Provider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = None
        # None = unknown, True/False once learned for this model.
        self._send_temperature = not any(
            tag in self.model_id for tag in _KNOWN_NO_SAMPLING
        )

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _call(self, system_prompt: str, user_prompt: str, with_temperature: bool):
        client = self._get_client()
        kwargs = dict(
            model=self.model_id,
            max_tokens=int(self.run_cfg.get("max_tokens", 512)),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if with_temperature:
            kwargs["temperature"] = float(self.run_cfg.get("temperature", 0))
        start = time.perf_counter()
        resp = client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return resp, latency_ms

    def _generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        import anthropic

        try:
            resp, latency_ms = self._call(
                system_prompt, user_prompt, self._send_temperature
            )
        except anthropic.BadRequestError as exc:
            # If the only problem was the sampling param, retry without it and
            # remember for subsequent prompts this run.
            if self._send_temperature and "temperature" in str(exc).lower():
                self._send_temperature = False
                resp, latency_ms = self._call(system_prompt, user_prompt, False)
            else:
                raise

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return ProviderResult(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=getattr(resp.usage, "input_tokens", None),
            completion_tokens=getattr(resp.usage, "output_tokens", None),
        )
