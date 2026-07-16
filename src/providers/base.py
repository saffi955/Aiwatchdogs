"""Provider interface shared by every adapter.

An adapter's single job: given a system prompt and a user prompt, return the
model's text plus timing/token metadata, or an error string if the call failed.
A failed call is NOT drift — the runner records ``error`` separately and excludes
that prompt from scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import get_api_key


@dataclass
class ProviderResult:
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class Provider:
    """Base class. Subclasses implement ``_generate``; ``generate`` adds retries,
    timing, and turns exceptions into ``error`` results."""

    def __init__(self, model_cfg: dict[str, Any], run_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg
        self.model_id = model_cfg["model_id"]
        self.env_key = model_cfg["env_key"]
        self.api_key = get_api_key(self.env_key)

    # --- to be implemented by subclasses ---------------------------------
    def _generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        raise NotImplementedError

    # --- shared orchestration --------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        if not self.api_key:
            return ProviderResult(
                text="", latency_ms=0.0,
                error=f"missing API key (env {self.env_key} not set)",
            )
        retries = int(self.run_cfg.get("retries", 3))
        backoff = float(self.run_cfg.get("retry_backoff_s", 2))
        last_err: str | None = None
        for attempt in range(retries):
            start = time.perf_counter()
            try:
                result = self._generate(system_prompt, user_prompt)
                if result.latency_ms == 0.0:
                    result.latency_ms = (time.perf_counter() - start) * 1000.0
                return result
            except Exception as exc:  # noqa: BLE001 - any SDK error is a transport error
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < retries - 1:
                    time.sleep(backoff * (2 ** attempt))
        return ProviderResult(text="", latency_ms=0.0, error=last_err)
