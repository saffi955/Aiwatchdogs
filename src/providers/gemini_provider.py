"""Google Gemini adapter (uses the free-tier-friendly google-generativeai SDK).

A safety block is a *behavior*, not an API error: Gemini returns a successful
response whose text accessor raises because the candidate was filtered. We turn
that into a marker string with no error, so the refusal_boundary checks can
correctly flag it as drift rather than the runner discarding it as a failed call.
"""

from __future__ import annotations

import time

from .base import Provider, ProviderResult


class GeminiProvider(Provider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

    def _get_model(self, system_prompt: str):
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                self.model_id, system_instruction=system_prompt
            )
        return self._model

    def _generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        model = self._get_model(system_prompt)
        generation_config = {
            "temperature": float(self.run_cfg.get("temperature", 0)),
            "max_output_tokens": int(self.run_cfg.get("max_tokens", 512)),
        }
        start = time.perf_counter()
        resp = model.generate_content(user_prompt, generation_config=generation_config)
        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            text = resp.text
        except (ValueError, Exception):  # noqa: BLE001 - blocked/empty candidate
            # Not a transport error — capture why the model produced no text so
            # checks (esp. refusal_boundary) can evaluate the behavior.
            reason = "unknown"
            try:
                if resp.candidates:
                    reason = str(resp.candidates[0].finish_reason)
                elif resp.prompt_feedback:
                    reason = str(resp.prompt_feedback.block_reason)
            except Exception:  # noqa: BLE001
                pass
            text = f"[NO_OUTPUT:{reason}]"

        usage = getattr(resp, "usage_metadata", None)
        return ProviderResult(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
        )
