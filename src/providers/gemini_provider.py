"""Google Gemini adapter (Google AI Studio REST API via `requests`).

Uses the REST endpoint directly rather than the google-generativeai SDK: the SDK
drags in native crypto deps that break in some environments, and REST needs only
`requests`. Auth is the AI Studio `?key=` query parameter.

A safety block is a *behavior*, not an API error: Gemini returns HTTP 200 with a
blockReason / a candidate that has no text. We turn that into a marker string
(no error) so the refusal_boundary checks flag it as drift rather than the runner
discarding it as a failed call. A non-200 (429 quota, 503 busy) is raised so the
base class treats it as a transient transport error and retries / records it as an
error — never as drift.
"""

from __future__ import annotations

import time

import requests

from .base import Provider, ProviderResult

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(Provider):
    def _generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        url = f"{_BASE}/models/{self.model_id}:generateContent"
        body = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": float(self.run_cfg.get("temperature", 0)),
                "maxOutputTokens": int(self.run_cfg.get("max_tokens", 512)),
            },
        }
        timeout = int(self.run_cfg.get("timeout_s", 30))
        start = time.perf_counter()
        resp = requests.post(url, params={"key": self.api_key}, json=body, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if resp.status_code != 200:
            # Transient / transport — let the base class retry, then record as error.
            raise RuntimeError(f"gemini HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no_candidates")
            text = f"[NO_OUTPUT:{reason}]"
        else:
            cand = candidates[0]
            parts = cand.get("content", {}).get("parts", [])
            if parts:
                text = "".join(p.get("text", "") for p in parts)
            else:
                text = f"[NO_OUTPUT:{cand.get('finishReason', 'empty')}]"

        usage = data.get("usageMetadata", {})
        return ProviderResult(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )
