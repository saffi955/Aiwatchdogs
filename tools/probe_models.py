"""One-off model probe: list every model each provider key can see, send one tiny
test prompt to each chat model, and write data/probe.json. Keys are read from the
environment and never written out. Run from GitHub Actions (open network)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "probe.json"
SKIP = ("whisper", "tts", "guard", "playai", "orpheus", "embed", "compound",
        "allam", "imagen", "veo", "aqa", "gemma-3n", "image", "audio", "live",
        "native-audio", "robotics", "computer-use")
MSG = [{"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Reply with exactly: OK"}]


def _err(r):
    return f"HTTP {r.status_code}: {r.text[:300]}"


def chat(base, key, model, extra_headers=None):
    h = {"Authorization": f"Bearer {key}"}
    h.update(extra_headers or {})
    t = time.perf_counter()
    try:
        r = requests.post(f"{base}/chat/completions", headers=h, timeout=60,
                          json={"model": model, "messages": MSG,
                                "temperature": 0, "max_tokens": 64})
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    ms = round((time.perf_counter() - t) * 1000)
    if r.status_code != 200:
        return {"ok": False, "error": _err(r), "ms": ms}
    try:
        d = r.json()
        txt = (d["choices"][0]["message"].get("content") or "")[:80]
        return {"ok": True, "reply": txt, "ms": ms}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"bad body {type(e).__name__}: {r.text[:200]}"}


def probe_groq(key):
    base = "https://api.groq.com/openai/v1"
    r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code != 200:
        return {"error": _err(r)}
    out = []
    for m in sorted(r.json()["data"], key=lambda m: m["id"]):
        row = {k: m.get(k) for k in ("id", "owned_by", "active", "context_window",
                                     "max_completion_tokens")}
        if any(s in m["id"].lower() for s in SKIP):
            row["skipped"] = True
        else:
            row.update(chat(base, key, m["id"]))
            time.sleep(2)
        out.append(row)
    return {"models": out}


def probe_openrouter(key):
    base = "https://openrouter.ai/api/v1"
    h = {"Authorization": f"Bearer {key}"}
    res = {}
    k = requests.get(f"{base}/key", headers=h, timeout=30)
    res["key_info"] = k.json().get("data") if k.status_code == 200 else _err(k)
    r = requests.get(f"{base}/models", timeout=30)
    if r.status_code != 200:
        res["error"] = _err(r)
        return res
    free = [m for m in r.json()["data"] if m["id"].endswith(":free")
            and "text" in (m.get("architecture", {}).get("output_modalities") or ["text"])]
    out = []
    for m in sorted(free, key=lambda m: m["id"]):
        row = {"id": m["id"], "context_length": m.get("context_length"),
               "max_completion_tokens": (m.get("top_provider") or {}).get("max_completion_tokens")}
        row.update(chat(base, key, m["id"]))
        out.append(row)
        time.sleep(3)
    res["free_models"] = out
    return res


def probe_gemini(key):
    base = "https://generativelanguage.googleapis.com/v1beta"
    r = requests.get(f"{base}/models", params={"key": key, "pageSize": 1000}, timeout=30)
    if r.status_code != 200:
        return {"error": _err(r)}
    out = []
    for m in r.json().get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        mid = m["name"].split("/", 1)[1]
        row = {"id": mid, "input_limit": m.get("inputTokenLimit"),
               "output_limit": m.get("outputTokenLimit")}
        if any(s in mid.lower() for s in SKIP):
            row["skipped"] = True
        else:
            t = requests.post(f"{base}/models/{mid}:generateContent", params={"key": key},
                              timeout=60, json={"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}],
                                                "generationConfig": {"temperature": 0, "maxOutputTokens": 64}})
            row.update({"ok": True} if t.status_code == 200 else {"ok": False, "error": _err(t)})
            time.sleep(2)
        out.append(row)
    return {"models": out}


def main():
    result = {"probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, env, fn in [("groq", "GROQ_API_KEY", probe_groq),
                          ("openrouter", "OPENROUTER_API_KEY", probe_openrouter),
                          ("gemini", "GEMINI_API_KEY", probe_gemini)]:
        key = os.environ.get(env)
        if not key:
            result[name] = {"error": f"{env} not set"}
            continue
        try:
            result[name] = fn(key)
        except Exception as e:  # noqa: BLE001
            result[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(f"[probe] {name} done")
    OUT.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
