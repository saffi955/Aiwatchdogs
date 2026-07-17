"""Entry point: run every test prompt against every watched model, score the
run, persist results, and alert on breach.

    python -m src.runner

What happens each run:
  1. Load config.yml + prompts/tests.json.
  2. For each watched model: send all prompts (fixed system prompt, temp 0,
     fixed max_tokens), run the machine checker for each, record pass/fail +
     latency + token counts. A failed API call is recorded as an *error*, never
     a failed check.
  3. Score the run against the model's rolling history.
  4. Write data/results/<ts>__<model>.json, append to data/history.json, and
     refresh docs/history.json for the dashboard.
  5. If the run breaches and clears the alert gate, fire the Discord webhook.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .alerter import send_alert
from .checkers import run_check
from .providers import build_provider
from .scorer import TestResult, score_run


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()


def load_tests() -> list[dict]:
    with open(config.TESTS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)["tests"]


def load_history() -> dict:
    if config.HISTORY_PATH.exists():
        with open(config.HISTORY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"models": {}}


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def run_model(model_cfg: dict, tests: list[dict], run_cfg: dict, run_at: str):
    """Send every prompt to one model; return (TestResult list, raw records)."""
    provider = build_provider(model_cfg, run_cfg)
    system_prompt = run_cfg.get("system_prompt", "You are a helpful assistant.")

    results: list[TestResult] = []
    raw_records: list[dict] = []
    # Space requests to stay under free-tier per-minute limits (reduces 429s).
    delay = float(run_cfg.get("request_delay_s", 0))

    for i, test in enumerate(tests):
        if delay and i:
            time.sleep(delay)
        pr = provider.generate(system_prompt, test["prompt"])
        if pr.error:
            results.append(
                TestResult(
                    id=test["id"], category=test["category"], passed=False,
                    errored=True, latency_ms=pr.latency_ms, length=0,
                    error=pr.error, got="", expected=test.get("expected"),
                )
            )
        else:
            check = run_check(test["check_type"], pr.text, test.get("expected"))
            results.append(
                TestResult(
                    id=test["id"], category=test["category"], passed=check.passed,
                    errored=False, latency_ms=pr.latency_ms,
                    length=len(pr.text.strip()), error=None, got=pr.text,
                    expected=test.get("expected"),
                )
            )
        raw_records.append(
            {
                "id": test["id"],
                "category": test["category"],
                "check_type": test["check_type"],
                "passed": results[-1].passed if not results[-1].errored else None,
                "errored": results[-1].errored,
                "error": pr.error,
                "latency_ms": round(pr.latency_ms, 1),
                "prompt_tokens": pr.prompt_tokens,
                "completion_tokens": pr.completion_tokens,
                "output": pr.text,
            }
        )
    return results, raw_records


def _compact_history_entry(summary_dict: dict) -> dict:
    """Trim a run summary to what the scorer + dashboard need (drop failures)."""
    keep = {k: v for k, v in summary_dict.items() if k != "failures"}
    return keep


def main() -> int:
    cfg = config.load_config()
    tests = load_tests()
    run_cfg = cfg["run"]
    history = load_history()
    run_at = _now_iso()

    exit_code = 0
    for model_cfg in config.watched_models(cfg):
        name = model_cfg["name"]
        model_hist = history["models"].get(name, [])
        try:
            results, raw_records = run_model(model_cfg, tests, run_cfg, run_at)
        except Exception as exc:  # noqa: BLE001 - never let one model kill the run
            print(f"[runner] {name}: unexpected failure: {exc}")
            exit_code = 1
            continue

        summary = score_run(
            results, model_hist, cfg,
            model=name, provider=model_cfg["provider"], run_at=run_at,
        )
        summary_dict = summary.to_dict()

        # Per-run detailed artifact.
        results_path = config.RESULTS_DIR / f"{run_at.replace(':', '')}__{_slug(name)}.json"
        save_json(
            results_path,
            {"summary": summary_dict, "model_id": model_cfg["model_id"], "results": raw_records},
        )

        # Append to rolling history.
        history["models"].setdefault(name, []).append(_compact_history_entry(summary_dict))

        status = (
            "ALERT" if summary.alerted
            else "calibrating" if summary.calibrating
            else "breach" if summary.breach
            else "ok"
        )
        print(
            f"[runner] {name}: drift={summary.drift_score} "
            f"errors={summary.error_count} status={status}"
        )

        if send_alert(summary):
            pass  # already logged inside

    save_json(config.HISTORY_PATH, history)
    _write_dashboard_history(history)
    return exit_code


def _write_dashboard_history(history: dict) -> None:
    """Compact per-model time series for the Chart.js dashboard."""
    out = {"generated_at": _now_iso(), "models": {}}
    for name, runs in history["models"].items():
        out["models"][name] = [
            {
                "run_at": r.get("run_at"),
                "drift_score": r.get("drift_score"),
                "scores": r.get("scores", {}),
                "verbosity_z": r.get("verbosity_z"),
                "latency_ms_avg": r.get("latency_ms_avg"),
                "alerted": r.get("alerted", False),
                "calibrating": r.get("calibrating", False),
            }
            for r in runs
        ]
    save_json(config.DOCS_HISTORY_PATH, out)


if __name__ == "__main__":
    raise SystemExit(main())
