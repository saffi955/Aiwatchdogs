"""End-to-end smoke test of the runner with a fake provider (no API calls).

    python -m tests.test_pipeline

Verifies the full flow — load config + tests, call the (fake) provider for every
prompt, score, and write data/results, data/history.json and docs/history.json —
without any network access, by monkeypatching the provider and redirecting all
output paths to a temp directory.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src import config, runner
from src.providers import ProviderResult
from tests.test_checkers import golden_output


def _load_prompt_to_golden() -> dict[str, str]:
    with open(config.TESTS_PATH, "r", encoding="utf-8") as fh:
        tests = json.load(fh)["tests"]
    return {t["prompt"]: golden_output(t) for t in tests}, tests


class FakeProvider:
    """Returns a correct answer for known prompts; optionally corrupts one
    category to simulate drift."""

    def __init__(self, prompt_map, tests, corrupt_category=None):
        self.prompt_map = prompt_map
        self.corrupt = corrupt_category
        self.prompt_to_cat = {t["prompt"]: t["category"] for t in tests}

    def generate(self, system_prompt, user_prompt):
        if self.corrupt and self.prompt_to_cat.get(user_prompt) == self.corrupt:
            return ProviderResult(text="!!! wrong answer !!!", latency_ms=810.0,
                                  prompt_tokens=10, completion_tokens=5)
        text = self.prompt_map.get(user_prompt, "")
        return ProviderResult(text=text, latency_ms=790.0,
                              prompt_tokens=10, completion_tokens=5)


def _redirect_paths(tmp: Path) -> None:
    config.RESULTS_DIR = tmp / "results"
    config.HISTORY_PATH = tmp / "history.json"
    config.DOCS_HISTORY_PATH = tmp / "docs_history.json"
    runner.config.RESULTS_DIR = config.RESULTS_DIR
    runner.config.HISTORY_PATH = config.HISTORY_PATH
    runner.config.DOCS_HISTORY_PATH = config.DOCS_HISTORY_PATH


def test_happy_path(tmp: Path) -> None:
    prompt_map, tests = _load_prompt_to_golden()
    runner.build_provider = lambda mcfg, rcfg: FakeProvider(prompt_map, tests)

    rc = runner.main()
    assert rc == 0, rc
    assert config.HISTORY_PATH.exists()
    assert config.DOCS_HISTORY_PATH.exists()
    result_files = list(config.RESULTS_DIR.glob("*.json"))
    assert result_files, "no per-run result files written"

    history = json.loads(config.HISTORY_PATH.read_text())
    for name, runs in history["models"].items():
        last = runs[-1]
        assert last["drift_score"] == 100.0, (name, last["drift_score"])
        assert last["calibrating"] is True
        assert last["error_count"] == 0
        assert last["alerted"] is False
    print(f"happy path: OK ({len(history['models'])} models, all drift=100)")


def test_drift_path(tmp: Path) -> None:
    prompt_map, tests = _load_prompt_to_golden()
    runner.build_provider = lambda mcfg, rcfg: FakeProvider(
        prompt_map, tests, corrupt_category="format_compliance"
    )
    runner.main()
    history = json.loads(config.HISTORY_PATH.read_text())
    # There is now a 2nd run per model; the corrupted one collapses format.
    for name, runs in history["models"].items():
        last = runs[-1]
        assert last["scores"]["format_compliance"] == 0.0, last["scores"]
        assert last["breach"] is True, name
    print("drift path: OK (format_compliance collapsed, breach flagged)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        _redirect_paths(Path(d))
        test_happy_path(Path(d))
        test_drift_path(Path(d))
    print("ALL PIPELINE CHECKS PASSED")
