"""Configuration loading and small shared paths.

Reads config.yml once and exposes typed accessors. Keeping all path logic here
means the runner, scorer, alerter and tests all agree on where data lives.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yml"
TESTS_PATH = ROOT / "prompts" / "tests.json"
RESULTS_DIR = ROOT / "data" / "results"
HISTORY_PATH = ROOT / "data" / "history.json"
DOCS_HISTORY_PATH = ROOT / "docs" / "history.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    weights = cfg["scoring"]["weights"]
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"scoring.weights must sum to 1.0, got {total}")
    for m in cfg["models"]:
        for key in ("name", "provider", "model_id", "env_key"):
            if key not in m:
                raise ValueError(f"model entry missing {key!r}: {m}")


def watched_models(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in cfg["models"] if m.get("watch", True)]


def get_api_key(env_key: str) -> str | None:
    key = os.environ.get(env_key)
    return key or None
