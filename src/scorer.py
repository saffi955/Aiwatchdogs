"""Turn per-prompt check results into a per-model drift score + alert decision.

Faithful to the blueprint's v1 formula:

    drift_score = 0.25*format + 0.20*instruction + 0.20*classification
                + 0.15*math    + 0.10*extraction  + 0.10*refusal
    penalty: -5 if |verbosity_z| > 2,  -5 if latency_z > 2
    alert if drift_score < baseline_avg - 15  OR  any category < 50

Guardrails baked in:
- API errors are excluded from pass rates (a failed call is NOT drift).
- A category whose tests ALL errored is dropped and the remaining weights are
  renormalized, so one provider hiccup doesn't sink the score.
- First `calibration_days` days of history never alert (no baseline yet).
- An alert requires `consecutive_breaches` breaching runs in a row.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TestResult:
    id: str
    category: str
    passed: bool          # meaningful only when errored is False
    errored: bool
    latency_ms: float
    length: int           # char length of the (stripped) response
    error: str | None = None
    got: str = ""
    expected: Any = None


@dataclass
class RunSummary:
    run_at: str
    model: str
    provider: str
    scores: dict[str, float] = field(default_factory=dict)  # category -> pass rate
    verbosity_avg_len: float = 0.0
    latency_ms_avg: float = 0.0
    verbosity_z: float = 0.0
    latency_z: float = 0.0
    drift_score: float = 0.0
    baseline_avg: float | None = None
    breach: bool = False
    breach_streak: int = 0
    alerted: bool = False
    calibrating: bool = False
    error_count: int = 0
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def category_pass_rates(results: list[TestResult]) -> dict[str, float]:
    """Pass rate (0-100) per category over non-errored tests only."""
    buckets: dict[str, list[bool]] = {}
    for r in results:
        if r.errored:
            continue
        buckets.setdefault(r.category, []).append(r.passed)
    return {
        cat: 100.0 * sum(vals) / len(vals)
        for cat, vals in buckets.items()
        if vals
    }


def _weighted_score(pass_rates: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean over the weighted categories that actually ran, with the
    weights renormalized across whichever of them are present."""
    present = {c: w for c, w in weights.items() if c in pass_rates}
    total_w = sum(present.values())
    if total_w == 0:
        return 0.0
    return sum(pass_rates[c] * w for c, w in present.items()) / total_w


def _rolling_values(history: list[dict], key: str, baseline_days: int) -> list[float]:
    cutoff = _now() .timestamp() - baseline_days * 86400
    out = []
    for run in history:
        ts = _parse(run.get("run_at"))
        if ts is None or ts.timestamp() >= cutoff:
            v = run.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
    return out


def _zscore(value: float, sample: list[float]) -> float:
    if len(sample) < 2:
        return 0.0
    mean = statistics.mean(sample)
    stdev = statistics.pstdev(sample)
    if stdev == 0:
        return 0.0
    return (value - mean) / stdev


def _distinct_days(history: list[dict]) -> int:
    days = set()
    for run in history:
        ts = _parse(run.get("run_at"))
        if ts is not None:
            days.add(ts.date())
    return len(days)


def score_run(
    results: list[TestResult],
    history: list[dict],
    cfg: dict,
    model: str,
    provider: str,
    run_at: str | None = None,
) -> RunSummary:
    scoring = cfg["scoring"]
    weights = scoring["weights"]
    run_at = run_at or _now().isoformat()

    pass_rates = category_pass_rates(results)
    non_errored = [r for r in results if not r.errored]

    verbosity_lengths = [
        r.length for r in non_errored if r.category == "verbosity_fingerprint"
    ]
    verbosity_avg = statistics.mean(verbosity_lengths) if verbosity_lengths else 0.0
    latency_avg = (
        statistics.mean([r.latency_ms for r in non_errored]) if non_errored else 0.0
    )

    verbosity_z = _zscore(
        verbosity_avg,
        _rolling_values(history, "verbosity_avg_len", scoring["baseline_days"]),
    )
    latency_z = _zscore(
        latency_avg,
        _rolling_values(history, "latency_ms_avg", scoring["baseline_days"]),
    )

    score = _weighted_score(pass_rates, weights)
    penalty = scoring.get("penalty_points", 5)
    if abs(verbosity_z) > scoring["verbosity_z_threshold"]:
        score -= penalty
    if latency_z > scoring["latency_z_threshold"]:
        score -= penalty
    score = max(0.0, min(100.0, score))

    # Baseline = mean drift_score over the rolling window.
    baseline_scores = _rolling_values(history, "drift_score", scoring["baseline_days"])
    baseline_avg = statistics.mean(baseline_scores) if baseline_scores else None

    collapse_threshold = scoring["category_collapse_threshold"]
    drop_threshold = scoring["drift_drop_threshold"]
    collapsed = [c for c, v in pass_rates.items() if v < collapse_threshold]
    score_dropped = (
        baseline_avg is not None and score < baseline_avg - drop_threshold
    )
    breach = bool(collapsed) or bool(score_dropped)

    prev_streak = history[-1].get("breach_streak", 0) if history else 0
    breach_streak = prev_streak + 1 if breach else 0

    calibrating = _distinct_days(history) < scoring["calibration_days"]
    alerted = (
        not calibrating
        and breach
        and breach_streak >= scoring["consecutive_breaches"]
    )

    failures = [
        {"id": r.id, "category": r.category, "expected": r.expected, "got": r.got[:200]}
        for r in results
        if not r.errored and not r.passed
    ]

    return RunSummary(
        run_at=run_at,
        model=model,
        provider=provider,
        scores={c: round(v, 1) for c, v in pass_rates.items()},
        verbosity_avg_len=round(verbosity_avg, 1),
        latency_ms_avg=round(latency_avg, 1),
        verbosity_z=round(verbosity_z, 2),
        latency_z=round(latency_z, 2),
        drift_score=round(score, 1),
        baseline_avg=round(baseline_avg, 1) if baseline_avg is not None else None,
        breach=breach,
        breach_streak=breach_streak,
        alerted=alerted,
        calibrating=calibrating,
        error_count=sum(1 for r in results if r.errored),
        failures=failures,
    )


# --- small datetime helpers ------------------------------------------------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None
