"""Offline verification of the drift scorer and alert gating.

    python -m tests.test_scorer
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src import config
from src.scorer import TestResult, score_run

CFG = config.load_config()

WEIGHTED = [
    "format_compliance", "instruction_following", "classification_stability",
    "math_reasoning", "extraction_accuracy", "refusal_boundary",
]
ALL_CATS = WEIGHTED + ["verbosity_fingerprint", "edge_case_handling"]


def make_results(failing: set[str] | None = None, length: int = 20) -> list[TestResult]:
    """Two tests per category; categories in `failing` fail both."""
    failing = failing or set()
    out = []
    for cat in ALL_CATS:
        for i in range(2):
            out.append(
                TestResult(
                    id=f"{cat}_{i}", category=cat,
                    passed=cat not in failing, errored=False,
                    latency_ms=800.0, length=length,
                )
            )
    return out


def hist_run(day_offset: int, drift: float, streak: int = 0) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(days=day_offset)).isoformat()
    return {
        "run_at": ts, "drift_score": drift, "verbosity_avg_len": 20.0,
        "latency_ms_avg": 800.0, "breach_streak": streak,
    }


def test_clean_run_calibrating() -> None:
    s = score_run(make_results(), [], CFG, "m", "p")
    assert s.drift_score == 100.0, s.drift_score
    assert s.calibrating is True
    assert s.alerted is False
    assert s.breach is False
    print("clean run / calibration: OK")


def test_category_collapse_two_breaches_alerts() -> None:
    # 3 distinct days of healthy history → past calibration.
    history = [hist_run(4, 100.0), hist_run(3, 100.0), hist_run(2, 100.0)]

    # First breaching run: format_compliance collapses to 0.
    s1 = score_run(make_results(failing={"format_compliance"}), history, CFG, "m", "p")
    assert s1.breach is True
    assert s1.breach_streak == 1
    assert s1.calibrating is False
    assert s1.alerted is False, "should not alert on a single breach"

    # Second consecutive breach → alert.
    history2 = history + [
        {**hist_run(1, s1.drift_score), "breach_streak": s1.breach_streak}
    ]
    s2 = score_run(make_results(failing={"format_compliance"}), history2, CFG, "m", "p")
    assert s2.breach_streak == 2
    assert s2.alerted is True, "should alert on the 2nd consecutive breach"
    print("category collapse / 2-breach alert gate: OK")


def test_score_drop_alert() -> None:
    history = [hist_run(4, 100.0), hist_run(3, 100.0), hist_run(2, 100.0),
               {**hist_run(1, 100.0), "breach_streak": 1}]
    # instruction_following collapses (weight 0.20) -> ~80, a >15 drop from 100.
    s = score_run(make_results(failing={"instruction_following"}), history, CFG, "m", "p")
    assert s.drift_score < 100 - CFG["scoring"]["drift_drop_threshold"]
    assert s.breach is True
    assert s.alerted is True
    print("score-drop alert: OK")


def test_errors_are_not_drift() -> None:
    results = make_results()
    # Mark all format tests as errored (API failure) — should be excluded, not fail.
    for r in results:
        if r.category == "format_compliance":
            r.errored = True
    s = score_run(results, [], CFG, "m", "p")
    assert "format_compliance" not in s.scores
    assert s.error_count == 2
    assert s.drift_score == 100.0  # remaining categories all pass
    print("errors excluded from drift: OK")


def test_verbosity_penalty() -> None:
    # Realistic verbosity history (small run-to-run variance ~20 chars), then a
    # spike to 200 chars -> large z-score -> penalty. A zero-variance baseline
    # would (safely) yield z=0; real data always varies.
    history = [
        {**hist_run(4, 100.0), "verbosity_avg_len": 18.0},
        {**hist_run(3, 100.0), "verbosity_avg_len": 20.0},
        {**hist_run(2, 100.0), "verbosity_avg_len": 22.0},
    ]
    s = score_run(make_results(length=200), history, CFG, "m", "p")
    assert abs(s.verbosity_z) > CFG["scoring"]["verbosity_z_threshold"], s.verbosity_z
    assert s.drift_score < 100.0  # penalty applied
    print("verbosity z-penalty: OK")


if __name__ == "__main__":
    test_clean_run_calibrating()
    test_category_collapse_two_breaches_alerts()
    test_score_drop_alert()
    test_errors_are_not_drift()
    test_verbosity_penalty()
    print("ALL SCORER CHECKS PASSED")
