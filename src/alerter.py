"""Fire a Discord webhook when a run breaches and clears the alert gate.

The alert names WHAT moved (which categories, verbosity/latency z-scores,
before/after drift score) — that "what changed" is the product value, per the
blueprint, not a bare number.
"""

from __future__ import annotations

import os

import requests

from .scorer import RunSummary


def _webhook_url() -> str | None:
    return os.environ.get("DISCORD_WEBHOOK_URL") or None


def build_message(summary: RunSummary) -> str:
    lines = [
        f"**⚠️ Model drift detected — {summary.model}** ({summary.provider})",
        f"`{summary.run_at}`",
        "",
        f"Drift score: **{summary.drift_score}**"
        + (
            f"  (baseline {summary.baseline_avg}, "
            f"Δ {round(summary.drift_score - summary.baseline_avg, 1)})"
            if summary.baseline_avg is not None
            else ""
        ),
        f"Breach streak: {summary.breach_streak}",
    ]

    collapsed = [c for c, v in summary.scores.items() if v < 50]
    if collapsed:
        lines.append(
            "Category collapse: "
            + ", ".join(f"{c} ({summary.scores[c]})" for c in collapsed)
        )
    if abs(summary.verbosity_z) > 2:
        lines.append(f"Verbosity shift: z={summary.verbosity_z}")
    if summary.latency_z > 2:
        lines.append(f"Latency shift: z={summary.latency_z}")

    if summary.failures:
        lines.append("")
        lines.append("Example failures:")
        for f in summary.failures[:5]:
            lines.append(
                f"• `{f['id']}` expected={f['expected']!r} got={f['got']!r}"
            )
    return "\n".join(lines)


def send_alert(summary: RunSummary, timeout: int = 15) -> bool:
    """Return True if an alert was actually sent."""
    if not summary.alerted:
        return False
    url = _webhook_url()
    if not url:
        print(f"[alerter] DISCORD_WEBHOOK_URL not set; would have alerted for "
              f"{summary.model}")
        return False
    content = build_message(summary)
    # Discord hard-caps message content at 2000 chars.
    resp = requests.post(url, json={"content": content[:1990]}, timeout=timeout)
    resp.raise_for_status()
    print(f"[alerter] sent Discord alert for {summary.model}")
    return True


# --- per-run heartbeat -----------------------------------------------------
# Unlike send_alert (fires only on a gated breach), the heartbeat posts once per
# run summarizing every watched model. It gives you an always-on "the watchdog
# ran and here's where each model stands" ping — and, crucially, proves the
# webhook works even when nothing has drifted.
def _status_label(summary: RunSummary) -> tuple[str, str]:
    if summary.alerted:
        return "🚨", "ALERT"
    if summary.error_count and not summary.scores:
        return "⛔", "all calls errored"
    if summary.calibrating:
        return "🟡", "calibrating"
    if summary.breach:
        return "⚠️", "breach"
    return "✅", "stable"


def build_heartbeat(summaries: list[RunSummary]) -> str:
    run_at = summaries[0].run_at if summaries else ""
    lines = [f"**🐕 Model Drift Watchdog — run** `{run_at}`", ""]
    for s in summaries:
        emoji, label = _status_label(s)
        base = f" (baseline {s.baseline_avg})" if s.baseline_avg is not None else ""
        note = f" · {s.error_count} err" if s.error_count else ""
        lines.append(
            f"{emoji} **{s.model}** — drift {s.drift_score}{base} · {label}{note}"
        )
    breached = [s for s in summaries if s.breach and not s.calibrating]
    if breached:
        lines.append("")
        lines.append("⚠️ Breaching this run: " + ", ".join(s.model for s in breached))
    return "\n".join(lines)


def send_heartbeat(summaries: list[RunSummary], timeout: int = 15) -> bool:
    """Post a one-message per-run summary of all models. Returns True if sent."""
    if not summaries:
        return False
    url = _webhook_url()
    if not url:
        print("[alerter] DISCORD_WEBHOOK_URL not set; skipping heartbeat")
        return False
    content = build_heartbeat(summaries)
    # Discord hard-caps message content at 2000 chars.
    resp = requests.post(url, json={"content": content[:1990]}, timeout=timeout)
    resp.raise_for_status()
    print(f"[alerter] sent Discord heartbeat ({len(summaries)} models)")
    return True
