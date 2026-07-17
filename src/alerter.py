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
