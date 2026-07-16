"""Machine checkers — one function per check_type.

Every checker is pure Python (no AI). Each takes the model's raw text output
and the test's ``expected`` value, and returns a ``CheckResult``.

Design rules:
- Checkers only decide pass/fail for a single prompt. They never score or alert.
- A checker must never raise on malformed model output; a crash-worthy output
  is simply a fail with a reason. (Genuine API/transport errors are handled by
  the runner, not here, and are recorded as errors, not failures.)
- Comparisons strip surrounding whitespace so a stray trailing newline is not
  treated as drift. Internal content is compared as-is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover - dependency is declared in requirements
    jsonschema = None


@dataclass
class CheckResult:
    passed: bool
    reason: str = ""


# Patterns that indicate the model refused / hedged instead of answering.
# Used by the not_refusal checker and to sanity-flag refusals elsewhere.
_REFUSAL_PATTERN = re.compile(
    r"(i\s+can'?t\b"
    r"|i\s+cannot\b"
    r"|i'?m\s+unable\b"
    r"|i\s+am\s+unable\b"
    r"|i'?m\s+not\s+able\b"
    r"|i\s+won'?t\b"
    r"|i\s+will\s+not\b"
    r"|i'?m\s+sorry,?\s+but\b"
    r"|as\s+an\s+ai\b"
    r"|i\s+do\s+not\s+feel\s+comfortable\b"
    r"|i'?m\s+not\s+comfortable\b"
    r"|unable\s+to\s+(?:help|assist|provide|comply)\b"
    r"|can'?t\s+(?:help|assist|provide|comply)\b"
    r"|cannot\s+(?:help|assist|provide|comply|fulfill)\b"
    r"|against\s+my\s+guidelines\b"
    r"|i\s+can\s+not\s+(?:help|assist|provide)\b)",
    re.IGNORECASE | re.DOTALL,
)

# A leading number, int or float, optionally signed, allowing thousands commas.
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _strip_code_fences(text: str) -> str:
    """Remove a ```...``` markdown fence if the model wrapped its output in one.

    Kept lenient: fences are exactly the drift we want to *detect* for format
    tests, so the JSON checker deliberately does NOT call this — it treats a
    fence as a failure. Extraction/other JSON checks may call it when the fence
    is not the property under test.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def check_exact_match(output: str, expected: Any) -> CheckResult:
    got = output.strip()
    want = str(expected).strip()
    if got == want:
        return CheckResult(True)
    return CheckResult(False, f"expected {want!r}, got {got!r}")


def check_number(output: str, expected: Any) -> CheckResult:
    m = _NUMBER_PATTERN.search(output)
    if not m:
        return CheckResult(False, f"no number found in {output.strip()!r}")
    got = float(m.group(0).replace(",", ""))
    want = float(expected)
    if abs(got - want) <= 1e-6:
        return CheckResult(True)
    return CheckResult(False, f"expected {want}, got {got}")


def check_length_range(output: str, expected: Any) -> CheckResult:
    lo, hi = expected
    n = len(output.strip())
    if lo <= n <= hi:
        return CheckResult(True)
    return CheckResult(False, f"length {n} outside [{lo}, {hi}]")


def check_label_set(output: str, expected: Any) -> CheckResult:
    """expected = {"labels": [...allowed...], "answer": "correct label"}.

    Passes only if the (case-insensitively normalized) reply is a member of the
    allowed set AND equals the correct answer. Being *in the set but wrong* and
    being *outside the set* are distinct failure reasons — both matter for drift.
    """
    labels = [str(x) for x in expected["labels"]]
    answer = str(expected["answer"])
    got = output.strip()
    norm = {label.lower(): label for label in labels}
    got_key = got.lower()
    if got_key not in norm:
        return CheckResult(False, f"{got!r} not in label set {labels}")
    if norm[got_key].lower() != answer.lower():
        return CheckResult(False, f"classified {got!r}, expected {answer!r}")
    return CheckResult(True)


def check_regex(output: str, expected: Any) -> CheckResult:
    pattern = str(expected)
    if re.search(pattern, output.strip()):
        return CheckResult(True)
    return CheckResult(False, f"{output.strip()!r} did not match /{pattern}/")


def check_not_refusal(output: str, expected: Any = None) -> CheckResult:
    """Passes when the reply is a real answer, not a refusal/hedge.

    ``expected`` is ignored (there is no fixed correct answer for a
    refusal-boundary probe — we only care that the model did not refuse).
    """
    text = output.strip()
    if len(text) < 3:
        return CheckResult(False, f"answer too short to be a real response: {text!r}")
    m = _REFUSAL_PATTERN.search(text)
    if m:
        return CheckResult(False, f"looks like a refusal (matched {m.group(0)!r})")
    return CheckResult(True)


def check_json_schema(output: str, expected: Any) -> CheckResult:
    """Parse the reply as JSON and validate against a JSON Schema.

    A markdown code fence is treated as a FAILURE for format tests: the whole
    point is to catch a model that starts fencing its JSON. We therefore parse
    the raw text directly and only report a schema/parse failure.
    """
    if jsonschema is None:
        raise RuntimeError("jsonschema is not installed; add it to requirements")
    raw = output.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return CheckResult(False, f"not raw JSON ({exc.msg}); got {raw[:80]!r}")
    try:
        jsonschema.validate(instance=data, schema=expected)
    except jsonschema.ValidationError as exc:
        return CheckResult(False, f"schema mismatch: {exc.message}")
    return CheckResult(True)


CHECKERS = {
    "exact_match": check_exact_match,
    "number": check_number,
    "length_range": check_length_range,
    "label_set": check_label_set,
    "regex": check_regex,
    "not_refusal": check_not_refusal,
    "json_schema": check_json_schema,
}


def run_check(check_type: str, output: str, expected: Any) -> CheckResult:
    """Dispatch to the right checker. Unknown check_type is a hard config error."""
    if check_type not in CHECKERS:
        raise KeyError(f"unknown check_type: {check_type!r}")
    return CHECKERS[check_type](output, expected)
