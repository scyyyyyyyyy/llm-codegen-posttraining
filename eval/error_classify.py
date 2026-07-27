"""Error taxonomy (v2 RQ2).

Classes: correct | syntax_error | runtime_error | logic_error | timeout

Pipeline: compile check -> execute -> inspect exception -> assertion fail.
Recorded at EVERY checkpoint so error-type dynamics can be plotted vs step.
Human-validate on >=50 problems and report the accuracy.
"""

from __future__ import annotations

from .sandbox import DEFAULT_TIMEOUT, precheck_compiles, run_one

RUNTIME_EXCEPTIONS = {
    "TypeError", "IndexError", "AttributeError", "RecursionError",
    "KeyError", "ValueError", "ZeroDivisionError", "NameError",
    "OverflowError", "StopIteration", "IndentationError", "ImportError",
    "ModuleNotFoundError", "UnboundLocalError", "MemoryError",
}

LABELS = ("correct", "syntax_error", "runtime_error", "logic_error", "timeout")


def classify(code: str, test: str, entry_point: str | None = None,
             timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return the error class for a single (code, test) pair."""
    if not precheck_compiles(code):
        return "syntax_error"

    res = run_one(code, test, entry_point, timeout=timeout)
    if res.passed:
        return "correct"
    if res.timed_out:
        return "timeout"

    exc = res.exception_type
    if exc in ("AssertionError", None):
        # No exception surfaced but exit != 0, or an explicit assert fail -> logic.
        return "logic_error"
    if exc in RUNTIME_EXCEPTIONS:
        return "runtime_error"
    # Unknown exception class: default to runtime (it did raise something).
    return "runtime_error"


def breakdown(labels: list[str]) -> dict[str, float]:
    """Percentage of each class over a list of labels (missing classes -> 0.0)."""
    total = len(labels) or 1
    return {lab: 100.0 * labels.count(lab) / total for lab in LABELS}


def validate_on_sample(examples: list[tuple[str, str, str, str]]) -> float:
    """examples = [(code, test, entry_point, gold_label), ...] -> accuracy.

    Use this on >=50 hand-labeled cases and report the number (RQ2 credibility).
    """
    if not examples:
        return 0.0
    correct = sum(
        1 for code, test, ep, gold in examples if classify(code, test, ep) == gold
    )
    return correct / len(examples)
