"""Reward functions and metrics.

Two reward designs are compared in the ablation:
  - binary_reward : 1.0 iff ALL tests pass, else 0.0  (baseline)
  - partial_reward: fraction of tests passed + small compile/runtime bonuses
                    (the core research contribution — gives hard problems a
                    dense gradient instead of an all-zero signal)
"""

from __future__ import annotations

from collections import Counter

from .error_classify import classify_error
from .safe_execute import runs_without_exception, safe_execute


def binary_reward(code: str, tests: list[str]) -> float:
    """1.0 only if every test passes."""
    return 1.0 if all(safe_execute(code, t) for t in tests) else 0.0


def partial_reward(code: str, tests: list[str]) -> float:
    """Fraction of tests passed, plus compile and runtime bonuses (capped at 1.0)."""
    results = [safe_execute(code, t) for t in tests]
    base_score = sum(results) / len(tests) if tests else 0.0

    try:
        compile(code, "<generated>", "exec")
        compile_bonus = 0.1
    except SyntaxError:
        compile_bonus = 0.0

    runtime_bonus = 0.1 if runs_without_exception(code) else 0.0
    return min(1.0, base_score + compile_bonus + runtime_bonus)


def pass_at_1(records: list[bool]) -> float:
    """Fraction of problems solved by the single sampled solution."""
    return sum(records) / len(records) if records else 0.0


def pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if num_samples - num_correct < k:
        return 1.0
    prod = 1.0
    for i in range(num_samples - num_correct + 1, num_samples + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def error_breakdown(codes: list[str], tests_per_problem: list[list[str]]) -> dict:
    """Percentage of each error class across a set of (code, tests) pairs."""
    labels = [
        classify_error(code, tests[0])
        for code, tests in zip(codes, tests_per_problem)
        if tests
    ]
    counts = Counter(labels)
    total = len(labels) or 1
    return {label: 100.0 * n / total for label, n in counts.items()}
