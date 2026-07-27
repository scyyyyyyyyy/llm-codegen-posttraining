"""Reward functions across the density spectrum (v2 §2, §8.1).

L0  binary        : 1.0 iff all tests pass          (sparse; double-ended failure)
L1  partial       : fraction of tests passed + bonuses (dense; Goodhart-prone)
    irt_weighted  : partial credit weighted by per-test informativeness (方案一)

RQ3 (Goodhart): reward may be computed on a VISIBLE test subset only; evaluation
uses held-out tests. `hacking_gap = pass(visible) - pass(held_out)`.
"""

from __future__ import annotations

from .sandbox import (
    precheck_compiles,
    run_one,
    runs_without_exception,
)

COMPILE_BONUS = 0.1
NO_RUNTIME_ERROR_BONUS = 0.1


def binary_reward(code: str, tests: list[str], entry_point: str | None = None) -> float:
    """L0: 1.0 iff every test passes, else 0.0."""
    if not tests:
        return 0.0
    return 1.0 if all(run_one(code, t, entry_point).passed for t in tests) else 0.0


def partial_reward(code: str, tests: list[str], entry_point: str | None = None) -> float:
    """L1: (#passed / #tests) + compile & no-runtime-error bonuses, capped at 1.0."""
    if not tests:
        return 0.0
    passed = sum(run_one(code, t, entry_point).passed for t in tests)
    base = passed / len(tests)
    compile_bonus = COMPILE_BONUS if precheck_compiles(code) else 0.0
    runtime_bonus = (
        NO_RUNTIME_ERROR_BONUS if runs_without_exception(code, entry_point) else 0.0
    )
    return min(1.0, base + compile_bonus + runtime_bonus)


class IRTTestWeights:
    """Online Item-Response-Theory weighting of unit tests (方案一, main innovation).

    Each unit test is treated as a 'test item'. From the pass/fail pattern across
    a rollout group, estimate per-test difficulty/discrimination online. Simplified
    start: weight = -log(group_pass_rate), with shrinkage for early-training noise.

    Rationale: trivial tests (everyone passes) carry ~0 information and get ~0
    weight -> hard-coding easy tests no longer inflates reward (principled Goodhart
    mitigation vs. random subsampling).
    """

    def __init__(self, shrinkage: float = 1.0):
        self.shrinkage = shrinkage
        # TODO: maintain sliding pass-rate per test id

    def update(self, test_id: str, passed: bool) -> None:
        raise NotImplementedError  # TODO: update sliding pass-rate

    def weight(self, test_id: str) -> float:
        raise NotImplementedError  # TODO: -log(rate) with shrinkage

    def irt_weighted_reward(self, code: str, tests: list[tuple[str, str]]) -> float:
        """tests = [(test_id, test_src), ...] -> weighted pass score."""
        raise NotImplementedError


def hacking_gap(pass_visible: float, pass_held_out: float) -> float:
    """RQ3 metric: how much reward comes from gaming the visible test subset."""
    return pass_visible - pass_held_out
