"""Statistics layer -- the project's differentiating signature (v2 §6).

Stance (Miller 2024, "Adding Error Bars to Evals"): LLM eval is a statistical
estimation problem and must be treated as one.

  - multi-seed: 3 training seeds/arm; main table reports mean +/- std
  - primary test: problem-level PAIRED bootstrap on pass@1 differences
    (10,000 resamples over problems) -> 95% CI
  - McNemar test on per-problem binary outcomes (robustness)
  - stratified CIs per easy/medium/hard layer (report wide CIs honestly)
  - power / MDE: minimum detectable effect at 80% power for a 164-problem paired
    design (lets you say "directionally consistent but not significant")
  - multiple comparisons: Holm-Bonferroni on primary pairwise tests;
    label exploratory analyses as exploratory

One notebook in analysis/ reproduces every table and figure from raw results.
"""

from __future__ import annotations

BOOTSTRAP_RESAMPLES = 10_000
TARGET_POWER = 0.80


def paired_bootstrap_ci(pass_a: list[bool], pass_b: list[bool],
                        resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """Problem-level paired bootstrap of mean(pass_a) - mean(pass_b).

    Returns (point_estimate, ci_low, ci_high) at 95%. TODO: resample problem
    indices, recompute the paired difference each draw.
    """
    raise NotImplementedError


def mcnemar_test(pass_a: list[bool], pass_b: list[bool]) -> float:
    """McNemar p-value on discordant pairs (b, c). TODO."""
    raise NotImplementedError


def stratified_ci(pass_a, pass_b, difficulty: list[str]) -> dict[str, tuple[float, float, float]]:
    """Paired bootstrap CI computed separately per difficulty layer."""
    raise NotImplementedError


def minimum_detectable_effect(n_problems: int = 164, power: float = TARGET_POWER,
                              alpha: float = 0.05) -> float:
    """MDE for a paired binary design at the given power. TODO."""
    raise NotImplementedError


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Return {comparison: reject?} under Holm-Bonferroni. TODO."""
    raise NotImplementedError
