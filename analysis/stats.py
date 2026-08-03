"""Statistics layer -- the project's differentiating signature (v2 §6).

Stance (Miller 2024, "Adding Error Bars to Evals"): LLM eval is a statistical
estimation problem, so report uncertainty, not just point estimates.

All functions take per-problem boolean arrays (pass/fail), aligned by task id
across the two systems being compared.
"""

from __future__ import annotations

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
TARGET_POWER = 0.80


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def pass_at_1(passes) -> float:
    p = _arr(passes)
    return float(p.mean()) if p.size else 0.0


def paired_bootstrap_ci(
    pass_a, pass_b, resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05, seed: int = 0,
) -> dict:
    """Problem-level paired bootstrap of mean(pass_a) - mean(pass_b).

    Resamples TASKS (not the two systems independently), which is what makes the
    paired design tight: per-task correlation cancels out. Returns the point
    estimate, a (1-alpha) CI, and the bootstrap p-value (share of resamples on
    the other side of 0, doubled).
    """
    a, b = _arr(pass_a), _arr(pass_b)
    assert a.shape == b.shape and a.size, "aligned, non-empty arrays required"
    diff = a - b
    n = diff.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = diff[idx].mean(axis=1)
    point = float(diff.mean())
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value for H0: mean diff = 0
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"delta": point, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p)), "n": int(n)}


def mcnemar_test(pass_a, pass_b) -> dict:
    """Exact McNemar test on discordant pairs (paired binary outcomes).

    b = A wrong & B right; c = A right & B wrong. Under H0 the discordants split
    50/50, so use an exact two-sided binomial on min(b,c) with n=b+c.
    """
    a = _arr(pass_a).astype(bool)
    b_ = _arr(pass_b).astype(bool)
    b = int(np.sum(~a & b_))     # A wrong, B right
    c = int(np.sum(a & ~b_))     # A right, B wrong
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    from math import comb
    k = min(b, c)
    # exact two-sided binomial at p=0.5 (symmetric): 2 * lower tail, capped at 1
    lower = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_value": float(min(1.0, 2 * lower))}


def stratified_ci(pass_a, pass_b, difficulty, **kw) -> dict:
    """Paired bootstrap CI computed separately per difficulty layer.

    difficulty: list/array of layer labels aligned with the pass arrays.
    """
    a, b = _arr(pass_a), _arr(pass_b)
    d = np.asarray(difficulty)
    out = {}
    for layer in ("easy", "medium", "hard"):
        m = d == layer
        if m.sum() == 0:
            continue
        out[layer] = paired_bootstrap_ci(a[m], b[m], **kw)
    return out


def minimum_detectable_effect(
    n: int, base_rate: float = 0.5, power: float = TARGET_POWER,
    alpha: float = 0.05,
) -> float:
    """Approx MDE (difference in pass@1) at the given power for n problems.

    Normal approximation for a difference in proportions; uses base_rate for the
    variance (0.5 = worst case / most conservative). This is the INDEPENDENT-
    sample approximation, which is conservative: the paired design has higher
    power, so the true MDE is somewhat smaller. Use to say "a 5% gap on 164
    problems is below the ~X% we can resolve -> directionally consistent but not
    significant."
    """
    from statistics import NormalDist
    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)
    p = base_rate
    return float(z * np.sqrt(2 * p * (1 - p) / n))


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns {comparison: reject_H0?}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    reject = {}
    still = True
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        if still and p <= thresh:
            reject[name] = True
        else:
            still = False
            reject[name] = False
    return reject
