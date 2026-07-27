"""Difficulty stratification (v2 RQ1 needs easy/medium/hard layers).

IMPORTANT: EvalPlus ships NO official difficulty labels. We derive a proxy and
are explicit about it (honesty is a plus, per v2 §1.3). Default proxy = terciles
of canonical-solution logical lines of code (LOC), computed over the dataset so
it is deterministic and reproducible.

Alternative proxies worth reporting later: baseline model pass-rate buckets, or
number of plus-tests. Keep the proxy fixed across all arms so stratified
comparisons stay valid.
"""

from __future__ import annotations

LAYERS = ("easy", "medium", "hard")


def solution_loc(canonical_solution: str) -> int:
    """Count non-blank, non-comment logical lines of the canonical solution."""
    n = 0
    for line in canonical_solution.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def assign_difficulty(tasks: dict) -> dict[str, str]:
    """Map task_id -> {easy,medium,hard} by terciles of canonical-solution LOC.

    `tasks` is the dict returned by get_human_eval_plus() / get_mbpp_plus().
    """
    locs = {tid: solution_loc(t["canonical_solution"]) for tid, t in tasks.items()}
    ordered = sorted(locs.values())
    if not ordered:
        return {}
    lo = ordered[len(ordered) // 3]
    hi = ordered[2 * len(ordered) // 3]

    def bucket(v: int) -> str:
        if v <= lo:
            return "easy"
        if v <= hi:
            return "medium"
        return "hard"

    return {tid: bucket(v) for tid, v in locs.items()}
