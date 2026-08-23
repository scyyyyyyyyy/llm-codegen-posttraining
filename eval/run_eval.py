"""Metrics & breakdown over evalplus outputs (v2 §4, RQ2, RQ4).

Headline pass@1 / pass@k come from evalplus's own evaluator output
(`*_eval_results.json`); this module parses that file and adds the project's
custom analyses: error-type breakdown and difficulty stratification.

evalplus eval_results.json layout (evalplus 0.3.x):
  {date, hash, eval: {task_id: [ {task_id, solution, base_status, plus_status,
                                  base_fail_tests, plus_fail_tests}, ... ]}}
  status in {"pass", "fail", "timeout", None}.
"""

from __future__ import annotations

import json

from .difficulty import LAYERS, assign_difficulty
from .error_classify import LABELS, breakdown, classify

PASS_AT_K_VALUES = (1, 4, 16, 64)


# ---------- evalplus results parsing ----------

def parse_eval_results(path: str, use_plus: bool = True) -> dict[str, list[bool]]:
    """Return {task_id: [correct_bool per sampled attempt]}.

    With use_plus, an attempt is correct iff it passes BOTH base and plus tests.
    """
    with open(path) as f:
        d = json.load(f)
    out: dict[str, list[bool]] = {}
    for tid, attempts in d["eval"].items():
        flags = []
        for a in attempts:
            base_ok = a.get("base_status") == "pass"
            plus_ok = a.get("plus_status") == "pass"
            flags.append(base_ok and plus_ok if use_plus else base_ok)
        out[tid] = flags
    return out


def pass_at_1(correct_by_task: dict[str, list[bool]]) -> float:
    """Greedy / first-sample pass@1 over tasks."""
    if not correct_by_task:
        return 0.0
    return sum(1 for f in correct_by_task.values() if f and f[0]) / len(correct_by_task)


def pass_at_k(correct_by_task: dict[str, list[bool]], ks=PASS_AT_K_VALUES) -> dict[int, float]:
    """Unbiased pass@k across tasks (reuses evalplus.estimate_pass_at_k)."""
    from evalplus.eval import estimate_pass_at_k

    totals = [len(f) for f in correct_by_task.values()]
    corrects = [sum(f) for f in correct_by_task.values()]
    result = {}
    for k in ks:
        if min(totals) >= k:
            result[k] = float(estimate_pass_at_k(totals, corrects, k).mean())
    return result


# ---------- custom analyses ----------

def load_samples(path: str) -> dict[str, list[str]]:
    """Load a (sanitized) samples.jsonl -> {task_id: [solution, ...]}."""
    out: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            code = r.get("solution") or r.get("completion") or ""
            out.setdefault(r["task_id"], []).append(code)
    return out


def samples_from_results(path: str) -> dict[str, list[str]]:
    """Extract solutions directly from an evalplus eval_results.json.

    Each attempt already carries its `solution`, so the breakdown needs no
    separate samples file.
    """
    with open(path) as f:
        d = json.load(f)
    return {
        tid: [a.get("solution", "") for a in attempts]
        for tid, attempts in d["eval"].items()
    }


def task_test(t: dict) -> tuple[str, str | None]:
    """Return (test_source, entry_point_to_drive) for a task.

    HumanEval+ ships a `check(candidate)` function in `test` -> drive it with a
    trailing `check(entry_point)` call. MBPP+ ships plain `assert` statements in
    `assertion` that call the function directly -> run as-is, no driver.
    """
    if t.get("test"):
        return t["test"], t["entry_point"]
    return t.get("assertion", ""), None


def error_breakdown(tasks: dict, greedy_samples: dict[str, list[str]]) -> dict[str, float]:
    """Classify the greedy (first) sample of each task against its base test."""
    labels = []
    for tid, t in tasks.items():
        sols = greedy_samples.get(tid)
        if not sols:
            continue
        test_src, ep = task_test(t)
        labels.append(classify(sols[0], test_src, entry_point=ep))
    return breakdown(labels)


def stratified_pass_at_1(correct_by_task, difficulty: dict[str, str]) -> dict[str, float]:
    """pass@1 within each easy/medium/hard layer."""
    out = {}
    for layer in LAYERS:
        sub = {tid: f for tid, f in correct_by_task.items() if difficulty.get(tid) == layer}
        out[layer] = pass_at_1(sub)
    return out


def stratified_error_breakdown(tasks, greedy_samples, difficulty) -> dict[str, dict[str, float]]:
    """Error breakdown within each difficulty layer."""
    out = {}
    for layer in LAYERS:
        sub = {tid: t for tid, t in tasks.items() if difficulty.get(tid) == layer}
        out[layer] = error_breakdown(sub, greedy_samples)
    return out
