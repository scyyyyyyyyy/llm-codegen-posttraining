"""Paired comparison of two arms from their evalplus eval_results.json files.

Aligns per-task greedy outcomes, then reports pass@1 for each arm, the paired
bootstrap CI + p-value of the difference, McNemar, per-difficulty CIs, and the
MDE for context. This is how the A0/A1/A0' claims get error bars.

Usage:
  python -m analysis.compare \
      --a <A_eval_results.json> --b <B_eval_results.json> \
      --labels A1 A0 --dataset humaneval
"""

from __future__ import annotations

import argparse
import json

from eval.difficulty import assign_difficulty
from eval.run_eval import parse_eval_results

from . import stats


def _greedy_correct(path: str, use_plus: bool = True) -> dict[str, bool]:
    """task_id -> correct? for the greedy (first) attempt."""
    per_task = parse_eval_results(path, use_plus=use_plus)
    return {tid: (flags[0] if flags else False) for tid, flags in per_task.items()}


def _load_tasks(dataset: str) -> dict:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def compare(a_path: str, b_path: str, dataset: str,
            labels=("A", "B"), use_plus: bool = True) -> dict:
    ca, cb = _greedy_correct(a_path, use_plus), _greedy_correct(b_path, use_plus)
    ids = sorted(set(ca) & set(cb))
    a = [ca[i] for i in ids]
    b = [cb[i] for i in ids]

    diff = stats.paired_bootstrap_ci(a, b)
    mc = stats.mcnemar_test(a, b)
    diffi = assign_difficulty(_load_tasks(dataset))
    layers = [diffi.get(i, "?") for i in ids]
    strat = stats.stratified_ci(a, b, layers)
    mde = stats.minimum_detectable_effect(len(ids))

    return {
        "labels": list(labels),
        "dataset": dataset,
        "n": len(ids),
        f"pass@1_{labels[0]}": stats.pass_at_1(a),
        f"pass@1_{labels[1]}": stats.pass_at_1(b),
        "delta_ci": diff,          # A - B with 95% CI + bootstrap p
        "mcnemar": mc,
        "stratified": strat,
        "mde_80pct_power": mde,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="eval_results.json for arm A")
    p.add_argument("--b", required=True, help="eval_results.json for arm B")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--base-only", action="store_true", help="use base tests, not plus")
    args = p.parse_args()
    res = compare(args.a, args.b, args.dataset, tuple(args.labels), not args.base_only)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
