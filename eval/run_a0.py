"""A0 baseline collector (v2 §4, arm A0 / A0').

Assumes evalplus generation + evaluation already ran (see scripts/run_a0.sh),
producing greedy and multi-sample eval_results.json files. This step is CPU-only:
it parses those, adds error breakdown + difficulty stratification, and writes a
single results/a0_<tag>.json summary + a printed table.

Usage:
  python -m eval.run_a0 --dataset humaneval --tag qwen1.5b \\
      --greedy-results  <greedy>_eval_results.json \\
      --greedy-samples  <greedy sanitized>.jsonl \\
      --passk-results   <n64>_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os

from .difficulty import assign_difficulty
from .run_eval import (
    error_breakdown,
    load_samples,
    parse_eval_results,
    pass_at_1,
    pass_at_k,
    samples_from_results,
    stratified_error_breakdown,
    stratified_pass_at_1,
)


def _load_tasks(dataset: str) -> dict:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def collect(dataset: str, tag: str, greedy_results: str, greedy_samples: str | None,
            passk_results: str | None, out_dir: str = "results",
            arm: str = "A0") -> dict:
    tasks = _load_tasks(dataset)
    difficulty = assign_difficulty(tasks)
    # Prefer an explicit samples file; otherwise read solutions from the results.
    greedy_sol = load_samples(greedy_samples) if greedy_samples else samples_from_results(greedy_results)

    # Headline pass@1 (base and base+plus) from evalplus results.
    correct_plus = parse_eval_results(greedy_results, use_plus=True)
    correct_base = parse_eval_results(greedy_results, use_plus=False)

    summary = {
        "arm": arm,
        "dataset": dataset,
        "tag": tag,
        "n_tasks": len(tasks),
        "pass@1_base": pass_at_1(correct_base),
        "pass@1_plus": pass_at_1(correct_plus),
        "pass@1_plus_by_difficulty": stratified_pass_at_1(correct_plus, difficulty),
        "error_breakdown": error_breakdown(tasks, greedy_sol),
        "error_breakdown_by_difficulty": stratified_error_breakdown(tasks, greedy_sol, difficulty),
        "difficulty_counts": {
            layer: sum(1 for v in difficulty.values() if v == layer)
            for layer in ("easy", "medium", "hard")
        },
    }

    if passk_results:
        correct_k = parse_eval_results(passk_results, use_plus=True)
        summary["pass@k_plus"] = pass_at_k(correct_k)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"a0_{tag}_{dataset}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    _print_table(summary)
    print(f"\nwrote {out_path}")
    return summary


def _print_table(s: dict) -> None:
    print(
        f"\n=== {s['arm']}: {s['tag']} on {s['dataset']} "
        f"({s['n_tasks']} tasks) ==="
    )
    print(f"pass@1 (base): {100*s['pass@1_base']:.1f}%   pass@1 (plus): {100*s['pass@1_plus']:.1f}%")
    print("pass@1 (plus) by difficulty:", {k: f"{100*v:.1f}%" for k, v in s["pass@1_plus_by_difficulty"].items()})
    print("error breakdown:", {k: f"{v:.1f}%" for k, v in s["error_breakdown"].items()})
    if "pass@k_plus" in s:
        print("pass@k (plus):", {k: f"{100*v:.1f}%" for k, v in s["pass@k_plus"].items()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--arm", default="A0", help="experiment arm stored in the summary")
    p.add_argument("--tag", required=True, help="e.g. qwen1.5b (A0) or qwen7b (A0')")
    p.add_argument("--greedy-results", required=True)
    p.add_argument("--greedy-samples", default=None,
                   help="optional; if omitted, solutions are read from --greedy-results")
    p.add_argument("--passk-results", default=None)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    collect(
        args.dataset,
        args.tag,
        args.greedy_results,
        args.greedy_samples,
        args.passk_results,
        args.out_dir,
        arm=args.arm,
    )


if __name__ == "__main__":
    main()
