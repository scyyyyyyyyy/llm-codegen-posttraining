"""Repair-augmented rejection-sampling SFT data (the "涨点" arm).

Tests whether the repairability insight -- near-correct-but-failing solutions are
one edit from correct -- can be turned into extra, execution-verified training
data that plain resampling misses.

Two SFT sets from the SAME student samples (isolates repair's contribution):
  - baseline (STaR)      : keep the student's samples that pass ALL train tests
  - treatment (STaR+rep) : baseline PLUS student-self-repaired near-misses that,
                           after a bounded self-repair loop on the train tests,
                           pass ALL train tests

No Goodhart: only solutions that pass every train test enter either set (same
execution-verified bar as the existing SFT). The pay-off is measured later as
held-out pass@1 of SFT(treatment) vs SFT(baseline).

GPU (vLLM). Usage:
  python -m data.build_repair_sft --pool data/snapshots/pre_a4/prompt_pool.clean.jsonl \
      --student Qwen/Qwen2.5-Coder-1.5B-Instruct --k 8 --rounds 3 --out-prefix data/sft_repair
"""

from __future__ import annotations

import argparse

from eval.repairability import error_feedback
from eval.sandbox import run_batch, run_one

from .common import extract_code, read_jsonl, to_chat_record, write_jsonl

SYSTEM = "You are an expert Python programmer. Write clean, correct code."


def _solve_prompt(tok, item):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Problem:\n{item['prompt_text']}\n\n"
                                     f"Write the function `{item['entry_point']}`."}],
        tokenize=False, add_generation_prompt=True)


def _repair_prompt(tok, item, code, feedback):
    user = (f"Your solution is failing some tests.\n\nProblem:\n{item['prompt_text']}\n\n"
            f"Your current solution:\n```python\n{code}\n```\n\n{feedback}\n\n"
            f"Make the smallest possible change to fix it. Return only the corrected "
            f"code for `{item['entry_point']}`.")
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def _passes_all(code, tests, workers=8):
    if not tests:
        return False
    return all(r.passed for r in run_batch([(code, t, None) for t in tests], workers=workers))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/snapshots/pre_a4/prompt_pool.clean.jsonl")
    p.add_argument("--student", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--k", type=int, default=8, help="student samples per prompt")
    p.add_argument("--rounds", type=int, default=3, help="max self-repair rounds")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--out-prefix", default="data/sft_repair")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    pool = [r for r in read_jsonl(args.pool) if r.get("tests")]
    tok = AutoTokenizer.from_pretrained(args.student)
    llm = LLM(model=args.student, max_model_len=2048, seed=args.seed)

    # --- 1. sample k per prompt; verify against ALL train tests ---
    outs = llm.generate([_solve_prompt(tok, it) for it in pool],
                        SamplingParams(n=args.k, temperature=args.temp, max_tokens=512))
    base_rows, failing = [], []          # failing: (item, best_failing_code)
    n_solved_by_sampling = 0
    for it, o in zip(pool, outs):
        cands = [extract_code(c.text) for c in o.outputs]
        solved = next((c for c in cands if _passes_all(c, it["tests"])), None)
        if solved is not None:
            base_rows.append(to_chat_record(it["prompt_text"], it["entry_point"], solved))
            n_solved_by_sampling += 1
        elif cands:
            # keep the highest partial-pass candidate as the repair seed
            best = max(cands, key=lambda c: sum(run_one(c, t, None).passed for t in it["tests"]))
            failing.append({"item": it, "code": best})
    print(f"sampling: {n_solved_by_sampling}/{len(pool)} solved; {len(failing)} to repair")

    # --- 2. self-repair the failures (student fixes its own near-misses) ---
    repaired_rows = []
    active = failing
    for rnd in range(1, args.rounds + 1):
        active = [f for f in active if not _passes_all(f["code"], f["item"]["tests"])]
        if not active:
            break
        prompts = [_repair_prompt(tok, f["item"], f["code"],
                                  error_feedback(f["code"], f["item"]["tests"]))
                   for f in active]
        rep = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=512))
        for f, o in zip(active, rep):
            f["code"] = extract_code(o.outputs[0].text)
        print(f"  self-repair round {rnd}: attempted {len(active)}")
    n_recovered = 0
    for f in failing:
        if _passes_all(f["code"], f["item"]["tests"]):
            repaired_rows.append(to_chat_record(
                f["item"]["prompt_text"], f["item"]["entry_point"], f["code"]))
            n_recovered += 1
    print(f"self-repair recovered {n_recovered} prompts that sampling could not solve")

    # --- 3. emit both SFT sets ---
    write_jsonl(f"{args.out_prefix}_baseline.jsonl", base_rows)
    write_jsonl(f"{args.out_prefix}_treatment.jsonl", base_rows + repaired_rows)
    print(f"baseline SFT rows : {len(base_rows)}")
    print(f"treatment SFT rows: {len(base_rows) + len(repaired_rows)} "
          f"(+{len(repaired_rows)} from repair)")
    print(f"wrote {args.out_prefix}_baseline.jsonl and {args.out_prefix}_treatment.jsonl")


if __name__ == "__main__":
    main()
