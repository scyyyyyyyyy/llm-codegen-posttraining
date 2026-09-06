"""Repairability probe -- a diagnostic (observation-only, NOT a training reward).

Question: is partial-credit reward (visible test-pass fraction) a faithful proxy
for "closeness to a correct solution"? A brute-force / hard-coded solution can
pass many visible tests yet be far from correct; a correct algorithm with a small
bug fails tests yet is one edit away. We separate the two via REPAIRABILITY:

  Give a FIXED strong repairer (7B teacher) the failing solution plus ERROR-ONLY
  feedback (no expected outputs), let it iterate <=R rounds against the VISIBLE
  tests, then judge the final solution on HELD-OUT tests.

      Repairability(s) = 1 if the bounded repair reaches held-out-correct, else 0

Three design choices keep it clean:
  - repair on VISIBLE tests, JUDGE on HELD-OUT  -> brute-force/hacking patches the
    visible subset but fails to generalize -> Repairability = 0 (as it should)
  - a fixed 7B repairer                          -> controls for repairer skill
  - error-only feedback (no expected outputs)    -> no answer leakage / hard-coding

Framing: partial reward is the IMMEDIATE reward r(s); repairability estimates the
VALUE V(s) -- the correctness reachable from s under a bounded repair policy.

GPU (vLLM). Smoke:  python -m eval.repairability --n 5 --rollouts 4 --rounds 2
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random

from data.common import extract_code, normalize_code, read_jsonl

from .sandbox import run_one

SYSTEM = "You are an expert Python programmer. Write clean, correct code."
STUDENT = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TEACHER = "/root/autodl-tmp/Qwen2.5-Coder-7B-Instruct"
POOL = "data/snapshots/pre_a4/prompt_pool.clean.jsonl"


# ---------- test helpers ----------

def _build_tests_from_evalplus(task: dict, max_tests: int) -> list[str]:
    """Build a LIST of individual `assert ep(*inp) == out` strings from an evalplus
    task, using the canonical solution as the oracle. The training pool tops out at
    3 tests/task (no partial-reward resolution); the eval sets have dozens, giving a
    real partial-reward range. Pure diagnostic use (no training) on the eval set.
    """
    import copy

    ep = task["entry_point"]
    src = task["prompt"] + task["canonical_solution"]
    ns: dict = {}
    try:
        exec(src, ns)  # canonical is trusted
    except Exception:
        return []
    fn = ns.get(ep)
    if not callable(fn):
        return []
    inputs = list(task.get("base_input") or []) + list(task.get("plus_input") or [])
    tests, canon_def = [], f"{src}\n"
    for inp in inputs[:max_tests]:
        try:
            out = fn(*copy.deepcopy(inp))
            t = f"assert {ep}({', '.join(map(repr, inp))}) == {out!r}"
        except Exception:
            continue
        # keep only tests the canonical itself passes (drops float/repr edge cases)
        if run_one(canon_def, t, None).passed:
            tests.append(t)
    return tests


def load_evalplus(dataset: str, n: int, max_tests: int, seed: int) -> list[dict]:
    """A few eval tasks as pool-schema rows with a rich per-input test list."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    tasks = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    rows = []
    for tid, t in tasks.items():
        tests = _build_tests_from_evalplus(t, max_tests)
        if len(tests) >= 4:  # need enough to split with partial-reward resolution
            rows.append({"id": tid, "prompt_text": t["prompt"],
                         "entry_point": t["entry_point"], "tests": tests})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def split_tests(tests: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Split a task's asserts into (visible, held-out), >=1 each. Deterministic."""
    idx = list(range(len(tests)))
    random.Random(seed).shuffle(idx)
    half = max(1, len(tests) // 2)
    vis = [tests[i] for i in idx[:half]]
    held = [tests[i] for i in idx[half:]] or [tests[idx[-1]]]
    return vis, held


def pass_fraction(code: str, tests: list[str]) -> float:
    if not tests:
        return 0.0
    return sum(run_one(code, t, None).passed for t in tests) / len(tests)


def all_pass(code: str, tests: list[str]) -> bool:
    return bool(tests) and all(run_one(code, t, None).passed for t in tests)


def defines_entry(code: str, entry_point: str) -> bool:
    """True iff `code` actually defines the target function (subprocess-checked).

    Rejects extraction failures (e.g. the model emitted only print/test-driver
    lines): those are not solutions, and counting them as 'repaired' just measures
    the teacher re-solving the task from scratch.
    """
    return run_one(code, f"assert callable({entry_point})", None).passed


def error_feedback(code: str, visible: list[str]) -> str:
    """Error-ONLY message for the first failing visible test (no expected values).

    We deliberately do NOT surface the assert text (it would leak the expected
    output and invite hard-coding). We only say whether it returned a wrong answer
    or crashed, and with what exception class.
    """
    for i, t in enumerate(visible):
        res = run_one(code, t, None)
        if res.passed:
            continue
        if res.timed_out:
            return f"On hidden test case {i + 1}, your function timed out (likely too slow / an infinite loop)."
        if res.exception_type in (None, "AssertionError"):
            return f"On hidden test case {i + 1}, your function returned an incorrect result."
        return f"On hidden test case {i + 1}, your function raised {res.exception_type}."
    # All visible tests pass but the solution is still wrong on held-out tests:
    # nudge toward generality WITHOUT revealing any held-out case (no leakage).
    return ("Your function passes the visible tests but is still incorrect on some "
            "hidden edge cases. Make it correct in general, not just on these cases.")


# ---------- prompts ----------

def _chat(tok, msgs) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def solve_prompt(tok, prompt_text: str, entry_point: str) -> str:
    return _chat(tok, [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"Problem:\n{prompt_text}\n\nWrite the function `{entry_point}`."},
    ])


def repair_prompt(tok, prompt_text: str, entry_point: str, code: str, feedback: str) -> str:
    user = (
        f"Your solution to this problem is failing some hidden tests.\n\n"
        f"Problem:\n{prompt_text}\n\n"
        f"Your current solution:\n```python\n{code}\n```\n\n"
        f"{feedback}\n\n"
        f"Fix the function `{entry_point}`. Return only the corrected code."
    )
    return _chat(tok, [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": user}])


# ---------- main probe ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="humaneval",
                   choices=["humaneval", "mbpp", "pool"],
                   help="test source. eval sets give many tests/task (real partial "
                        "range); 'pool' tops out at 3 tests -> no resolution.")
    p.add_argument("--pool", default=POOL)
    p.add_argument("--max-tests", type=int, default=12,
                   help="cap on tests/task built from the eval oracle")
    p.add_argument("--student", default=STUDENT)
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--n", type=int, default=5, help="number of problems (smoke: 5)")
    p.add_argument("--rollouts", type=int, default=4, help="student samples per problem")
    p.add_argument("--rounds", type=int, default=2, help="max repair rounds R")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/repairability_smoke.json")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # --- pick problems that have enough tests to split with partial-reward range ---
    if args.source == "pool":
        pool = [r for r in read_jsonl(args.pool) if len(r.get("tests", [])) >= 2]
        random.Random(args.seed).shuffle(pool)
        pool = pool[: args.n]
    else:
        pool = load_evalplus(args.source, args.n, args.max_tests, args.seed)
    for r in pool:
        r["visible"], r["heldout"] = split_tests(r["tests"], args.seed)
    print(f"{len(pool)} problems (>=2 tests), R={args.rounds}, rollouts={args.rollouts}")

    # --- 1. student: sample rollouts, keep the FAILING ones (not held-out correct) ---
    tok_s = AutoTokenizer.from_pretrained(args.student)
    student = LLM(model=args.student, max_model_len=2048)
    s_out = student.generate(
        [solve_prompt(tok_s, r["prompt_text"], r["entry_point"]) for r in pool],
        SamplingParams(n=args.rollouts, temperature=args.temp, max_tokens=512))
    del student

    solutions = []
    skipped_nonfn = 0
    for r, o in zip(pool, s_out):
        for cand in o.outputs:
            code = extract_code(cand.text)
            if all_pass(code, r["heldout"]):
                continue  # already correct on held-out -> not an interesting failure
            if not defines_entry(code, r["entry_point"]):
                skipped_nonfn += 1  # extraction failure, not a solution -> exclude
                continue
            solutions.append({
                "task": r["id"], "prompt_text": r["prompt_text"],
                "entry_point": r["entry_point"],
                "visible": r["visible"], "heldout": r["heldout"],
                "orig_code": code,
                # partial reward = what the L1 signal sees (visible pass fraction):
                "partial_reward": pass_fraction(code, r["visible"]),
                "heldout_frac_before": pass_fraction(code, r["heldout"]),
                "code": code, "rounds": 0, "converged": False, "done": False,
            })
    print(f"collected {len(solutions)} failing rollouts "
          f"({skipped_nonfn} skipped: no function defined)")
    if not solutions:
        print("no failing rollouts to probe (try more rollouts / higher temp)")
        return

    # --- 2. teacher: repair loop on VISIBLE tests, error-only feedback ---
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    tok_t = AutoTokenizer.from_pretrained(args.teacher)
    teacher = LLM(model=args.teacher, max_model_len=2048)

    for rnd in range(1, args.rounds + 1):
        # Keep repairing until the solution is held-out-correct (early stop only;
        # held-out is never shown to the repairer -- feedback comes from visible).
        active = [s for s in solutions if not s["done"] and not all_pass(s["code"], s["heldout"])]
        if not active:
            break
        prompts = [repair_prompt(tok_t, s["prompt_text"], s["entry_point"],
                                 s["code"], error_feedback(s["code"], s["visible"]))
                   for s in active]
        outs = teacher.generate(prompts, SamplingParams(
            n=1, temperature=0.0, max_tokens=512))
        for s, o in zip(active, outs):
            new = extract_code(o.outputs[0].text)
            if normalize_code(new) == normalize_code(s["code"]):
                s["converged"] = True
                s["done"] = True          # fixed point: no further change
            else:
                s["code"] = new
                s["rounds"] = rnd
        print(f"  round {rnd}: repaired {len(active)} solutions")
    del teacher

    # --- 3. score: Repairability = held-out correct AFTER bounded repair ---
    rows = []
    for s in solutions:
        final = s["code"]
        repairable = int(all_pass(final, s["heldout"]))
        sim = difflib.SequenceMatcher(None, s["orig_code"], final).ratio()
        rows.append({
            "task": s["task"], "entry_point": s["entry_point"],
            "partial_reward": round(s["partial_reward"], 3),
            "heldout_frac_before": round(s["heldout_frac_before"], 3),
            "repairability": repairable,
            "rounds": s["rounds"], "converged": s["converged"],
            "sim_orig_final": round(sim, 3),
            "orig_code": s["orig_code"], "final_code": final,
        })

    _report(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"config": vars(args), "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


def _report(rows: list[dict]) -> None:
    import numpy as np

    pr = np.array([r["partial_reward"] for r in rows])
    rep = np.array([r["repairability"] for r in rows], dtype=float)
    # LOCAL repairability = reached held-out-correct AND via a local edit (high sim).
    # reach alone is dominated by "can the teacher re-solve the task"; gating on sim
    # keeps only genuine near-correct fixes, which is what "closeness" means.
    val = np.array([r["repairability"] * r["sim_orig_final"] for r in rows])
    TAU = 0.6
    local = np.array([float(r["repairability"] and r["sim_orig_final"] >= TAU) for r in rows])
    print(f"\n=== Repairability probe: {len(rows)} failing solutions ===")
    print(f"mean partial_reward (visible) = {pr.mean():.3f}")
    print(f"mean Repairability  (reach, any edit)     = {rep.mean():.3f}")
    print(f"mean LocalRepair    (reach AND sim>= {TAU}) = {local.mean():.3f}  "
          f"<- the closeness-valid version")

    def _corr(y, name):
        if pr.std() > 0 and y.std() > 0:
            print(f"corr(partial_reward, {name}) = {float(np.corrcoef(pr, y)[0,1]):+.3f}")

    # Headline: does partial reward predict fixability? (both raw and locality-gated)
    _corr(rep, "Repairability(reach)")
    _corr(val, "reach*sim (graded)")
    _corr(local, "LocalRepair")
    print("  (near 0 => test-pass fraction does NOT capture closeness-to-correct)")

    # Rates within partial-reward bins -> look for non-monotonic / spread
    print("\npartial_reward bin      n    reach    LocalRepair")
    for lo, hi in [(0.0, 0.34), (0.34, 0.67), (0.67, 1.0001)]:
        sub = [r for r in rows if lo <= r["partial_reward"] < hi]
        if sub:
            reach = sum(r["repairability"] for r in sub) / len(sub)
            lr = sum(r["repairability"] and r["sim_orig_final"] >= TAU for r in sub) / len(sub)
            print(f"  [{lo:.2f}, {hi:.2f})        {len(sub):>3}     {reach:.2f}     {lr:.2f}")

    # CONFOUND CHECK 1 -- repair vs resample. If "repairable" low-partial solutions
    # were rewritten from scratch (low sim), repairability measures teacher skill,
    # not the original's closeness. High sim => genuine local fix.
    rep_rows = [r for r in rows if r["repairability"] == 1]
    if rep_rows:
        sims = np.array([r["sim_orig_final"] for r in rep_rows])
        print(f"\nrepair-vs-resample: among repairable solutions, mean sim(orig,final) "
              f"= {sims.mean():.2f}  (high => local repair, low => teacher rewrote)")

    # Case studies: the two diagnostic corners, WITH code so we can eyeball.
    hi_pr_low_rep = [r for r in rows if r["partial_reward"] >= 0.5 and r["repairability"] == 0]
    lo_pr_hi_rep = [r for r in rows if r["partial_reward"] < 0.34 and r["repairability"] == 1]
    print(f"\nhigh partial / NOT repairable (brute-force-like): {len(hi_pr_low_rep)}")
    print(f"low  partial / repairable (near-correct w/ bug):  {len(lo_pr_hi_rep)}")

    def _dump(label, rs, k=2):
        for r in rs[:k]:
            print(f"\n  --- {label} | {r['task']} {r['entry_point']} "
                  f"partial={r['partial_reward']} repair={r['repairability']} "
                  f"rounds={r['rounds']} sim={r['sim_orig_final']} ---")
            print("  ORIG :", r["orig_code"].replace("\n", "\n         ")[:300])
            if r["repairability"] or r["final_code"] != r["orig_code"]:
                print("  FINAL:", r["final_code"].replace("\n", "\n         ")[:300])

    _dump("HIGH-partial NOT-repairable (brute?)", hi_pr_low_rep)
    _dump("LOW-partial repairable (near-correct bug?)", lo_pr_hi_rep)


if __name__ == "__main__":
    main()
