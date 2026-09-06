"""Controlled repairability experiment (Exp 4) -- the decisive test of the idea.

The natural-rollout probe (eval.repairability) under-samples the crux: solutions
that pass MANY visible tests yet are far from correct (brute-force / hard-coded).
Here we CONSTRUCT the two populations with known ground truth and ask whether
repairability separates them even when partial reward says the opposite:

  A. NEAR-CORRECT  : the canonical solution with ONE small AST mutation injected
                     (flip a comparison, off-by-one constant, +/- swap). By
                     construction it is one edit from correct -> should be highly
                     LOCALLY repairable. Its visible pass rate is whatever the bug
                     causes (often mid/low).
  B. HARD-CODED    : `def f(*a): if a == <visible input>: return <output>; ... ;
                     return None`. Passes EVERY visible test (partial = 1.0) but is
                     not an algorithm -> should NOT be repairable from visible
                     feedback and fails held-out.

If  partial(B) > partial(A)  while  LocalRepair(A) >> LocalRepair(B),  that is a
clean inversion: test-pass fraction ranks B above A, repairability ranks A above B
-> partial reward does not measure closeness-to-correct, repairability does.

Same repair protocol as eval.repairability: fixed 7B repairer, error-only feedback
on VISIBLE tests, <=R rounds, judged on HELD-OUT tests. Diagnostic only.

GPU (vLLM).  Smoke:  python -m eval.repairability_controlled --n 8 --rounds 2
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import os
import random

from data.common import extract_code

from .repairability import (
    STUDENT,  # noqa: F401  (kept for parity / not used here)
    TEACHER,
    all_pass,
    defines_entry,
    error_feedback,
    pass_fraction,
    repair_prompt,
    split_tests,
)
from .sandbox import run_one

# ---------- ground-truth I/O from the canonical solution ----------

def build_io(task: dict, max_tests: int) -> tuple[str, list[tuple[list, object]]]:
    """(canonical source, [(input_list, expected_output), ...]) validated by exec."""
    ep = task["entry_point"]
    src = task["prompt"] + task["canonical_solution"]
    ns: dict = {}
    try:
        exec(src, ns)
    except Exception:
        return src, []
    fn = ns.get(ep)
    if not callable(fn):
        return src, []
    inputs = list(task.get("base_input") or []) + list(task.get("plus_input") or [])
    io = []
    for inp in inputs[:max_tests]:
        try:
            out = fn(*copy.deepcopy(inp))
        except Exception:
            continue
        t = f"assert {ep}({', '.join(map(repr, inp))}) == {out!r}"
        if run_one(src + "\n", t, None).passed:  # repr round-trips
            io.append((list(inp), out))
    return src, io


def assert_of(ep: str, inp: list, out) -> str:
    return f"assert {ep}({', '.join(map(repr, inp))}) == {out!r}"


# ---------- population A: near-correct (single AST mutation) ----------

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}


def mutate_once(src: str, seed: int) -> str | None:
    """Apply ONE small mutation to `src` (comparison flip / op swap / +-1 const)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    cands = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in _CMP:
            cands.append(("cmp", node))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            cands.append(("bin", node))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, int)
              and not isinstance(node.value, bool)):
            cands.append(("int", node))
    if not cands:
        return None
    kind, node = random.Random(seed).choice(cands)
    if kind == "cmp":
        node.ops[0] = _CMP[type(node.ops[0])]()
    elif kind == "bin":
        node.op = _BIN[type(node.op)]()
    else:
        node.value = node.value + random.Random(seed + 7).choice([-1, 1])
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def make_near_correct(src: str, ep: str, heldout: list[str], k: int) -> list[str]:
    """Up to k single-mutation variants that still define the fn and FAIL held-out."""
    out, seen = [], set()
    for seed in range(60):
        m = mutate_once(src, seed)
        if not m or m in seen or m == src:
            continue
        seen.add(m)
        if defines_entry(m, ep) and not all_pass(m, heldout):
            out.append(m)
        if len(out) >= k:
            break
    return out


# ---------- population B: hard-coded to the visible tests ----------

def make_hardcoded(ep: str, visible_io: list[tuple[list, object]]) -> str:
    lines = [f"def {ep}(*args):"]
    for inp, out in visible_io:
        lines.append(f"    if args == {tuple(inp)!r}: return {out!r}")
    lines.append("    return None  # no algorithm -> fails held-out")
    return "\n".join(lines)


# ---------- shared repair loop (fixed 7B repairer) ----------

def repair_all(teacher, tok_t, items: list[dict], rounds: int):
    from vllm import SamplingParams

    for rnd in range(1, rounds + 1):
        active = [it for it in items if not it["done"] and not all_pass(it["code"], it["heldout"])]
        if not active:
            break
        prompts = [repair_prompt(tok_t, it["prompt_text"], it["entry_point"],
                                 it["code"], error_feedback(it["code"], it["visible"]))
                   for it in active]
        outs = teacher.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=512))
        for it, o in zip(active, outs):
            new = extract_code(o.outputs[0].text)
            if new.strip() == it["code"].strip():
                it["done"] = True
            else:
                it["code"] = new
                it["rounds"] = rnd
        print(f"  round {rnd}: repaired {len(active)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    p.add_argument("--teacher", default=TEACHER)
    p.add_argument("--n", type=int, default=8, help="number of tasks (smoke: 8)")
    p.add_argument("--max-tests", type=int, default=12)
    p.add_argument("--mutants", type=int, default=2, help="near-correct variants per task")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--tau", type=float, default=0.6, help="sim threshold for a LOCAL repair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/repairability_controlled.json")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    tasks = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()

    # --- build the two constructed populations (CPU) ---
    items: list[dict] = []
    picked = 0
    for tid, t in tasks.items():
        if picked >= args.n:
            break
        src, io = build_io(t, args.max_tests)
        if len(io) < 4:
            continue
        vis_io, held_io = _split_io(io, args.seed)
        ep = t["entry_point"]
        visible = [assert_of(ep, i, o) for i, o in vis_io]
        heldout = [assert_of(ep, i, o) for i, o in held_io]

        near = make_near_correct(src, ep, heldout, args.mutants)
        hard = make_hardcoded(ep, vis_io)
        if not near or all_pass(hard, heldout):  # need a real bug and a real brute
            continue
        picked += 1
        for code in near:
            items.append(_mk(tid, t["prompt"], ep, visible, heldout, code, "near_correct"))
        items.append(_mk(tid, t["prompt"], ep, visible, heldout, hard, "hard_coded"))

    print(f"{picked} tasks -> {sum(i['pop']=='near_correct' for i in items)} near-correct, "
          f"{sum(i['pop']=='hard_coded' for i in items)} hard-coded")
    if not items:
        print("no constructed items (try more tasks)")
        return

    # --- repair with the fixed 7B teacher ---
    tok_t = AutoTokenizer.from_pretrained(args.teacher)
    teacher = LLM(model=args.teacher, max_model_len=2048)
    repair_all(teacher, tok_t, items, args.rounds)
    del teacher

    # --- score ---
    rows = []
    for it in items:
        repairable = int(all_pass(it["code"], it["heldout"]))
        sim = difflib.SequenceMatcher(None, it["orig_code"], it["code"]).ratio()
        rows.append({
            "task": it["task"], "pop": it["pop"],
            "partial_reward": round(pass_fraction(it["orig_code"], it["visible"]), 3),
            "repairability": repairable,
            "local_repair": int(repairable and sim >= args.tau),
            "rounds": it["rounds"], "sim": round(sim, 3),
            "orig_code": it["orig_code"], "final_code": it["code"],
        })
    _report(rows, args.tau)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"config": vars(args), "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


def _split_io(io, seed):
    idx = list(range(len(io)))
    random.Random(seed).shuffle(idx)
    half = max(1, len(io) // 2)
    return [io[i] for i in idx[:half]], [io[i] for i in idx[half:]] or [io[idx[-1]]]


def _mk(tid, prompt, ep, visible, heldout, code, pop):
    return {"task": tid, "prompt_text": prompt, "entry_point": ep,
            "visible": visible, "heldout": heldout, "orig_code": code, "code": code,
            "pop": pop, "rounds": 0, "done": False}


def _report(rows, tau):
    import numpy as np

    print(f"\n=== Controlled repairability: {len(rows)} constructed solutions ===")
    print(f"{'population':<14}{'n':>4}{'mean partial':>14}{'reach':>8}{'LocalRepair':>13}{'mean sim':>10}")
    for pop in ("near_correct", "hard_coded"):
        sub = [r for r in rows if r["pop"] == pop]
        if not sub:
            continue
        n = len(sub)
        print(f"{pop:<14}{n:>4}"
              f"{np.mean([r['partial_reward'] for r in sub]):>14.3f}"
              f"{np.mean([r['repairability'] for r in sub]):>8.2f}"
              f"{np.mean([r['local_repair'] for r in sub]):>13.2f}"
              f"{np.mean([r['sim'] for r in sub]):>10.2f}")
    near = [r for r in rows if r["pop"] == "near_correct"]
    hard = [r for r in rows if r["pop"] == "hard_coded"]
    if near and hard:
        pn = np.mean([r["partial_reward"] for r in near])
        ph = np.mean([r["partial_reward"] for r in hard])
        ln = np.mean([r["local_repair"] for r in near])
        lh = np.mean([r["local_repair"] for r in hard])
        print(f"\nINVERSION CHECK: partial  hard {ph:.2f} vs near {pn:.2f}  "
              f"(hard should be HIGHER)")
        print(f"                 LocalRepair near {ln:.2f} vs hard {lh:.2f}  "
              f"(near should be HIGHER)")
        if ph >= pn and ln > lh:
            print("  => INVERSION CONFIRMED: partial ranks hard-coded above near-correct,")
            print("     repairability ranks near-correct above hard-coded. The idea works.")
        else:
            print("  => no clean inversion (see numbers / scale up / check construction).")


if __name__ == "__main__":
    main()
