"""Week-1 gate (v2 §7 D1-2): prove the eval pipeline is trustworthy BEFORE any
training. Two checks, both CPU-only:

  1. Ground-truth check: every canonical solution must pass its base test
     (pass rate == 100%). A failure here means our runner/harness is broken.
  2. Classifier spot-check: deliberately corrupted solutions must be labeled
     syntax_error / runtime_error / logic_error / timeout correctly.

Run:  python -m eval.verify_pipeline --n 50
"""

from __future__ import annotations

import argparse

from .error_classify import classify
from .sandbox import run_one


def check_ground_truth(n: int) -> tuple[int, int, list[str]]:
    """Run n canonical solutions against their base tests. Return (passed, total, failures)."""
    from evalplus.data import get_human_eval_plus

    tasks = list(get_human_eval_plus().values())[:n]
    passed, failures = 0, []
    for t in tasks:
        code = t["prompt"] + t["canonical_solution"]
        res = run_one(code, t["test"], entry_point=t["entry_point"])
        if res.passed:
            passed += 1
        else:
            failures.append(f"{t['task_id']}: rc={res.returncode} {res.exception_type} {res.stderr[:80]}")
    return passed, len(tasks), failures


def check_classifier() -> tuple[int, int, list[str]]:
    """Corrupt a known-good solution four ways; verify the four labels."""
    from evalplus.data import get_human_eval_plus

    t = get_human_eval_plus()["HumanEval/0"]  # has_close_elements
    good = t["prompt"] + t["canonical_solution"]
    test, ep = t["test"], t["entry_point"]

    cases = [
        ("correct", good),
        ("syntax_error", good + "\n    return  (("),            # unbalanced paren
        ("runtime_error", "def has_close_elements(numbers, threshold):\n    return undefined_name"),
        ("logic_error", "def has_close_elements(numbers, threshold):\n    return True"),
        ("timeout", "def has_close_elements(numbers, threshold):\n    while True:\n        pass"),
    ]
    ok, mism = 0, []
    for gold, code in cases:
        got = classify(code, test, entry_point=ep, timeout=5)
        if got == gold:
            ok += 1
        else:
            mism.append(f"expected {gold}, got {got}")
    return ok, len(cases), mism


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="# canonical solutions to verify")
    args = p.parse_args()

    print(f"[1/2] ground-truth check on {args.n} HumanEval+ canonical solutions...")
    passed, total, failures = check_ground_truth(args.n)
    print(f"      pass rate: {passed}/{total} = {100*passed/total:.1f}%")
    for f in failures[:10]:
        print("      FAIL", f)

    print("[2/2] classifier spot-check...")
    ok, n, mism = check_classifier()
    print(f"      correct labels: {ok}/{n}")
    for m in mism:
        print("      MISMATCH", m)

    gate_ok = passed == total and ok == n
    print("\nGATE:", "PASS ✅" if gate_ok else "FAIL ❌")
    raise SystemExit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
