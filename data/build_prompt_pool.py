"""Assemble the contamination-safe training prompt pool (A1 §3).

Sources: MBPP train split + KodCode (function-style Python with tests). Each item
is normalized to the schema in data/common.py and pre-filtered so that its
reference solution passes its own tests (guards against bad tests).

Usage: python -m data.build_prompt_pool --out data/prompt_pool.jsonl
"""

from __future__ import annotations

import argparse

from eval.sandbox import run_one

from .common import extract_entry_point, write_jsonl

TARGET_MIN, TARGET_MAX = 800, 1500


def load_mbpp_train() -> list[dict]:
    """MBPP train split -> pool schema. Tests are plain asserts calling the fn.

    Newer `datasets` requires a namespaced repo id (bare "mbpp" no longer resolves);
    the "full" config's train split (374 problems) is disjoint from the test split
    that MBPP+ derives from.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    rows = []
    for ex in ds:
        code = ex["code"]
        ep = extract_entry_point(code)
        if not ep:
            continue
        rows.append({
            "id": f"mbpp/{ex['task_id']}",
            "source": "mbpp",
            "prompt_text": ex["text"],
            "entry_point": ep,
            "tests": list(ex["test_list"]),
            "reference_solution": code,
        })
    return rows


def load_kodcode(limit: int = 1000) -> list[dict]:
    """KodCode -> pool schema. Field names vary by release; adapt on load."""
    from datasets import load_dataset

    ds = load_dataset("KodCode/KodCode-V1", split="train", streaming=True)
    rows = []
    for ex in ds:
        # Common field names across releases; adjust if your snapshot differs.
        question = ex.get("question") or ex.get("prompt") or ex.get("instruction")
        solution = ex.get("solution") or ex.get("response") or ex.get("code")
        test = ex.get("test") or ex.get("test_code") or ex.get("test_list")
        lang = (ex.get("language") or ex.get("lang") or "python").lower()
        if not (question and solution and test) or "python" not in lang:
            continue
        ep = extract_entry_point(solution)
        if not ep:
            continue
        tests = test if isinstance(test, list) else [test]
        rows.append({
            "id": f"kodcode/{ex.get('id', len(rows))}",
            "source": "kodcode",
            "prompt_text": question,
            "entry_point": ep,
            "tests": [t for t in tests if "assert" in t],
            "reference_solution": solution,
        })
        if len(rows) >= limit:
            break
    return [r for r in rows if r["tests"]]


def reference_passes(item: dict, timeout: int = 10) -> bool:
    """True iff the reference solution passes all of its own tests."""
    code = item["reference_solution"]
    return all(run_one(code, t, entry_point=None, timeout=timeout).passed
               for t in item["tests"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/prompt_pool.jsonl")
    p.add_argument("--kodcode-limit", type=int, default=900)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    pool = load_mbpp_train() + load_kodcode(args.kodcode_limit)
    print(f"loaded {len(pool)} raw problems")

    kept, dropped = [], 0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for item, ok in zip(pool, ex.map(reference_passes, pool)):
            if ok:
                kept.append(item)
            else:
                dropped += 1

    write_jsonl(args.out, kept)
    print(f"pre-filter: kept {len(kept)}, dropped {dropped} (bad reference/tests)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
