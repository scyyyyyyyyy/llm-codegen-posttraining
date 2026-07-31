"""Near-duplicate audit between the training pool and the eval sets (A1 §3).

Dual filter, both reported:
  1. exact entry-point / function-signature collision
  2. word n-gram overlap between problem statements (standard decontamination,
     cf. GPT-3 / Llama): a shared contiguous n-gram is strong evidence of
     duplication. Pure-Python, no heavy deps.

Usage:
  python -m data.contamination_audit --pool data/prompt_pool.jsonl \
      --out data/prompt_pool.clean.jsonl --report results/contamination_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

from .common import read_jsonl, write_jsonl

NGRAM = 8


def _eval_items() -> list[dict]:
    """HumanEval+ / MBPP+ problems as {entry_point, statement}."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    items = []
    for t in get_human_eval_plus().values():
        items.append({"entry_point": t["entry_point"], "statement": t["prompt"]})
    for t in get_mbpp_plus().values():
        items.append({"entry_point": t["entry_point"], "statement": t["prompt"]})
    return items


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(words: list[str], n: int) -> set[tuple]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def signature_collisions(pool: list[dict], eval_items: list[dict]) -> set[str]:
    """Pool ids whose entry_point exactly matches an eval entry_point."""
    eval_eps = {e["entry_point"] for e in eval_items}
    return {p["id"] for p in pool if p["entry_point"] in eval_eps}


def ngram_collisions(pool: list[dict], eval_items: list[dict],
                     n: int = NGRAM) -> set[str]:
    """Pool ids sharing any word n-gram with an eval statement."""
    eval_ngrams: set[tuple] = set()
    for e in eval_items:
        eval_ngrams |= _ngrams(_words(e["statement"]), n)
    hits = set()
    for p in pool:
        if _ngrams(_words(p["prompt_text"]), n) & eval_ngrams:
            hits.add(p["id"])
    return hits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/prompt_pool.jsonl")
    p.add_argument("--out", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--report", default="results/contamination_report.json")
    p.add_argument("--ngram", type=int, default=NGRAM)
    args = p.parse_args()

    pool = read_jsonl(args.pool)
    eval_items = _eval_items()

    sig = signature_collisions(pool, eval_items)
    ng = ngram_collisions(pool, eval_items, args.ngram)
    removed = sig | ng
    clean = [r for r in pool if r["id"] not in removed]

    report = {
        "n_before": len(pool),
        "removed_signature": len(sig),
        "removed_ngram_only": len(ng - sig),
        "removed_total": len(removed),
        "n_after": len(clean),
        "ngram_n": args.ngram,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=2)
    write_jsonl(args.out, clean)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
