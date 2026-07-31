"""Near-duplicate audit between the training pool and the eval sets (A1 §3).

Dual filter, both reported:
  1. exact entry-point / function-signature collision
  2. problem-statement embedding cosine >= threshold

Removes any training item too close to a HumanEval+/MBPP+ item and writes a report.

Usage:
  python -m data.contamination_audit --pool data/prompt_pool.jsonl \
      --out data/prompt_pool.clean.jsonl --report results/contamination_report.json
"""

from __future__ import annotations

import argparse
import json
import os

from .common import read_jsonl, write_jsonl

SIM_THRESHOLD = 0.90
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _eval_items() -> list[dict]:
    """HumanEval+ / MBPP+ problems as {entry_point, statement}."""
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    items = []
    for t in get_human_eval_plus().values():
        items.append({"entry_point": t["entry_point"], "statement": t["prompt"]})
    for t in get_mbpp_plus().values():
        items.append({"entry_point": t["entry_point"], "statement": t["prompt"]})
    return items


def signature_collisions(pool: list[dict], eval_items: list[dict]) -> set[str]:
    """Pool ids whose entry_point exactly matches an eval entry_point."""
    eval_eps = {e["entry_point"] for e in eval_items}
    return {p["id"] for p in pool if p["entry_point"] in eval_eps}


def embedding_collisions(pool: list[dict], eval_items: list[dict],
                         threshold: float = SIM_THRESHOLD) -> set[str]:
    """Pool ids whose statement embedding cosine >= threshold vs any eval item."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer(EMBED_MODEL)
    pool_emb = model.encode([p["prompt_text"] for p in pool],
                            convert_to_tensor=True, normalize_embeddings=True)
    eval_emb = model.encode([e["statement"] for e in eval_items],
                            convert_to_tensor=True, normalize_embeddings=True)
    sims = cos_sim(pool_emb, eval_emb)          # [n_pool, n_eval]
    maxsim = sims.max(dim=1).values
    return {pool[i]["id"] for i in range(len(pool)) if float(maxsim[i]) >= threshold}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/prompt_pool.jsonl")
    p.add_argument("--out", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--report", default="results/contamination_report.json")
    p.add_argument("--threshold", type=float, default=SIM_THRESHOLD)
    args = p.parse_args()

    pool = read_jsonl(args.pool)
    eval_items = _eval_items()

    sig = signature_collisions(pool, eval_items)
    emb = embedding_collisions(pool, eval_items, args.threshold)
    removed = sig | emb
    clean = [r for r in pool if r["id"] not in removed]

    report = {
        "n_before": len(pool),
        "removed_signature": len(sig),
        "removed_embedding_only": len(emb - sig),
        "removed_total": len(removed),
        "n_after": len(clean),
        "threshold": args.threshold,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=2)
    write_jsonl(args.out, clean)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
