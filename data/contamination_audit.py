"""Near-duplicate audit between the training pool and the eval sets (v2 §4.2).

Dual filter:
  1. exact function-signature match
  2. problem-statement embedding similarity (cosine >= ~0.9)

Removes any training problem too close to a HumanEval+/MBPP+ eval problem, and
REPORTS the number of removed items (write the count into the repo/blog). A real
contamination audit is what most portfolio projects lack -- cheap, high-trust.
"""

from __future__ import annotations

import argparse

SIM_THRESHOLD = 0.9


def signature_collisions(pool, eval_sets) -> set:
    """Training problem ids whose function signature exactly matches an eval item."""
    raise NotImplementedError


def embedding_collisions(pool, eval_sets, threshold: float = SIM_THRESHOLD) -> set:
    """Training problem ids with statement embedding cosine >= threshold vs eval."""
    raise NotImplementedError  # TODO: embed statements, cosine sim, threshold


def audit(pool_path: str, out_path: str) -> dict:
    """Run both filters, drop collisions, write cleaned pool + a JSON report.

    Report fields: n_before, n_removed_signature, n_removed_embedding, n_after.
    """
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/prompt_pool.jsonl")
    p.add_argument("--out", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--report", default="results/contamination_report.json")
    args = p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
