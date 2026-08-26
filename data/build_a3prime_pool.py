"""Build the fixed visible/held-out training split for A3-prime.

The repository's clean pool contains 310 MBPP prompts with three raw test
assertions and 82 KodCode prompts with one test block.  A few MBPP rows repeat
an assertion verbatim, so tests are de-duplicated before splitting.  The
original A3-prime configuration selected three tests, which is effectively
identical to A3.  This builder retains prompts with enough independent tests,
assigns a deterministic visible subset, and keeps the remainder strictly held
out from the reward.
"""

from __future__ import annotations

import argparse
import hashlib
import random

from .common import read_jsonl, write_jsonl


def split_item(item: dict, visible_count: int, seed: int) -> dict | None:
    # A held-out assertion must provide information that was not already
    # exposed to the reward. Preserve order while collapsing exact duplicates.
    tests = list(dict.fromkeys(item["tests"]))
    if len(tests) <= visible_count:
        return None
    digest = hashlib.sha256(f"{seed}:{item['id']}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(len(tests)))
    rng.shuffle(order)
    visible_idx = set(order[:visible_count])
    out = dict(item)
    out["all_tests"] = tests
    out["tests"] = [t for i, t in enumerate(tests) if i in visible_idx]
    out["heldout_tests"] = [t for i, t in enumerate(tests) if i not in visible_idx]
    out["a3prime_split_seed"] = seed
    return out


def matched_control_item(split: dict) -> dict:
    """Use the same retained prompt but expose every test to A3's reward."""
    out = dict(split)
    out["tests"] = list(split["all_tests"])
    out["heldout_tests"] = []
    out["a3prime_condition"] = "matched_full_reward_control"
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--out", default="data/prompt_pool.a3prime.jsonl")
    p.add_argument(
        "--control-out",
        default="data/prompt_pool.a3matched.jsonl",
        help="same retained prompts with every test exposed to the A3 reward",
    )
    p.add_argument("--visible-count", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260818)
    args = p.parse_args()
    if args.visible_count < 1:
        p.error("--visible-count must be positive")

    rows = read_jsonl(args.pool)
    split = [x for r in rows if (x := split_item(r, args.visible_count, args.seed))]
    write_jsonl(args.out, split)
    if args.control_out:
        write_jsonl(args.control_out, [matched_control_item(x) for x in split])
    print(
        f"A3-prime pool: {len(split)}/{len(rows)} prompts; "
        f"visible={args.visible_count}, held-out>=1 -> {args.out}; "
        f"matched A3 control -> {args.control_out or 'disabled'}"
    )


if __name__ == "__main__":
    main()
