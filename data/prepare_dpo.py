"""Construct DPO preference pairs from scored candidates.

Three strategies, compared in the ablation:
  A. binary       : chosen = any test-passing solution, rejected = any failing one
  B. partial      : chosen = highest partial_reward, rejected = lowest
                    (the core experiment — yields pairs even when nothing fully
                    passes, so hard problems still contribute signal)
  C. error_aware  : chosen = correct solution, rejected = best logic_error
                    (teaches DPO to specifically fix wrong-output bugs)

Length bias guard: drop pairs where the chosen code is much longer than the
rejected one, so DPO learns correctness rather than verbosity.
"""

from __future__ import annotations

import argparse
import json
import random

MIN_REWARD_GAP = 0.2  # drop near-tie pairs (both ~equally bad)
MAX_LEN_RATIO = 1.5  # drop pairs where chosen is >1.5x longer than rejected


def _length_ok(chosen: str, rejected: str) -> bool:
    lr = len(rejected) or 1
    return len(chosen) / lr <= MAX_LEN_RATIO


def build_binary_pairs(data: dict[str, list[dict]]) -> list[dict]:
    pairs = []
    for candidates in data.values():
        correct = [c for c in candidates if c["binary_reward"] == 1.0]
        incorrect = [c for c in candidates if c["binary_reward"] == 0.0]
        if correct and incorrect:
            ch, rj = random.choice(correct)["code"], random.choice(incorrect)["code"]
            if _length_ok(ch, rj):
                pairs.append({"chosen": ch, "rejected": rj})
    return pairs


def build_partial_pairs(data: dict[str, list[dict]]) -> list[dict]:
    pairs = []
    for candidates in data.values():
        s = sorted(candidates, key=lambda x: x["partial_reward"])
        gap = s[-1]["partial_reward"] - s[0]["partial_reward"]
        if gap >= MIN_REWARD_GAP and _length_ok(s[-1]["code"], s[0]["code"]):
            pairs.append(
                {"chosen": s[-1]["code"], "rejected": s[0]["code"], "reward_gap": gap}
            )
    return pairs


def build_error_aware_pairs(data: dict[str, list[dict]]) -> list[dict]:
    pairs = []
    for candidates in data.values():
        correct = [c for c in candidates if c["binary_reward"] == 1.0]
        logic_err = [c for c in candidates if c["error_type"] == "logic_error"]
        if correct and logic_err:
            best_wrong = max(logic_err, key=lambda x: x["partial_reward"])
            ch, rj = random.choice(correct)["code"], best_wrong["code"]
            if _length_ok(ch, rj):
                pairs.append({"chosen": ch, "rejected": rj})
    return pairs


STRATEGIES = {
    "binary": build_binary_pairs,
    "partial": build_partial_pairs,
    "error_aware": build_error_aware_pairs,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="partial")
    parser.add_argument("--out", default="data/dpo_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    with open(args.candidates) as f:
        data = json.load(f)

    pairs = STRATEGIES[args.strategy](data)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"built {len(pairs)} {args.strategy} pairs -> {args.out}")


if __name__ == "__main__":
    main()
