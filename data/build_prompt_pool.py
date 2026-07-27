"""Assemble the contamination-safe training prompt pool (v2 §4.2).

Target: 800-1500 problems, EACH with an executable unit test.

Sources:
  - MBPP original train split (374 problems, 3 asserts each; Austin et al. 2021)
  - TACO / APPS intro-medium function-style subset (own tests); drop complex
    stdin/stdout IO, keep function-style problems
  - optional: GPT-4o synthetic problems + synthetic tests (< $10); each synthetic
    test must first pass GPT-4o's own solution AND be non-trivial

HARD RULE: HumanEval+ / MBPP+ evaluation sets NEVER enter this pool. HumanEval+
has no train split (164 eval-only problems); MBPP+ is an augmented TEST set.
Training on either = train-test contamination -> all numbers void.
"""

from __future__ import annotations

import argparse

TARGET_MIN, TARGET_MAX = 800, 1500


def load_mbpp_train():
    """MBPP train split with its 3 built-in asserts. TODO: via datasets."""
    raise NotImplementedError


def load_taco_apps_function_subset():
    """Intro-medium function-style TACO/APPS problems with tests.

    TODO: filter out stdin/stdout IO problems; keep pure-function signatures.
    """
    raise NotImplementedError


def prefilter_by_ground_truth(problems):
    """Drop any problem whose ground-truth solution fails its own tests.

    Guards against bad TACO/APPS test quality (risk table, v2 §11).
    """
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/prompt_pool.jsonl")
    p.add_argument("--with-synthetic", action="store_true")
    args = p.parse_args()
    # TODO: merge sources -> prefilter -> write JSONL -> then run contamination_audit.py
    raise NotImplementedError


if __name__ == "__main__":
    main()
