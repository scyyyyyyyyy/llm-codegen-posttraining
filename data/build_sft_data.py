"""Build distilled SFT data via teacher rejection sampling (v2 §4.2).

For each problem in the clean prompt pool: sample k=4 solutions from the TEACHER
(Qwen2.5-Coder-7B-Instruct), keep at most 1 solution that passes ALL visible
tests. Yields ~700-1200 "distilled SFT" examples.

Key design: SFT and OPD share the SAME teacher.
  SFT = teacher's OFF-policy distillation
  OPD = teacher's ON-policy distillation
Their difference isolates the value of on-policyness itself (exposure-bias fix,
cf. DAgger, Ross et al. 2011).
"""

from __future__ import annotations

import argparse

TEACHER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SAMPLES_PER_PROBLEM = 4

SYSTEM_PROMPT = "You are an expert Python programmer. Write clean, correct code."


def teacher_sample(problems, k: int = SAMPLES_PER_PROBLEM):
    """vLLM-serve the teacher, sample k solutions per problem. TODO."""
    raise NotImplementedError


def keep_passing(problem, candidates):
    """Return one candidate passing all visible tests, or None. TODO."""
    raise NotImplementedError


def to_chat_record(problem, solution) -> dict:
    """Chat-format SFT record. TODO: system / user(problem+signature) / assistant."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--out", default="data/sft.jsonl")
    args = p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
