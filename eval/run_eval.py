"""Evaluation driver (v2 §4, RQ4).

Metrics:
  - greedy pass@1 (primary)
  - pass@k for k in {1,4,16,64} at temperature 0.8 (diversity / ceiling, RQ4)
  - error-type breakdown per checkpoint (RQ2)
  - teacher-student win matrix (OPD only): does the student solve problems the
    teacher got wrong? -> evidence OPD amplifies latent ability, not pure mimicry.

Benchmarks are EVALUATION-ONLY: HumanEval+ (164) and the MBPP+ eval subset.
They must never appear in any training pool (see data/contamination_audit.py).
"""

from __future__ import annotations

import argparse

PASS_AT_K_VALUES = (1, 4, 16, 64)
SAMPLING_TEMPERATURE = 0.8
NUM_SAMPLES = 64


def pass_at_1_greedy(model, dataset) -> float:
    """Greedy decode, one sample/problem, fraction passing all tests."""
    raise NotImplementedError


def pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    TODO: 1 - C(n-c, k)/C(n, k), numerically stable product form.
    """
    raise NotImplementedError


def error_breakdown(model, dataset) -> dict[str, float]:
    """Percentage of syntax/runtime/logic/timeout across the dataset."""
    raise NotImplementedError


def teacher_student_win_matrix(teacher, student, dataset) -> dict:
    """Per-problem 2x2 contingency of teacher-correct x student-correct (RQ4)."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["humaneval_plus", "mbpp_plus"], required=True)
    p.add_argument("--mode", choices=["greedy", "passk"], default="greedy")
    p.add_argument("--out", default="results/eval.json")
    args = p.parse_args()
    # TODO: load model (vLLM), run selected mode, dump per-problem results to --out
    raise NotImplementedError


if __name__ == "__main__":
    main()
