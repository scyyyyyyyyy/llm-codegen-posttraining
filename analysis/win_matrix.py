"""Paired teacher-student win matrices from raw EvalPlus task outcomes.

Aggregate pass rates are insufficient for RQ4: a student can match the teacher's
rate while solving a different set of tasks. This module records the four paired
cells (both, teacher-only, student-only, neither) and preserves the task IDs for
auditability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _solved(flags: list[bool], any_sample: bool) -> bool:
    if not flags:
        raise ValueError("each task must contain at least one evaluated sample")
    return any(flags) if any_sample else bool(flags[0])


def win_matrix(
    teacher: dict[str, list[bool]],
    student: dict[str, list[bool]],
    *,
    any_sample: bool = False,
) -> dict:
    """Return the exact paired 2x2 matrix and its constituent task IDs."""
    if set(teacher) != set(student):
        teacher_only_ids = sorted(set(teacher) - set(student))
        student_only_ids = sorted(set(student) - set(teacher))
        raise ValueError(
            "teacher/student task sets differ: "
            f"missing_from_student={teacher_only_ids[:5]}, "
            f"missing_from_teacher={student_only_ids[:5]}"
        )
    if not teacher:
        raise ValueError("empty evaluation")

    cells = {
        "both_solve": [],
        "teacher_only": [],
        "student_only": [],
        "neither_solves": [],
    }
    for task_id in sorted(teacher):
        teacher_ok = _solved(teacher[task_id], any_sample)
        student_ok = _solved(student[task_id], any_sample)
        if teacher_ok and student_ok:
            cell = "both_solve"
        elif teacher_ok:
            cell = "teacher_only"
        elif student_ok:
            cell = "student_only"
        else:
            cell = "neither_solves"
        cells[cell].append(task_id)

    n = len(teacher)
    counts = {name: len(ids) for name, ids in cells.items()}
    return {
        "n_tasks": n,
        "criterion": "any_sample" if any_sample else "greedy_first_sample",
        # Rows are teacher fail/pass; columns are student fail/pass.
        "matrix": [
            [counts["neither_solves"], counts["student_only"]],
            [counts["teacher_only"], counts["both_solve"]],
        ],
        "counts": counts,
        "rates": {name: value / n for name, value in counts.items()},
        "teacher_pass_rate": (
            counts["teacher_only"] + counts["both_solve"]
        ) / n,
        "student_pass_rate": (
            counts["student_only"] + counts["both_solve"]
        ) / n,
        "student_minus_teacher": (
            counts["student_only"] - counts["teacher_only"]
        ) / n,
        "task_ids": cells,
    }


def main() -> None:
    from eval.run_eval import parse_eval_results

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-eval", required=True)
    parser.add_argument("--student-eval", required=True)
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--teacher-tag", default="qwen7b")
    parser.add_argument("--student-tag", required=True)
    parser.add_argument("--any-sample", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    teacher = parse_eval_results(args.teacher_eval, use_plus=True)
    student = parse_eval_results(args.student_eval, use_plus=True)
    result = win_matrix(teacher, student, any_sample=args.any_sample)
    result.update(
        {
            "dataset": args.dataset,
            "teacher_tag": args.teacher_tag,
            "student_tag": args.student_tag,
            # Preserve caller-provided labels instead of publishing
            # machine-specific absolute paths.
            "teacher_eval": args.teacher_eval,
            "student_eval": args.student_eval,
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "dataset", "teacher_tag", "student_tag", "matrix",
        "teacher_pass_rate", "student_pass_rate", "student_minus_teacher",
    )}, indent=2))


if __name__ == "__main__":
    main()
