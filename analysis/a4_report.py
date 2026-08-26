"""Validate and aggregate the public A4 on-policy-distillation artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


DATASETS = {"humaneval": 164, "mbpp": 378}
A4_TAGS = ("a4-opd-s0", "a4-opd-s1", "a4-opd-s2")
A1_TAGS = ("a1", "a1-s1", "a1-s2")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_curve(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 98:
        raise ValueError(f"{path}: expected 98 optimizer records, got {len(records)}")
    if [item["optimizer_step"] for item in records] != list(range(1, 99)):
        raise ValueError(f"{path}: optimizer steps are not exactly 1..98")
    if records[-1]["rollouts_completed"] != 3136:
        raise ValueError(f"{path}: rollout budget is incomplete")
    if any(item["rollouts_this_update"] != 32 for item in records):
        raise ValueError(f"{path}: each optimizer step must contain 32 rollouts")
    numeric_keys = (
        "sampled_reverse_kl_per_token",
        "student_nll_per_token",
        "teacher_nll_per_token",
        "gradient_norm",
        "cumulative_completion_tokens",
        "elapsed_seconds",
        "max_gpu_memory_allocated_bytes",
    )
    if not all(
        math.isfinite(float(item[key])) for item in records for key in numeric_keys
    ):
        raise ValueError(f"{path}: curve contains non-finite values")
    return records


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two values are required")
    x_bar = (len(values) - 1) / 2
    y_bar = statistics.mean(values)
    denominator = sum((x - x_bar) ** 2 for x in range(len(values)))
    return sum((x - x_bar) * (y - y_bar) for x, y in enumerate(values)) / denominator


def summarize_curve(records: list[dict]) -> dict:
    values = [float(item["sampled_reverse_kl_per_token"]) for item in records]
    window = 10
    return {
        "records": len(records),
        "rollouts": records[-1]["rollouts_completed"],
        "completion_tokens": records[-1]["cumulative_completion_tokens"],
        "first_10_mean_sampled_kl": statistics.mean(values[:window]),
        "last_10_mean_sampled_kl": statistics.mean(values[-window:]),
        "last_minus_first_10_mean_sampled_kl": (
            statistics.mean(values[-window:]) - statistics.mean(values[:window])
        ),
        "linear_slope_per_update": linear_slope(values),
        "elapsed_seconds": records[-1]["elapsed_seconds"],
        "peak_gpu_memory_bytes": max(
            item["max_gpu_memory_allocated_bytes"] for item in records
        ),
    }


def validate_summary(
    path: Path,
    dataset: str,
    *,
    tag: str,
    arm: str,
    require_passk: bool = False,
) -> dict:
    payload = load_json(path)
    expected = {
        "dataset": dataset,
        "tag": tag,
        "arm": arm,
        "n_tasks": DATASETS[dataset],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}")
    pass1 = float(payload["pass@1_plus"])
    if not 0 <= pass1 <= 1:
        raise ValueError(f"{path}: invalid pass@1_plus")
    passk = payload.get("pass@k_plus", {})
    if require_passk and set(passk) != {"1", "4", "16", "64"}:
        raise ValueError(f"{path}: incomplete pass@k")
    if any(not 0 <= float(value) <= 1 for value in passk.values()):
        raise ValueError(f"{path}: invalid pass@k value")
    return payload


def validate_matrix(path: Path, dataset: str, tag: str) -> dict:
    payload = load_json(path)
    expected = {
        "dataset": dataset,
        "student_tag": tag,
        "teacher_tag": "qwen7b",
        "n_tasks": DATASETS[dataset],
        "criterion": "greedy_first_sample",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}")
    for key in ("student_eval", "teacher_eval"):
        if key in payload and Path(payload[key]).is_absolute():
            raise ValueError(f"{path}: contains a machine-specific absolute path")

    counts = payload["counts"]
    names = ("neither_solves", "student_only", "teacher_only", "both_solve")
    if sum(int(counts[name]) for name in names) != DATASETS[dataset]:
        raise ValueError(f"{path}: counts do not partition the tasks")
    expected_matrix = [
        [counts["neither_solves"], counts["student_only"]],
        [counts["teacher_only"], counts["both_solve"]],
    ]
    if payload.get("matrix") != expected_matrix:
        raise ValueError(f"{path}: matrix/count mismatch")
    task_ids = payload["task_ids"]
    if any(len(task_ids[name]) != counts[name] for name in names):
        raise ValueError(f"{path}: task-id/count mismatch")
    flattened = [task for name in names for task in task_ids[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"{path}: task IDs are not a disjoint partition")
    return payload


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "seeds": len(values),
    }


def build_report(results: Path) -> dict:
    manifest = load_json(results / "a4_artifact_manifest.json")
    if manifest.get("scope") != "A4 OPD, seeds 0/1/2":
        raise ValueError("A4 artifact manifest has the wrong scope")

    report: dict = {
        "status": "verified",
        "design": manifest["design"],
        "provenance": "results/a4_artifact_manifest.json",
        "A4": {},
        "teacher_reference": {},
        "seed2_passk_comparison": {},
    }
    a4_values = {dataset: [] for dataset in DATASETS}
    a1_values = {dataset: [] for dataset in DATASETS}
    kl_changes = []

    teacher = {}
    for dataset in DATASETS:
        reconstructed = validate_summary(
            results / f"a0_qwen7b-matrix_{dataset}.json",
            dataset,
            tag="qwen7b-matrix",
            arm="A0'",
        )
        published = validate_summary(
            results / f"a0_qwen7b_{dataset}.json",
            dataset,
            tag="qwen7b",
            arm="A0'",
        )
        teacher[dataset] = reconstructed
        report["teacher_reference"][dataset] = {
            "paired_reconstruction_pass@1_plus": reconstructed["pass@1_plus"],
            "published_pass@1_plus": published["pass@1_plus"],
            "difference": reconstructed["pass@1_plus"] - published["pass@1_plus"],
            "purpose": "paired task outcomes only; not a new A0-prime claim",
        }

    for seed, (a4_tag, a1_tag) in enumerate(zip(A4_TAGS, A1_TAGS)):
        curve = summarize_curve(load_curve(results / f"opd_kl_s{seed}.jsonl"))
        kl_changes.append(curve["last_minus_first_10_mean_sampled_kl"])
        arm = {
            "seed": seed,
            "init": a1_tag,
            "retained_adapter_sha256": manifest["retained_adapters"][a4_tag],
            "kl_curve": curve,
            "benchmarks": {},
        }
        for dataset in DATASETS:
            a4_summary = validate_summary(
                results / f"a0_{a4_tag}_{dataset}.json",
                dataset,
                tag=a4_tag,
                arm="A4",
                require_passk=(seed == 2),
            )
            a1_summary = validate_summary(
                results / f"a0_{a1_tag}_{dataset}.json",
                dataset,
                tag=a1_tag,
                arm="A1",
                require_passk=(seed == 2),
            )
            matrix = validate_matrix(
                results / f"win_matrix_{a4_tag}_{dataset}.json", dataset, a4_tag
            )
            if not math.isclose(
                matrix["student_pass_rate"], a4_summary["pass@1_plus"], abs_tol=1e-12
            ):
                raise ValueError(f"{a4_tag}/{dataset}: matrix/student summary mismatch")
            if not math.isclose(
                matrix["teacher_pass_rate"],
                teacher[dataset]["pass@1_plus"],
                abs_tol=1e-12,
            ):
                raise ValueError(f"{a4_tag}/{dataset}: matrix/teacher summary mismatch")

            a4_pass1 = float(a4_summary["pass@1_plus"])
            a1_pass1 = float(a1_summary["pass@1_plus"])
            a4_values[dataset].append(a4_pass1)
            a1_values[dataset].append(a1_pass1)
            arm["benchmarks"][dataset] = {
                "pass@1_plus": a4_pass1,
                "seed_matched_A1_pass@1_plus": a1_pass1,
                "A4_minus_A1": a4_pass1 - a1_pass1,
                "pass@k_plus": a4_summary.get("pass@k_plus", {}),
                "teacher_student_matrix": matrix["matrix"],
                "teacher_only": matrix["counts"]["teacher_only"],
                "student_only": matrix["counts"]["student_only"],
            }
        report["A4"][a4_tag] = arm

    report["A4"]["aggregate"] = {
        dataset: {
            "A4_pass@1_plus": _mean_sd(a4_values[dataset]),
            "A1_pass@1_plus": _mean_sd(a1_values[dataset]),
            "mean_A4_minus_A1": (
                statistics.mean(a4_values[dataset])
                - statistics.mean(a1_values[dataset])
            ),
        }
        for dataset in DATASETS
    }
    report["A4"]["aggregate"]["mean_last_minus_first_10_sampled_kl"] = (
        statistics.mean(kl_changes)
    )

    for dataset in DATASETS:
        a4 = validate_summary(
            results / f"a0_a4-opd-s2_{dataset}.json",
            dataset,
            tag="a4-opd-s2",
            arm="A4",
            require_passk=True,
        )["pass@k_plus"]
        a1 = validate_summary(
            results / f"a0_a1-s2_{dataset}.json",
            dataset,
            tag="a1-s2",
            arm="A1",
            require_passk=True,
        )["pass@k_plus"]
        a3 = validate_summary(
            results / f"a0_a3-partial-s2_{dataset}.json",
            dataset,
            tag="a3-partial-s2",
            arm="A3",
            require_passk=True,
        )["pass@k_plus"]
        report["seed2_passk_comparison"][dataset] = {
            "A1": a1,
            "A3": a3,
            "A4": a4,
            "A4_minus_A1": {key: a4[key] - a1[key] for key in a4},
            "A4_minus_A3": {key: a4[key] - a3[key] for key in a4},
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/a4_report.json")
    args = parser.parse_args()

    report = build_report(Path(args.results_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
