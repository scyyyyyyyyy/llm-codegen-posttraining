"""Aggregate the matched A3 versus A3-prime confirmatory experiment.

The compared arms share prompt IDs, seed-matched A1 initializations, 78 trainer
steps, and the same GRPO batch geometry.  The only treatment difference is
whether reward sees one fixed visible assertion (A3-prime) or every assertion
(A3-matched).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .stats import holm_bonferroni, paired_bootstrap_ci


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_summary(
    matched_results: Path,
    a3prime_results: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    expected_steps: int = 78,
    expected_prompts: int = 310,
) -> dict:
    """Build the serialized confirmatory summary from checked-in artifacts."""

    rows = []
    endpoint_pvalues: dict[str, float] = {}
    for seed in seeds:
        matched_tag = f"a3matched-s{seed}"
        prime_tag = f"a3prime-s{seed}"
        curve = _read_jsonl(matched_results / f"epr_curve_{matched_tag}.jsonl")
        if len(curve) != expected_steps:
            raise ValueError(
                f"{matched_tag}: expected {expected_steps} EPR records, got {len(curve)}"
            )

        matched_gap = _read_json(
            matched_results / f"a3prime_gap_{matched_tag}.json"
        )
        prime_gap = _read_json(a3prime_results / f"a3prime_gap_{prime_tag}.json")
        matched_by_id = {row["id"]: row for row in matched_gap["per_prompt"]}
        prime_by_id = {row["id"]: row for row in prime_gap["per_prompt"]}
        if matched_by_id.keys() != prime_by_id.keys():
            raise ValueError(f"seed {seed}: A3-matched and A3-prime prompt IDs differ")
        if len(matched_by_id) != expected_prompts:
            raise ValueError(
                f"seed {seed}: expected {expected_prompts} prompt pairs, "
                f"got {len(matched_by_id)}"
            )

        prompt_ids = sorted(matched_by_id)
        prime_values = [
            prime_by_id[prompt_id]["visible_test_pass_rate"]
            - prime_by_id[prompt_id]["heldout_test_pass_rate"]
            for prompt_id in prompt_ids
        ]
        matched_values = [
            matched_by_id[prompt_id]["visible_test_pass_rate"]
            - matched_by_id[prompt_id]["heldout_test_pass_rate"]
            for prompt_id in prompt_ids
        ]
        gap_effect = paired_bootstrap_ci(prime_values, matched_values, seed=seed)

        endpoints = {}
        for dataset in ("humaneval", "mbpp"):
            comparison = _read_json(
                matched_results
                / f"paired_a3matched_vs_a3prime_s{seed}_{dataset}.json"
            )
            endpoint_pvalues[f"s{seed}_{dataset}"] = comparison["delta_ci"][
                "p_value"
            ]
            endpoints[dataset] = comparison

        rows.append(
            {
                "seed": seed,
                "epr_mean": statistics.fmean(item["epr"] for item in curve),
                "matched_visible_minus_heldout": matched_gap["hacking_gap"],
                "a3prime_visible_minus_heldout": prime_gap["hacking_gap"],
                "a3prime_minus_matched_gap_effect": gap_effect,
                "endpoint_comparisons_matched_minus_a3prime": endpoints,
            }
        )

    return {
        "design": (
            "A3-prime versus full-reward A3 matched on prompt IDs, 78 steps, "
            "seed-matched A1 starts"
        ),
        "seeds": rows,
        "holm_bonferroni_endpoint_rejections": holm_bonferroni(endpoint_pvalues),
        "aggregate": {
            "matched_epr_mean": statistics.fmean(row["epr_mean"] for row in rows),
            "matched_gap_mean": statistics.fmean(
                row["matched_visible_minus_heldout"] for row in rows
            ),
            "a3prime_gap_mean": statistics.fmean(
                row["a3prime_visible_minus_heldout"] for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-results", type=Path, default=Path("results"))
    parser.add_argument("--a3prime-results", type=Path, default=Path("results"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/a3matched_confirmatory_summary.json"),
    )
    args = parser.parse_args()

    summary = build_summary(args.matched_results, args.a3prime_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
