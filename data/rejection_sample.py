"""Rejection sampling: serve the SFT model with vLLM, sample N candidates per
problem, execute each against tests, and record both reward signals and the
error type. Output feeds preference-pair construction.
"""

from __future__ import annotations

import argparse
import json
import re

N_SAMPLES = 16  # 16 > 8: hard problems need enough draws to yield a correct one
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 512

_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block, else return the raw text."""
    m = _CODE_BLOCK.search(text)
    return m.group(1).strip() if m else text.strip()


def sample_candidates(model_path: str, problems: list[dict]) -> dict[str, list[dict]]:
    """Sample N solutions per problem and score them. Returns {problem_id: [candidate]}."""
    from vllm import LLM, SamplingParams

    from eval.compute_metrics import binary_reward, partial_reward
    from eval.error_classify import classify_error

    llm = LLM(model=model_path)
    sampling = SamplingParams(
        n=N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS
    )

    out: dict[str, list[dict]] = {}
    for problem in problems:
        result = llm.generate(problem["prompt"], sampling)
        candidates = []
        for o in result[0].outputs:
            code = extract_code(o.text)
            candidates.append(
                {
                    "code": code,
                    "partial_reward": partial_reward(code, problem["tests"]),
                    "binary_reward": binary_reward(code, problem["tests"]),
                    "error_type": classify_error(code, problem["tests"][0]),
                }
            )
        out[problem["id"]] = candidates
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to SFT checkpoint")
    parser.add_argument("--problems", required=True, help="problems JSONL")
    parser.add_argument("--out", default="data/candidates.json")
    args = parser.parse_args()

    with open(args.problems) as f:
        problems = [json.loads(line) for line in f]

    candidates = sample_candidates(args.model, problems)
    with open(args.out, "w") as f:
        json.dump(candidates, f)
    print(f"sampled {N_SAMPLES} candidates for {len(problems)} problems -> {args.out}")


if __name__ == "__main__":
    main()
