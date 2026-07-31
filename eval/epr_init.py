"""EPR@init — the headline A1 metric (A1 §0).

For a given policy (base student, or an A1 checkpoint) and the training prompt
pool, sample G rollouts per prompt, compute the binary execution reward of each,
and report the fraction of prompt-groups with non-zero reward variance — i.e. the
fraction of prompts that would produce gradient under GRPO. A good SFT cold-start
should raise this (rescue all-fail prompts into the learnable zone).

GPU (vLLM). Usage:
  python -m eval.epr_init --model checkpoints/sft-s0 --tag a1 \
      --pool data/prompt_pool.clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

from data.common import extract_code, read_jsonl

from .epr import group_has_gradient
from .sandbox import run_batch

G = 8
TEMPERATURE = 1.0
MAX_TOKENS = 512


def _user_prompt(item: dict) -> str:
    return f"Problem:\n{item['prompt_text']}\n\nWrite the function `{item['entry_point']}`."


def compute_epr(model_path: str, pool: list[dict], g: int = G,
                workers: int = 8) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_path)
    sys = "You are an expert Python programmer. Write clean, correct code."
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": sys},
             {"role": "user", "content": _user_prompt(it)}],
            tokenize=False, add_generation_prompt=True)
        for it in pool
    ]
    llm = LLM(model=model_path, max_model_len=2048)
    outs = llm.generate(prompts, SamplingParams(
        n=g, temperature=TEMPERATURE, max_tokens=MAX_TOKENS))

    with_grad = 0
    per_prompt = []
    for item, o in zip(pool, outs):
        rewards = []
        for cand in o.outputs:
            code = extract_code(cand.text)
            jobs = [(code, t, None) for t in item["tests"]]
            passed_all = all(r.passed for r in run_batch(jobs, workers=workers)) if jobs else False
            rewards.append(1.0 if passed_all else 0.0)
        has = group_has_gradient(rewards)
        with_grad += int(has)
        per_prompt.append({"id": item["id"], "mean_reward": sum(rewards) / len(rewards),
                           "has_gradient": has})
    return {
        "n_prompts": len(pool),
        "epr_init": with_grad / max(1, len(pool)),
        "mean_group_reward": sum(x["mean_reward"] for x in per_prompt) / max(1, len(pool)),
        "per_prompt": per_prompt,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="e.g. base or a1")
    p.add_argument("--pool", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--g", type=int, default=G)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    res = compute_epr(args.model, read_jsonl(args.pool), args.g)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"epr_init_{args.tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"EPR@init ({args.tag}): {100*res['epr_init']:.1f}%  "
          f"mean group reward {res['mean_group_reward']:.3f}  -> {out}")


if __name__ == "__main__":
    main()
