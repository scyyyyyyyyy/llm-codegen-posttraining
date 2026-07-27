"""GRPO training -- arms A2 (binary), A3 (partial), A3' (partial + subsampling).

Config (v2 §5.3):
  TRL GRPOTrainer + vLLM colocate
  G=8 rollouts/prompt (dynamically up to 16 on hard prompts if budget allows)
  temperature 1.0, max_new_tokens 512
  lr 1e-6 ~ 3e-6 (LoRA); KL coeff 0, raise to 1e-3 only if output degrades
  reward = sandboxed execution: eval.rewards.binary_reward / partial_reward

Must log (v2 §5.3): EPR (stratified by difficulty), mean reward, response length,
50-problem quick eval every 200 steps. Flag length bias if a solution length
grows > 50% (Singhal et al. 2023).

RQ3 mitigation (A3'): compute reward on a VISIBLE test subset with per-step random
subsampling; evaluate on held-out tests to measure hacking_gap.
"""

from __future__ import annotations

import argparse


def build_reward_fn(reward_type: str, subsample: bool):
    """Return a callable(code, tests)->float wired to eval.rewards + EPRLogger.

    reward_type: "binary" | "partial" | "irt"
    subsample:   if True, score on a random visible-test subset each step (A3').
    """
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/grpo_partial.yaml")
    p.add_argument("--init", default="checkpoints/sft", help="A1 SFT checkpoint")
    p.add_argument("--reward", choices=["binary", "partial", "irt"], required=True)
    p.add_argument("--subsample", action="store_true", help="A3' Goodhart mitigation")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="checkpoints/grpo")
    args = p.parse_args()
    # TODO: TRL GRPOTrainer + vLLM colocate; attach EPRLogger; checkpoint / 200 steps.
    raise NotImplementedError


if __name__ == "__main__":
    main()
