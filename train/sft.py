"""SFT: the shared starting point A1 for all RL/OPD arms (v2 §5.2).

Config (v2 §5.2):
  LoRA r=16, alpha=32; targets = attn {q,k,v,o} + MLP {gate,up,down}
  lr 2e-4, cosine, 3 epochs, effective batch 16, max_len 2048, packing on
  framework: TRL SFTTrainer; ~1-2 H100-hours

Train 3 seeds (statistics plan, v2 §6).
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sft.yaml")
    p.add_argument("--data", default="data/sft.jsonl")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="checkpoints/sft")
    args = p.parse_args()
    # TODO: load base model + LoRA (peft), TRL SFTTrainer, wandb logging,
    #       every-200-step quick eval on 50 problems, save per-seed checkpoint.
    raise NotImplementedError


if __name__ == "__main__":
    main()
