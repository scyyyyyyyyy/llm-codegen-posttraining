"""GRPO training -- arms A2 (binary), A3 (partial), A3' (+ test subsampling).

Starts from the A1 SFT checkpoint. The reward is the sandboxed execution reward
(binary or partial credit). The reward function doubles as the EPR logger: for
each generation group it records whether the group has reward variance (produces
gradient) -> results/epr_curve_<reward>.jsonl, the RQ1 signal.

GPU (TRL GRPOTrainer + vLLM). Usage:
  python train/grpo.py --reward binary  --init checkpoints/sft-s0 --out checkpoints/a2-binary
  python train/grpo.py --reward partial --init checkpoints/sft-s0 --out checkpoints/a3-partial
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow `python train/grpo.py` (script mode) to import the repo's packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.common import extract_code, read_jsonl
from eval.epr import group_has_gradient
from eval.rewards import binary_reward, partial_reward

SYSTEM = "You are an expert Python programmer. Write clean, correct code."


def _completion_text(c) -> str:
    """TRL completions are either a str or a list of chat messages."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


class RewardFn:
    """Callable reward for GRPOTrainer that also logs EPR per generation group."""

    def __init__(self, reward_type: str, num_generations: int, log_path: str,
                 subsample: int | None = None):
        self.reward_type = reward_type
        self.g = num_generations
        self.log_path = log_path
        self.subsample = subsample
        self.step = 0
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        open(log_path, "w").close()

    def __call__(self, prompts, completions, tests=None, entry_point=None, **kw):
        import random

        codes = [extract_code(_completion_text(c)) for c in completions]
        rewards = []
        for code, ts, ep in zip(codes, tests, entry_point):
            use = ts
            if self.subsample and len(ts) > self.subsample:
                use = random.sample(ts, self.subsample)
            if self.reward_type == "binary":
                rewards.append(binary_reward(code, use, ep))
            else:
                rewards.append(partial_reward(code, use, ep))

        # EPR: contiguous groups of G share a prompt in TRL GRPO ordering.
        groups = [rewards[i:i + self.g] for i in range(0, len(rewards), self.g)]
        epr = sum(group_has_gradient(gr) for gr in groups) / max(1, len(groups))
        mean_r = sum(rewards) / max(1, len(rewards))
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"step": self.step, "epr": epr, "mean_reward": mean_r}) + "\n")
        self.step += 1
        return rewards


def build_dataset(pool_path: str):
    from datasets import Dataset

    rows = read_jsonl(pool_path)
    return Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"Problem:\n{r['prompt_text']}\n\nWrite the function `{r['entry_point']}`."},
            ],
            "tests": r["tests"],
            "entry_point": r["entry_point"],
        }
        for r in rows
    ])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=["binary", "partial"], required=True)
    p.add_argument("--init", default="checkpoints/sft-s0", help="A1 SFT checkpoint")
    p.add_argument("--pool", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--out", default="checkpoints/grpo")
    p.add_argument("--subsample", type=int, default=None, help="A3': visible-test subset size")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--beta", type=float, default=0.0, help="KL coeff; raise if output degrades")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = build_dataset(args.pool)
    tag = args.reward + ("-sub" if args.subsample else "")
    reward = RewardFn(args.reward, args.num_generations,
                      f"results/epr_curve_{tag}.jsonl", args.subsample)

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = GRPOConfig(
        output_dir=args.out + "-train",
        num_generations=args.num_generations,
        temperature=1.0,
        max_prompt_length=512,
        max_completion_length=512,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.num_generations,  # >=1 prompt * G
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        bf16=torch.cuda.is_available(),
        use_vllm=True,
        logging_steps=10,
        save_steps=100,
        seed=args.seed,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
    )

    trainer = GRPOTrainer(
        model=args.init,
        reward_funcs=reward,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
    )
    trainer.train()

    # merge onto a fresh base -> full model for eval / next arm
    import gc

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = args.out + "-train/adapter"
    trainer.save_model(adapter_dir)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(args.init, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.init).save_pretrained(args.out)
    print(f"saved merged model -> {args.out}  (EPR curve: results/epr_curve_{tag}.jsonl)")


if __name__ == "__main__":
    main()
