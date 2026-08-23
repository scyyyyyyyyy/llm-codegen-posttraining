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
                 subsample: int | None = None, resume_step: int | None = None):
        self.reward_type = reward_type
        self.g = num_generations
        self.log_path = log_path
        self.subsample = subsample
        self.step = 0
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if resume_step is not None:
            if not os.path.exists(log_path):
                raise FileNotFoundError(
                    f"cannot resume at step {resume_step}: missing EPR log {log_path}")
            with open(log_path) as f:
                lines = [line for line in f if line.strip()]
            if len(lines) < resume_step:
                raise ValueError(
                    f"cannot resume at step {resume_step}: EPR log has only {len(lines)} records")
            with open(log_path, "w") as f:
                f.writelines(lines[:resume_step])
            self.step = resume_step
        else:
            with open(log_path, "w"):
                pass

    def __call__(self, prompts, completions, tests=None, entry_point=None, **kw):
        import random

        codes = [extract_code(_completion_text(c)) for c in completions]
        if tests is None or len(tests) != len(codes):
            raise ValueError("TRL must provide one test list per completion")
        rewards = []
        for code, ts in zip(codes, tests):
            use = ts
            if self.subsample and len(ts) > self.subsample:
                use = random.sample(ts, self.subsample)
            # pool tests are plain asserts that call the fn directly -> entry_point
            # must be None (no check() wrapper), matching build_sft_data / epr_init.
            if self.reward_type == "binary":
                rewards.append(binary_reward(code, use))
            else:
                rewards.append(partial_reward(code, use))

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


def validate_generation_batch(
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    expected_effective_batch_size: int,
) -> int:
    """Guard against silently changing the rollout/update budget."""
    values = {
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_generations": num_generations,
        "expected_effective_batch_size": expected_effective_batch_size,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"GRPO batch parameters must be positive: {values}")
    effective = per_device_train_batch_size * gradient_accumulation_steps
    if effective != expected_effective_batch_size:
        raise ValueError(
            "GRPO rollout budget changed: "
            f"microbatch({per_device_train_batch_size}) * "
            f"accumulation({gradient_accumulation_steps}) = {effective}, "
            f"expected {expected_effective_batch_size}"
        )
    if effective % num_generations:
        raise ValueError(
            f"effective batch {effective} is not divisible by "
            f"num_generations {num_generations}"
        )
    return effective


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=["binary", "partial"], required=True)
    p.add_argument("--init", default="checkpoints/sft-s0", help="A1 SFT checkpoint")
    p.add_argument("--pool", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--out", default="checkpoints/grpo")
    p.add_argument("--subsample", type=int, default=None, help="A3': visible-test subset size")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-prompts", type=int, default=None,
                   help="limit the pool for a smoke test; unset for a full run")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="override epoch length; use 1 for a smoke test")
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--expected-effective-batch-size", type=int, default=32,
                   help="guardrail for an unchanged rollout/update budget")
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="trade compute for activation memory without changing rollout budget",
    )
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--epr-log", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--skip-merge", action="store_true",
                   help="save only the adapter (useful for a smoke test)")
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--beta", type=float, default=0.0, help="KL coeff; raise if output degrades")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    effective_batch_size = validate_generation_batch(
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.num_generations,
        args.expected_effective_batch_size,
    )
    if not 0.0 < args.vllm_gpu_memory_utilization < 1.0:
        p.error("--vllm-gpu-memory-utilization must be strictly between 0 and 1")
    print(json.dumps({
        "grpo_runtime": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "num_generations": args.num_generations,
            "prompt_groups_per_update": effective_batch_size // args.num_generations,
        }
    }, sort_keys=True))

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = build_dataset(args.pool)
    if args.max_prompts is not None:
        ds = ds.select(range(min(args.max_prompts, len(ds))))
    # seed 0 keeps the original (already-committed) filename; seeds >0 get a
    # suffix so multi-seed runs never clobber each other's EPR curves.
    seed_sfx = "" if args.seed == 0 else f"_s{args.seed}"
    tag = args.reward + ("-sub" if args.subsample else "") + seed_sfx
    epr_log = args.epr_log or f"results/epr_curve_{tag}.jsonl"
    resume_step = None
    if args.resume_from_checkpoint:
        state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        with open(state_path) as f:
            resume_step = int(json.load(f)["global_step"])
    reward = RewardFn(args.reward, args.num_generations,
                      epr_log, args.subsample, resume_step=resume_step)

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
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        use_vllm=True,
        vllm_mode="colocate",                    # share the training GPU
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=2048,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
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
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # merge onto a fresh base -> full model for eval / next arm
    import gc

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = args.out + "-train/adapter"
    trainer.save_model(adapter_dir)
    if args.skip_merge:
        print(f"training adapter saved -> {adapter_dir}; merge skipped")
        return
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(args.init, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.init).save_pretrained(args.out)
    print(f"saved merged model -> {args.out}  (EPR curve: {epr_log})")


if __name__ == "__main__":
    main()
