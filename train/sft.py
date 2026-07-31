"""A1 SFT: LoRA fine-tune the 1.5B student on distilled, verified data (A1 §4).

Completion-only loss (assistant tokens only) via TRL's chat-aware SFTTrainer,
LoRA, 3 seeds, per-200-step checkpoints. Shared init for all RL/OPD arms.

Usage:
  python train/sft.py --data data/sft_base.jsonl --seed 0 --out checkpoints/sft-s0
"""

from __future__ import annotations

import argparse
import os

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sft_base.jsonl")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--push-to-hub", default=None, help="HF repo id, e.g. user/a1-sft")
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Conversational dataset: each row is {"messages": [...]}. SFTTrainer applies
    # the chat template itself; assistant_only_loss masks the prompt tokens.
    ds = load_dataset("json", data_files=args.data, split="train")

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,     # effective batch 16
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=2048,
        packing=False,
        assistant_only_loss=True,          # completion-only loss (mask the prompt)
        bf16=torch.cuda.is_available(),
        logging_steps=20,
        save_steps=200,                    # best checkpoint is often mid-training
        save_total_limit=8,
        seed=args.seed,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
    )

    trainer = SFTTrainer(
        model=args.base_model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.out)
    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
