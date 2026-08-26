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
    p.add_argument("--max-samples", type=int, default=None,
                   help="limit the dataset for a smoke test; unset for a full run")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="override epoch length; use 1 for a smoke test")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--skip-merge", action="store_true",
                   help="save only the adapter (useful for a one-step smoke test)")
    p.add_argument("--push-to-hub", default=None, help="HF repo id, e.g. user/a1-sft")
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Conversational dataset: each row is {"messages": [...]}. SFTTrainer applies
    # the chat template itself; assistant_only_loss masks the prompt tokens.
    ds = load_dataset("json", data_files=args.data, split="train")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Trainer writes adapters/checkpoints here; args.out holds ONLY the merged
    # full model, so vLLM never sees stray adapter safetensors alongside it.
    train_dir = args.out + "-train"

    cfg = SFTConfig(
        output_dir=train_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # Transformers 5.x folds the ratio form into warmup_steps: values in
        # [0, 1) are interpreted as a fraction of total training steps.
        warmup_steps=0.05,
        max_length=2048,
        packing=False,
        assistant_only_loss=True,          # completion-only loss (mask the prompt)
        bf16=torch.cuda.is_available(),
        logging_steps=20,
        save_steps=args.save_steps,        # best checkpoint is often mid-training
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
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the LoRA adapter, then merge it into a FRESH base model. Merging the
    # live trainer model leaves a 'base_model.' prefix on the weights that vLLM
    # rejects; loading the adapter onto a clean base gives standard weight names.
    # Result is a full model dir (+ tokenizer) that vLLM / eval / RL arms load.
    import gc

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = train_dir + "/adapter"
    trainer.save_model(adapter_dir)
    if args.skip_merge:
        print(f"smoke/training adapter saved -> {adapter_dir}; merge skipped")
        return
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.out)
    if args.push_to_hub:
        merged.push_to_hub(args.push_to_hub)
    print(f"saved merged model -> {args.out}  (adapter at {adapter_dir})")


if __name__ == "__main__":
    main()
