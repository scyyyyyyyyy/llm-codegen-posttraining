"""SFT with trl's SFTTrainer + LoRA on Qwen2.5-Coder-1.5B-Instruct."""

from __future__ import annotations

import argparse

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sft.jsonl")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default="checkpoints/sft")
    args = parser.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=args.data, split="train")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        num_train_epochs=3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_seq_length=2048,
        packing=True,
        logging_steps=10,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
