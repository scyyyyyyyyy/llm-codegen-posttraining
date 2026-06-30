"""DPO with trl's DPOTrainer on top of the SFT checkpoint.

Watch rewards/chosen (up), rewards/rejected (down), rewards/margins (growing).
Code DPO collapses easily (concentrated response distribution -> KL blows up):
if margins stop growing, raise beta to 0.2-0.3 or switch to IPO. DPO overfits
fast — usually stop within 1 epoch.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dpo_pairs.jsonl")
    parser.add_argument("--model", required=True, help="SFT (or prior-round) checkpoint")
    parser.add_argument("--output-dir", default="checkpoints/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    dataset = load_dataset("json", data_files=args.data, split="train")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=5e-7,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=1,  # DPO overfits fast
        loss_type="sigmoid",
        max_length=2048,
        max_prompt_length=512,
        logging_steps=10,
        report_to="wandb",
    )

    trainer = DPOTrainer(
        model=args.model,
        args=dpo_config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
