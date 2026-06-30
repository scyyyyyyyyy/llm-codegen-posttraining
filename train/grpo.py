"""Optional: GRPO using execution reward directly, no preference pairs.

Lets you compare RL-from-reward against DPO-from-preferences on efficiency and
final pass@1. The reward is the same partial_reward used elsewhere.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--reward", choices=["binary", "partial"], default="partial")
    args = parser.parse_args()

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    from eval.compute_metrics import binary_reward, partial_reward

    reward_fn = partial_reward if args.reward == "partial" else binary_reward
    dataset = load_dataset("json", data_files=args.problems, split="train")

    def reward_func(completions, tests, **kwargs):
        from data.rejection_sample import extract_code

        return [reward_fn(extract_code(c), t) for c, t in zip(completions, tests)]

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        num_generations=8,
        max_completion_length=512,
        logging_steps=10,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
