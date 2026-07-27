"""On-policy distillation -- arm A4, the dense limit L2 (v2 §5.4).

Teacher: Qwen2.5-Coder-7B-Instruct (same family => shared tokenizer, avoids the
known teacher-student thinking-pattern mismatch failure mode).

Loop:
  student samples rollouts -> teacher forward pass for per-token logprob ->
  reverse-KL loss on completion tokens only (mode-seeking; MiniLLM, GKD).
  lr ~1e-5 (LoRA); token budget matched to GRPO's generation volume.
  teacher: forward only, 7B half-precision, single-card batched scoring.

Must log: teacher-student per-token KL descent curve (RQ4 collapse analysis).
Reference recipe: Lu & Thinking Machines Lab (2025), or TRL distillation support.

A5 (optional): A4 -> A3, i.e. OPD then GRPO-partial, mimicking the industrial
dense->sparse pipeline.
"""

from __future__ import annotations

import argparse

TEACHER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def reverse_kl_loss(student_logprobs, teacher_logprobs, completion_mask):
    """Reverse-KL over completion tokens only. TODO."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/opd.yaml")
    p.add_argument("--init", default="checkpoints/sft", help="A1 SFT checkpoint")
    p.add_argument("--teacher", default=TEACHER_MODEL)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="checkpoints/opd")
    args = p.parse_args()
    # TODO: student rollout (vLLM) -> teacher logprobs -> reverse-KL update;
    #       log per-token KL curve; checkpoint / 200 steps.
    raise NotImplementedError


if __name__ == "__main__":
    main()
