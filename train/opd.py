"""On-policy distillation (OPD) for arm A4.

The student samples its own completions. On exactly those tokens, the frozen
teacher supplies dense per-token feedback through the sampled reverse-KL
estimate ``log p_student - log p_teacher``. The update is the same
importance-sampling policy-gradient loss used by an RL trainer, with the
negative sampled KL as the immediate (discount-zero) token advantage.

The default budget is deliberately comparable to the GRPO arms: 392 prompts x
8 student rollouts = 3,136 completions and 98 optimizer steps at accumulation
32. Only completion tokens contribute to the loss.

The implementation is single-GPU and restartable. It keeps the 7B teacher in
forward-only half precision, trains a rank-32 LoRA on the 1.5B student, and asks
Qwen to materialize logits only at completion prediction positions. This is
important for fitting both models on a 24 GB card.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

# Allow ``python train/opd.py`` to import repository packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.common import read_jsonl

STUDENT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TEACHER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SYSTEM = "You are an expert Python programmer. Write clean, correct code."
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _require_same_shape(*tensors) -> None:
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"all token tensors must have the same shape, got {shapes}")


def _masked_mean(values, mask):
    """Mean over non-zero mask entries, preserving autograd."""
    _require_same_shape(values, mask)
    weight = mask.to(dtype=values.dtype)
    count = weight.sum()
    if int(count.detach().item()) == 0:
        raise ValueError("completion_mask selects zero tokens")
    return (values * weight).sum() / count


def sampled_reverse_kl(student_logprobs, teacher_logprobs, completion_mask):
    """Monte-Carlo per-token estimate of KL(student || teacher).

    The tokens must have been sampled from ``student_logprobs``' policy. A
    finite sample mean may be negative even though the exact KL is non-negative.
    """
    _require_same_shape(student_logprobs, teacher_logprobs, completion_mask)
    return _masked_mean(
        student_logprobs.detach() - teacher_logprobs.detach(), completion_mask
    )


def reverse_kl_loss(
    student_logprobs,
    teacher_logprobs,
    completion_mask,
    sampled_student_logprobs=None,
):
    """Policy-gradient loss for sampled per-token reverse KL.

    ``sampled_student_logprobs`` is the behavior-policy log probability saved
    with a rollout. In this strictly on-policy implementation it defaults to a
    detached copy of ``student_logprobs``. The value of the importance ratio is
    therefore one, but its gradient is not: it moves over-weighted sampled
    tokens down and under-weighted sampled tokens up relative to the teacher.

    We intentionally use discount factor zero, matching the reference OPD
    recipe: every token optimizes only its immediate teacher feedback.
    """
    import torch

    if sampled_student_logprobs is None:
        sampled_student_logprobs = student_logprobs.detach()
    _require_same_shape(
        student_logprobs,
        sampled_student_logprobs,
        teacher_logprobs,
        completion_mask,
    )
    old = sampled_student_logprobs.detach()
    advantage = -(old - teacher_logprobs.detach())
    ratio = torch.exp(student_logprobs - old)
    return _masked_mean(-(ratio * advantage), completion_mask)


def selective_log_softmax(logits, token_ids):
    """Log probability of selected tokens without retaining full log-softmax."""
    _require_same_shape(logits[..., 0], token_ids)
    selected = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).float()
    normalizer = torch_logsumexp_float(logits)
    return selected - normalizer


def torch_logsumexp_float(logits):
    """A small seam kept separate so the numerical path is unit-testable."""
    return logits.float().logsumexp(dim=-1)


def completion_prediction_positions(
    prompt_length: int, sequence_length: int, *, device=None
):
    """Hidden-state positions whose logits predict the completion tokens."""
    import torch

    if prompt_length < 1:
        raise ValueError("prompt must contain at least one token")
    if sequence_length <= prompt_length:
        raise ValueError("sequence contains no completion token")
    # Logit at position i predicts input token i+1.
    return torch.arange(prompt_length - 1, sequence_length - 1, device=device)


def rollout_schedule(n_prompts: int, rollouts_per_prompt: int, seed: int) -> list[int]:
    """Deterministic shuffled prompt indices for an OPD run."""
    if n_prompts <= 0 or rollouts_per_prompt <= 0:
        raise ValueError("n_prompts and rollouts_per_prompt must be positive")
    schedule: list[int] = []
    for repeat in range(rollouts_per_prompt):
        indices = list(range(n_prompts))
        random.Random((seed + 1) * 1_000_003 + repeat).shuffle(indices)
        schedule.extend(indices)
    return schedule


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{row['prompt_text']}\n\n"
                f"Write the function `{row['entry_point']}`."
            ),
        },
    ]


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_curve(path: Path, resume_step: int | None) -> None:
    """Create a fresh curve or truncate it to a durable checkpoint boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume_step is None:
        path.write_text("")
        return
    if not path.exists():
        raise FileNotFoundError(f"cannot resume without KL curve: {path}")
    records = [line for line in path.read_text().splitlines() if line.strip()]
    if len(records) < resume_step:
        raise ValueError(
            f"checkpoint is at update {resume_step}, but {path} has only "
            f"{len(records)} records"
        )
    path.write_text("\n".join(records[:resume_step]) + ("\n" if resume_step else ""))


def latest_checkpoint(train_dir: str | os.PathLike[str]) -> Path | None:
    root = Path(train_dir)
    candidates = []
    for path in root.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (
            path.is_dir()
            and (path / "state.json").is_file()
            and (path / "training.pt").is_file()
            and (path / "adapter" / "adapter_model.safetensors").is_file()
        ):
            candidates.append((step, path))
    return max(candidates, default=(None, None))[1]


def _optimizer_to(optimizer, device) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_torch(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location="cpu")


def save_checkpoint(
    *,
    student,
    optimizer,
    scheduler,
    train_dir: Path,
    state: dict[str, Any],
    save_total_limit: int,
) -> Path:
    """Atomically publish an adapter + optimizer checkpoint."""
    import torch

    step = int(state["optimizer_step"])
    destination = train_dir / f"checkpoint-{step}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    train_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=train_dir))
    try:
        student.save_pretrained(tmp / "adapter")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            },
            tmp / "training.pt",
        )
        _json_dump_atomic(tmp / "state.json", state)
        os.replace(tmp, destination)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    checkpoints = []
    for path in train_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except (IndexError, ValueError):
            pass
    for _, old in sorted(checkpoints)[:-save_total_limit]:
        shutil.rmtree(old)
    return destination


def _dtype_from_arg(name: str):
    import torch

    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16")
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _tokenizer_fingerprint(tokenizer) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(tokenizer.get_vocab().items()):
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _render_prompt(tokenizer, row: dict[str, Any], max_prompt_tokens: int):
    rendered = tokenizer.apply_chat_template(
        prompt_messages(row), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] > max_prompt_tokens:
        raise ValueError(
            f"prompt {row.get('id', '<unknown>')} has "
            f"{encoded['input_ids'].shape[1]} tokens, exceeding the strict "
            f"limit {max_prompt_tokens}; refusing to silently truncate it"
        )
    return encoded["input_ids"]


def _trainable_parameters(model) -> Iterable:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=STUDENT_MODEL, help="matching A1 checkpoint")
    parser.add_argument("--teacher", default=TEACHER_MODEL)
    parser.add_argument("--pool", default="data/prompt_pool.clean.jsonl")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", default="checkpoints/a4-opd")
    parser.add_argument("--kl-log", default=None)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-merge", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_sequence_length <= args.max_new_tokens:
        raise ValueError("--max-sequence-length must exceed --max-new-tokens")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.save_steps <= 0 or args.save_total_limit <= 0:
        raise ValueError("checkpoint settings must be positive")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

    if not torch.cuda.is_available():
        raise RuntimeError("OPD requires a CUDA GPU")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = _dtype_from_arg(args.dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.pool)
    if args.max_prompts is not None:
        rows = rows[: min(len(rows), args.max_prompts)]
    schedule = rollout_schedule(len(rows), args.rollouts_per_prompt, args.seed)
    if args.max_rollouts is not None:
        schedule = schedule[: min(len(schedule), args.max_rollouts)]
    if args.max_optimizer_steps is not None:
        cap = args.max_optimizer_steps * args.gradient_accumulation_steps
        schedule = schedule[: min(len(schedule), cap)]
    if not schedule:
        raise ValueError("rollout budget is empty")

    total_updates = math.ceil(len(schedule) / args.gradient_accumulation_steps)
    warmup_steps = round(total_updates * args.warmup_ratio)
    train_dir = Path(args.out + "-train")
    curve_path = Path(args.kl_log or f"results/opd_kl_s{args.seed}.jsonl")
    resume_path = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None

    signature = {
        "format": 1,
        "init": os.path.abspath(args.init),
        "teacher": os.path.abspath(args.teacher),
        "pool": os.path.abspath(args.pool),
        "pool_sha256": file_sha256(args.pool),
        "seed": args.seed,
        "n_prompts": len(rows),
        "rollouts_per_prompt": args.rollouts_per_prompt,
        "rollout_budget": len(schedule),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_new_tokens": args.max_new_tokens,
        "max_sequence_length": args.max_sequence_length,
        "temperature": args.temperature,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
    }

    resume_state: dict[str, Any] | None = None
    resume_payload = None
    if resume_path:
        resume_state = json.loads((resume_path / "state.json").read_text())
        if resume_state.get("signature") != signature:
            raise ValueError("resume checkpoint signature does not match this run")
        if int(resume_state["rollouts_completed"]) % args.gradient_accumulation_steps:
            raise ValueError("checkpoint is not at an optimizer-step boundary")
        resume_payload = _load_torch(resume_path / "training.pt")
        prepare_curve(curve_path, int(resume_state["optimizer_step"]))
    else:
        prepare_curve(curve_path, None)

    tokenizer = AutoTokenizer.from_pretrained(args.init)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    student_tokenizer_hash = _tokenizer_fingerprint(tokenizer)
    teacher_tokenizer_hash = _tokenizer_fingerprint(teacher_tokenizer)
    if student_tokenizer_hash != teacher_tokenizer_hash:
        raise ValueError("teacher and student tokenizer vocabularies/token IDs differ")
    del teacher_tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading frozen teacher {args.teacher} ({dtype})")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.config.use_cache = False

    print(f"loading student {args.init} ({dtype})")
    student_base = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)
    # Qwen family checkpoints can pad the embedding/output matrix to different
    # hardware-friendly row counts even when their tokenizer token->ID mapping
    # is identical. That padding is harmless; every real tokenizer ID merely
    # needs to be representable by both models.
    required_vocab_rows = max(tokenizer.get_vocab().values()) + 1
    teacher_vocab_rows = teacher.get_input_embeddings().num_embeddings
    student_vocab_rows = student_base.get_input_embeddings().num_embeddings
    if required_vocab_rows > min(teacher_vocab_rows, student_vocab_rows):
        raise ValueError(
            "tokenizer IDs exceed a model embedding matrix: "
            f"required={required_vocab_rows}, teacher={teacher_vocab_rows}, "
            f"student={student_vocab_rows}"
        )
    print(
        "validated shared tokenizer with padded model vocabularies: "
        f"tokens={required_vocab_rows}, teacher_rows={teacher_vocab_rows}, "
        f"student_rows={student_vocab_rows}"
    )
    student_base.config.use_cache = False
    if resume_path:
        student = PeftModel.from_pretrained(
            student_base, resume_path / "adapter", is_trainable=True
        )
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        student = get_peft_model(student_base, lora)
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    student.train()
    student.print_trainable_parameters()

    trainable = list(_trainable_parameters(student))
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    rollouts_completed = 0
    optimizer_step = 0
    cumulative_tokens = 0
    elapsed_before = 0.0
    if resume_state is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        _optimizer_to(optimizer, device)
        scheduler.load_state_dict(resume_payload["scheduler"])
        rollouts_completed = int(resume_state["rollouts_completed"])
        optimizer_step = int(resume_state["optimizer_step"])
        cumulative_tokens = int(resume_state["cumulative_completion_tokens"])
        elapsed_before = float(resume_state.get("elapsed_seconds", 0.0))
        random.setstate(resume_payload["python_rng_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state_all"])
        print(
            f"resumed {resume_path}: rollout={rollouts_completed}, "
            f"update={optimizer_step}"
        )
    del resume_payload

    if rollouts_completed > len(schedule):
        raise ValueError("resume checkpoint is past this rollout budget")

    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    update_metrics: list[dict[str, float]] = []
    last_checkpoint_step = optimizer_step
    max_prompt_tokens = args.max_sequence_length - args.max_new_tokens

    for rollout_position in range(rollouts_completed, len(schedule)):
        row = rows[schedule[rollout_position]]
        prompt_cpu = _render_prompt(tokenizer, row, max_prompt_tokens)
        prompt_ids = prompt_cpu.to(device)
        prompt_length = int(prompt_ids.shape[1])
        attention = torch.ones_like(prompt_ids)

        # Eval mode activates generation KV caching. Qwen has zero base-model
        # dropout and this LoRA explicitly uses dropout=0, so eval/train modes
        # represent the same sampling policy.
        student.eval()
        with torch.no_grad():
            sequence = student.generate(
                input_ids=prompt_ids,
                attention_mask=attention,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        student.train()
        sequence_length = int(sequence.shape[1])
        positions = completion_prediction_positions(
            prompt_length, sequence_length, device=device
        )
        targets = sequence[:, prompt_length:]
        completion_mask = torch.ones_like(targets, dtype=torch.float32)
        full_attention = torch.ones_like(sequence)
        completion_tokens = int(targets.numel())

        # Score with the teacher first and retain only one scalar per sampled
        # token. Its large vocabulary logits are freed before the student
        # backward graph is materialized.
        with torch.no_grad():
            teacher_output = teacher(
                input_ids=sequence,
                attention_mask=full_attention,
                use_cache=False,
                logits_to_keep=positions,
            )
            teacher_logprobs = selective_log_softmax(teacher_output.logits, targets)
        del teacher_output

        student_output = student(
            input_ids=sequence,
            attention_mask=full_attention,
            use_cache=False,
            logits_to_keep=positions,
        )
        student_logprobs = selective_log_softmax(student_output.logits, targets)
        loss = reverse_kl_loss(student_logprobs, teacher_logprobs, completion_mask)
        if not (
            torch.isfinite(teacher_logprobs).all()
            and torch.isfinite(student_logprobs).all()
            and torch.isfinite(loss)
        ):
            raise FloatingPointError(
                f"non-finite OPD value at rollout {rollout_position}"
            )

        update_start = (
            rollout_position // args.gradient_accumulation_steps
        ) * args.gradient_accumulation_steps
        update_end = min(
            update_start + args.gradient_accumulation_steps, len(schedule)
        )
        accumulation_size = update_end - update_start
        (loss / accumulation_size).backward()

        with torch.no_grad():
            token_kl = student_logprobs.detach() - teacher_logprobs
            update_metrics.append(
                {
                    "kl_sum": float(token_kl.sum().item()),
                    "student_nll_sum": float((-student_logprobs.detach()).sum().item()),
                    "teacher_nll_sum": float((-teacher_logprobs).sum().item()),
                    "tokens": float(completion_tokens),
                    "loss": float(loss.detach().item()),
                    "truncated": float(completion_tokens == args.max_new_tokens),
                }
            )
        del student_output, student_logprobs, teacher_logprobs, loss, token_kl

        rollouts_completed = rollout_position + 1
        cumulative_tokens += completion_tokens
        if rollouts_completed != update_end:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite OPD gradient at optimizer step {optimizer_step + 1}"
            )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        tokens_this_update = sum(item["tokens"] for item in update_metrics)
        rollouts_this_update = len(update_metrics)
        elapsed = elapsed_before + time.monotonic() - started
        record = {
            "optimizer_step": optimizer_step,
            "rollouts_completed": rollouts_completed,
            "rollouts_this_update": rollouts_this_update,
            "sampled_reverse_kl_per_token": (
                sum(item["kl_sum"] for item in update_metrics) / tokens_this_update
            ),
            "student_nll_per_token": (
                sum(item["student_nll_sum"] for item in update_metrics)
                / tokens_this_update
            ),
            "teacher_nll_per_token": (
                sum(item["teacher_nll_sum"] for item in update_metrics)
                / tokens_this_update
            ),
            "policy_loss_mean_rollout": (
                sum(item["loss"] for item in update_metrics) / rollouts_this_update
            ),
            "completion_tokens_this_update": int(tokens_this_update),
            "mean_completion_length": tokens_this_update / rollouts_this_update,
            "truncated_completion_fraction": (
                sum(item["truncated"] for item in update_metrics)
                / rollouts_this_update
            ),
            "cumulative_completion_tokens": cumulative_tokens,
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": float(grad_norm.detach().item()),
            "elapsed_seconds": elapsed,
            "max_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        append_jsonl_durable(curve_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        update_metrics.clear()
        torch.cuda.reset_peak_memory_stats(device)

        should_save = (
            optimizer_step % args.save_steps == 0
            or rollouts_completed == len(schedule)
        )
        if should_save:
            state = {
                "signature": signature,
                "optimizer_step": optimizer_step,
                "rollouts_completed": rollouts_completed,
                "cumulative_completion_tokens": cumulative_tokens,
                "elapsed_seconds": elapsed,
                "curve": str(curve_path.resolve()),
                "tokenizer_sha256": student_tokenizer_hash,
            }
            checkpoint = save_checkpoint(
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dir=train_dir,
                state=state,
                save_total_limit=args.save_total_limit,
            )
            last_checkpoint_step = optimizer_step
            print(f"saved {checkpoint}", flush=True)

    if optimizer_step != total_updates:
        raise RuntimeError(f"expected {total_updates} updates, completed {optimizer_step}")
    if last_checkpoint_step != optimizer_step:
        raise RuntimeError("final optimizer state was not checkpointed")

    adapter_dir = train_dir / "adapter"
    student.save_pretrained(adapter_dir)
    final_state = {
        "signature": signature,
        "optimizer_step": optimizer_step,
        "rollouts_completed": rollouts_completed,
        "cumulative_completion_tokens": cumulative_tokens,
        "elapsed_seconds": elapsed_before + time.monotonic() - started,
        "curve": str(curve_path.resolve()),
        "tokenizer_sha256": student_tokenizer_hash,
        "completed": True,
    }
    _json_dump_atomic(train_dir / "state.json", final_state)

    if args.skip_merge:
        print(f"adapter saved -> {adapter_dir}; merge skipped")
        return

    del teacher, optimizer, scheduler, trainable, student, student_base
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        args.init, dtype=dtype, low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out_path = Path(args.out).resolve()
    protected = {Path(args.init).resolve(), Path(args.teacher).resolve(), Path("/")}
    if out_path in protected or len(out_path.parts) < 3:
        raise ValueError(f"refusing unsafe merged-model output path: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    publish_tmp = Path(
        tempfile.mkdtemp(prefix=f".{out_path.name}-publishing-", dir=out_path.parent)
    )
    try:
        merged.save_pretrained(publish_tmp)
        tokenizer.save_pretrained(publish_tmp)
        if not (publish_tmp / "config.json").is_file() or not (
            (publish_tmp / "model.safetensors").is_file()
            or (publish_tmp / "model.safetensors.index.json").is_file()
        ):
            raise RuntimeError("merged model publication is incomplete")
        if out_path.exists():
            shutil.rmtree(out_path)
        os.replace(publish_tmp, out_path)
    except BaseException:
        shutil.rmtree(publish_tmp, ignore_errors=True)
        raise
    print(
        f"saved merged A4 model -> {out_path} "
        f"(adapter: {adapter_dir}; KL curve: {curve_path})"
    )


if __name__ == "__main__":
    main()
