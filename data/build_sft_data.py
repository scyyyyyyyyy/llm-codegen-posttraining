"""Build distilled SFT data via teacher rejection sampling (A1 §3).

For each clean prompt: sample k solutions from the 7B teacher, keep those passing
all tests (verifiable filter), normalize format, keep <=N diverse solutions, and
tag learnability from a base-student greedy pass. Emits three variants for the
ablations: sft_base (1 sol), sft_div (<=2 diverse), sft_learn (frontier-only).

GPU (vLLM). Usage:
  python -m data.build_sft_data --pool data/prompt_pool.clean.jsonl \
      --teacher /root/autodl-tmp/Qwen2.5-Coder-7B-Instruct \
      --student Qwen/Qwen2.5-Coder-1.5B-Instruct --out-prefix data/sft
"""

from __future__ import annotations

import argparse

from eval.sandbox import run_batch

from .common import (
    extract_code,
    normalize_code,
    read_jsonl,
    to_chat_record,
    write_jsonl,
)

K = 4
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 512
N_DIVERSE = 2


def _user_prompt(item: dict) -> str:
    return f"Problem:\n{item['prompt_text']}\n\nWrite the function `{item['entry_point']}`."


def _chat_prompts(tokenizer, items: list[dict]) -> list[str]:
    sys = "You are an expert Python programmer. Write clean, correct code."
    out = []
    for it in items:
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": _user_prompt(it)}]
        out.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    return out


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))


def select_diverse(solutions: list[str], n: int = N_DIVERSE) -> list[str]:
    """Greedily keep up to n solutions that are mutually dissimilar."""
    kept: list[str] = []
    for s in solutions:
        ns = normalize_code(s)
        if all(_jaccard(ns, normalize_code(k)) < 0.8 for k in kept):
            kept.append(s)
        if len(kept) >= n:
            break
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="data/prompt_pool.clean.jsonl")
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--out-prefix", default="data/sft")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--eval-workers", type=int, default=8)
    args = p.parse_args()

    pool = read_jsonl(args.pool)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # --- teacher: k samples/prompt ---
    tk_t = AutoTokenizer.from_pretrained(args.teacher)
    teacher = LLM(model=args.teacher, max_model_len=2048)
    t_out = teacher.generate(
        _chat_prompts(tk_t, pool),
        SamplingParams(n=args.k, temperature=TEMPERATURE, top_p=TOP_P,
                       max_tokens=MAX_TOKENS))
    del teacher

    # --- student: greedy for learnability ---
    tk_s = AutoTokenizer.from_pretrained(args.student)
    student = LLM(model=args.student, max_model_len=2048)
    s_out = student.generate(
        _chat_prompts(tk_s, pool),
        SamplingParams(n=1, temperature=0.0, max_tokens=MAX_TOKENS))
    del student

    # --- verify (execution filter) ---
    base_rows, div_rows, learn_rows = [], [], []
    for item, t_o, s_o in zip(pool, t_out, s_out):
        cands = [extract_code(o.text) for o in t_o.outputs]
        # one batched execution per (candidate, test)
        jobs, spans = [], []
        for c in cands:
            start = len(jobs)
            for t in item["tests"]:
                jobs.append((c, t, None))
            spans.append((start, len(jobs)))
        res = run_batch(jobs, workers=args.eval_workers) if jobs else []
        verified = [cands[i] for i, (a, b) in enumerate(spans)
                    if b > a and all(r.passed for r in res[a:b])]
        if not verified:
            continue

        student_code = extract_code(s_o.outputs[0].text)
        sjobs = [(student_code, t, None) for t in item["tests"]]
        student_solved = all(r.passed for r in run_batch(sjobs, workers=args.eval_workers))

        ep = item["entry_point"]
        base_rows.append(to_chat_record(item["prompt_text"], ep, verified[0]))
        for sol in select_diverse(verified):
            div_rows.append(to_chat_record(item["prompt_text"], ep, sol))
        if not student_solved:
            learn_rows.append(to_chat_record(item["prompt_text"], ep, verified[0]))

    write_jsonl(f"{args.out_prefix}_base.jsonl", base_rows)
    write_jsonl(f"{args.out_prefix}_div.jsonl", div_rows)
    write_jsonl(f"{args.out_prefix}_learn.jsonl", learn_rows)
    print(f"verified problems: {len(base_rows)}  "
          f"(div examples {len(div_rows)}, learnable-frontier {len(learn_rows)})")


if __name__ == "__main__":
    main()
