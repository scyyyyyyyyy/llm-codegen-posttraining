# The Reward Density Spectrum

A compute-matched empirical study of **reward signal density** in small-model code
post-training, on **Qwen2.5-Coder-1.5B-Instruct**. Three training signals sit on one
axis — reward granularity:

| Level | Method | Reward granularity | Signal |
|-------|--------|--------------------|--------|
| L0 | GRPO + binary reward | whole trajectory {0,1} | all tests pass → 1 |
| L1 | GRPO + partial credit | test-level, r = passed/total | continuous [0,1] |
| L2 | On-policy distillation (OPD) | token-level | teacher per-token log-ratio |

## Research questions

- **RQ1 — Gradient starvation.** Under GRPO's group-relative advantage, a group
  with equal rewards yields zero gradient. Binary reward triggers this at *both*
  ends (all-fail hard prompts, all-pass easy prompts). Measured by **EPR (Effective
  Prompt Ratio)** — the signature metric.
- **RQ2 — Error taxonomy dynamics.** How syntax/runtime/logic/timeout errors evolve
  across training steps for each signal.
- **RQ3 — Goodhart tax.** Partial credit is a proxy target; `hacking_gap =
  pass(visible tests) − pass(held-out tests)`. IRT-weighted test scoring as a
  principled mitigation.
- **RQ4 — Diversity ceiling.** pass@k (k=1,4,16,64) and a teacher–student win matrix:
  does dense signal sharpen or genuinely extend ability?

## Experiment matrix

`A0` student baseline · `A0'` teacher (7B) baseline · `A1` SFT (shared start) ·
`A2` GRPO-binary · `A3` GRPO-partial · `A3'` +test subsampling · `A4` OPD ·
`A5` (optional) OPD→GRPO. Every arm: 3 seeds, same prompt set, same SFT start,
budget aligned by generated tokens.

## Layout

```
data/   build_prompt_pool.py · contamination_audit.py · build_sft_data.py
train/  sft.py · grpo.py · opd.py
eval/   sandbox.py · error_classify.py · rewards.py · epr.py · run_eval.py
analysis/ stats.py + reproduction notebook
configs/  sft · grpo_binary · grpo_partial · grpo_partial_subsample · opd
```

## Pipeline

```bash
pip install -r requirements.txt

# 1. contamination-safe data (eval sets NEVER enter training)
python data/build_prompt_pool.py --out data/prompt_pool.jsonl
python data/contamination_audit.py --pool data/prompt_pool.jsonl
python data/build_sft_data.py --pool data/prompt_pool.clean.jsonl

# 2. shared SFT start (A1), then the arms
python train/sft.py  --config configs/sft.yaml --seed 0
python train/grpo.py --config configs/grpo_binary.yaml  --reward binary  --seed 0
python train/grpo.py --config configs/grpo_partial.yaml --reward partial --seed 0
python train/opd.py  --config configs/opd.yaml --seed 0

# 3. eval + statistics
python eval/run_eval.py --model checkpoints/grpo --dataset humaneval_plus --mode passk
```

## What makes this not a toy

Measurable mechanism (EPR), a two-sided conclusion (starvation vs Goodhart
trade-off), statistical rigor (3 seeds, problem-level paired bootstrap, stratified
CIs, MDE/power), and an explicit contamination audit — all on a $0–300 budget.

**Scope:** smallest configuration (1.5B + LoRA + function-level benchmarks);
conclusions are not claimed to extrapolate to large or agentic-coding scale.
