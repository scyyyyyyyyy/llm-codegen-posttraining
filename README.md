# The Reward Density Spectrum: Sparse vs. Dense Signals in Small-Model Code Post-training

A compute-matched empirical study of how **reward signal density** shapes
reinforcement learning and distillation for a small code model
(**Qwen2.5-Coder-1.5B-Instruct**) on **HumanEval+** and **MBPP+**.

> **Status.** Infrastructure and the zero-shot baselines (A0, A0') are complete and
> reproducible. Training arms (SFT → GRPO → OPD) and the statistical analysis are in
> progress; see [Roadmap](#roadmap).

---

## 1. Motivation

Verifiable-reward RL (RLVR) with group-relative advantages (GRPO) is the dominant
post-training recipe for code, but the reward it optimizes is almost always
**binary** — a rollout scores 1 only if *every* unit test passes. Under GRPO's
group-normalized advantage `A_i = (r_i − mean(r)) / std(r)`, a group whose rollouts
all share the same reward contributes **zero gradient**. Binary reward triggers this
at *both* ends: hard prompts where every rollout fails (all-zero) and easy prompts
where every rollout passes (all-one). The effective training set silently shrinks
from both ends toward the middle.

This project treats reward granularity as a controllable axis and measures its
consequences — signal availability, reward hacking, and diversity — rather than
chasing a single headline pass@1.

### The reward density spectrum

| Level | Method | Reward granularity | Signal |
|-------|--------|--------------------|--------|
| **L0** | GRPO + binary reward | whole trajectory `{0,1}` | all tests pass → 1 |
| **L1** | GRPO + partial credit | test-level, `r = passed/total` | continuous `[0,1]` |
| **L2** | On-policy distillation (OPD) | token-level | teacher per-token log-ratio |

L2 is the dense limit: OPD can be viewed as a special case of dense,
KL-constrained RL where the teacher's per-token log-ratio is an implicit reward.

## 2. Research questions

- **RQ1 — Gradient starvation.** Does pass@1 improvement track the fraction of
  prompts that actually produce gradient? We measure the **Effective Prompt Ratio
  (EPR)** — the share of prompt-groups with non-zero reward variance — over training,
  stratified by difficulty.
- **RQ2 — Error-type dynamics.** Which failure modes (syntax / runtime / logic /
  timeout) does each signal repair, tracked *per checkpoint* rather than only at the end?
- **RQ3 — The Goodhart tax of dense reward.** Partial credit is a proxy objective.
  We compute reward on a *visible* test subset and evaluate on *held-out* tests:
  `hacking_gap = pass(visible) − pass(held-out)`, and test IRT-weighted scoring as a
  principled mitigation.
- **RQ4 — Diversity ceiling.** Does dense signal sharpen or genuinely extend ability?
  Measured via pass@k (k = 1, 4, 16, 64) and a teacher–student win matrix.

## 3. Experimental design

**Models.** Student `Qwen2.5-Coder-1.5B-Instruct`; teacher `Qwen2.5-Coder-7B-Instruct`.

**Arms.** `A0` student baseline · `A0'` teacher baseline · `A1` SFT (shared start) ·
`A2` GRPO-binary · `A3` GRPO-partial · `A3'` + visible-test subsampling · `A4` OPD ·
`A5` OPD→GRPO. Every arm uses the same prompt set and SFT start, with budget aligned
by generated tokens (GPU-hours also reported). 3 seeds per arm.

**Benchmarks (evaluation only).** HumanEval+ (164) and MBPP+ (378), EvalPlus suite.
These never enter any training pool.

**Data (contamination-safe).** Training prompts drawn from MBPP's train split and
function-style TACO/APPS subsets, each with executable tests. A near-duplicate audit
(signature match + statement-embedding similarity) removes anything close to the
evaluation sets, and reports the number removed.

**Statistics.** Problem-level paired bootstrap (95% CI), McNemar, per-difficulty
stratified CIs, and a power/MDE analysis, following a "treat evals as statistical
estimation" stance.

## 4. Results

### 4.1 Zero-shot baselines (A0 student / A0' teacher, greedy pass@1)

pass@1 by difficulty is on the `+` (harder) test set. Difficulty is a documented
proxy (terciles of canonical-solution LOC; EvalPlus ships no official labels).

| Model | Benchmark | pass@1 base | pass@1 plus | easy | medium | hard |
|-------|-----------|:---:|:---:|:---:|:---:|:---:|
| **A0** 1.5B | HumanEval+ | 71.3% | 65.2% | 70.6% | 66.7% | 54.8% |
| **A0** 1.5B | MBPP+      | 69.6% | 59.0% | 70.4% | 62.1% | 39.8% |
| **A1** SFT 1.5B | HumanEval+ | 78.0% | 72.6% | 77.9% | 75.9% | 59.5% |
| **A1** SFT 1.5B | MBPP+      | 72.5% | 60.8% | 73.0% | 62.1% | 41.5% |
| **A0'** 7B | HumanEval+ | 91.5% | 87.2% | 92.6% | 88.9% | 76.2% |
| **A0'** 7B | MBPP+      | 82.8% | 72.0% | 84.1% | 69.7% | 54.5% |

The **student→teacher gap** (+22 pts on HumanEval+, +13 on MBPP+) is the ceiling
post-training targets and the OPD reference.

**Error breakdown** (greedy, base tests):

| Model | Benchmark | correct | syntax | runtime | logic | timeout |
|-------|-----------|:---:|:---:|:---:|:---:|:---:|
| A0 1.5B | HumanEval+ | 69.5% | 0.0% | 5.5% | 25.0% | 0.0% |
| A0 1.5B | MBPP+      | 69.0% | 0.0% | 5.0% | 25.9% | 0.0% |
| A1 SFT 1.5B | HumanEval+ | 76.8% | 0.0% | 5.5% | 17.7% | 0.0% |
| A1 SFT 1.5B | MBPP+      | 72.0% | 0.0% | 5.8% | 22.2% | 0.0% |
| A0' 7B | HumanEval+ | 90.2% | 0.0% | 2.4% | 7.3% | 0.0% |
| A0' 7B | MBPP+      | 82.3% | 0.0% | 2.9% | 14.8% | 0.0% |

**pass@k (plus, temp 0.8, n=64; k=1/4/16/64).** A0 1.5B: HumanEval+
61.8/80.4/88.8/**92.1**%, MBPP+ 56.0/71.4/79.3/**84.7**%. A0' 7B: HumanEval+
82.3/91.0/93.7/94.5%, MBPP+ 67.2/80.5/86.3/89.4%. The full A0→A1→A2→A3 pass@k table
is in [docs/results.md](docs/results.md): post-training lifts pass@1 but leaves
pass@64 flat — it **sharpens rather than extends** (RQ4).

### 4.2 Findings so far

1. **A steep difficulty gradient, especially on MBPP+** (hard = 39.8% vs. easy
   70.4%). The hard tier is exactly the regime where binary reward degenerates to an
   all-zero signal — the motivating case for partial credit (RQ1).
2. **Syntax errors are already ~0%** on both benchmarks. A strong instruct base makes
   essentially no parse errors, so the common "SFT eliminates syntax errors" narrative
   does not apply here — a finding worth stating plainly.
3. **Logic errors are the dominant and stable failure mode (~25%)** across both
   benchmarks. This — not syntax — is the headroom that RL and distillation must
   address (RQ2).

### 4.3 A1 SFT cold-start (seed 0)

SFT (LoRA, teacher-distilled, 255 verified examples) raises pass@1(plus) to
**72.6% HumanEval+ / 60.8% MBPP+** (from A0's 65.2 / 59.0), cutting logic errors
(HE+ 25→18%) — H2 supported. But **EPR@init drops 58%→33%**: a stronger cold-start
saturates easy prompts and sharpens the output distribution, *shrinking* the
gradient-producing set for RL (H1 refuted — the more interesting result). The
competence-vs-gradient-availability tension carries into A2–A4. Details in
[docs/results.md](docs/results.md).

### 4.4 A2 / A3 GRPO — reward density: mechanism yes, accuracy no

The RQ1 test, in two parts:

- **Training dynamics (EPR), replicated over 3 seeds.** Partial-credit reward raises
  mean EPR from **0.325 (binary) to 0.480 (partial)** — densifying the reward
  moves substantially more prompt-groups into the gradient-producing zone. The
  mechanism is real and reproducible.
- **Held-out accuracy — null.** One epoch of LoRA GRPO with *either* reward does
  **not** move held-out pass@1 off the SFT start (A1 72.6 / A2 71.3 / A3 72.6 on
  HumanEval+; ~60% on MBPP+, all deltas ≤1.5%). Higher EPR did not convert to higher
  accuracy in this light-touch regime. Seed 2 repeats the same pattern: A2/A3 are
  73.8/71.3% on HumanEval+ and 59.5/60.1% on MBPP+. There is no consistent
  cross-benchmark endpoint advantage (conservative MDE ~15% / ~10%).

The honest headline is a **two-sided result**: reward density measurably changes the
*learnable signal* (EPR) but not the *endpoint* here — the story lives in the
mechanism, not a pass@1 win. pass@k (§4.1) adds the RQ4 half: post-training sharpens
rather than extends. This matches the pre-registered risk plan and frames why the
dense limit (A4 OPD) — the only arm expected to raise the pass@k ceiling — matters.

## 5. Repository layout

```
data/   build_prompt_pool.py · contamination_audit.py · build_sft_data.py
train/  sft.py · grpo.py · opd.py
eval/   sandbox.py · error_classify.py · rewards.py · epr.py
        run_eval.py · run_a0.py · verify_pipeline.py · difficulty.py
analysis/ stats.py + reproduction notebook
configs/  sft · grpo_binary · grpo_partial · grpo_partial_subsample · opd
scripts/  run_a0.sh
docs/     autodl.md   (GPU setup)
```

Core eval modules run on CPU; generation/scoring run on GPU (see `docs/autodl.md`).

## 6. Reproduction

```bash
pip install -r requirements.txt            # core: vllm + evalplus==0.3.1
pip install -r requirements-train.txt      # + training stack (A1+)
pip install -r requirements-analysis.txt   # + local stats/plots

# CPU gate: canonical solutions must pass 100%, classifier must be sane
python -m eval.verify_pipeline --n 164

# A0 / A0' baselines (GPU; see docs/autodl.md)
MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct TAG=qwen1.5b bash scripts/run_a0.sh
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct   TAG=qwen7b   bash scripts/run_a0.sh
# -> results/a0_<tag>_<dataset>.json  (pass@1, pass@k, error breakdown, difficulty)

# A2/A3 seed-2 replication from the shared A1 seed-2 checkpoint
python train/grpo.py --reward binary --init checkpoints/a1-s2 \
  --pool data/prompt_pool.clean.jsonl --out checkpoints/a2-binary-s2 --seed 2 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 8 \
  --expected-effective-batch-size 32 --gradient-checkpointing \
  --vllm-gpu-memory-utilization 0.25 --save-steps 20 \
  --epr-log results/epr_curve_a2-binary-s2.jsonl

python train/grpo.py --reward partial --init checkpoints/a1-s2 \
  --pool data/prompt_pool.clean.jsonl --out checkpoints/a3-partial-s2 --seed 2 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 8 \
  --expected-effective-batch-size 32 --gradient-checkpointing \
  --vllm-gpu-memory-utilization 0.25 --save-steps 20 \
  --epr-log results/epr_curve_a3-partial-s2.jsonl

ARM=A2 MODEL=checkpoints/a2-binary-s2 TAG=a2-binary-s2 bash scripts/run_a0.sh
ARM=A3 MODEL=checkpoints/a3-partial-s2 TAG=a3-partial-s2 bash scripts/run_a0.sh
```

## 7. Scope and limitations

This is the smallest meaningful configuration: a 1.5B model, LoRA, and function-level
benchmarks. Conclusions are about *mechanism* at small scale and are not claimed to
extrapolate to large models or agentic coding. The value is in measurable signal
availability (EPR), a two-sided reward-density trade-off (starvation vs. Goodhart),
statistical rigor, and an explicit contamination audit — on a \$0–300 budget.

## <a name="roadmap"></a>8. Roadmap

- [x] Eval infrastructure: sandbox, error taxonomy, reward functions, pass@k
- [x] Contamination-safe eval verification (164/164 ground-truth, classifier checks)
- [x] A0 student baseline (HumanEval+ / MBPP+)
- [x] A0' teacher baseline + pass@k
- [ ] pass@k for A0 (student)
- [x] pass@k for A0 (student) — HumanEval+ 92.1% / MBPP+ 84.7% @k=64
- [x] A1 SFT (distilled from teacher) — seed 0: pass@1 + EPR@init
- [x] A2/A3 GRPO binary vs partial — **EPR 0.325 vs 0.480 over 3 seeds** (RQ1 mechanism)
- [x] A2/A3 held-out eval + full pass@k (A0→A1→A2→A3) — **RQ1: higher EPR did *not*
  lift held-out pass@1; RQ4: post-training sharpens, not extends (pass@64 flat)**
- [x] A2/A3 seed 2 + exact paired-bootstrap/McNemar comparison
- [ ] A3' visible-test subsampling (RQ3 Goodhart)
- [ ] A4 OPD + teacher–student win matrix (RQ4) — the dense limit
- [ ] Statistical analysis writeup + blog

## References

GRPO (Shao et al., 2024) · DAPO dynamic sampling (Yu et al., 2025) · On-policy
distillation / GKD (Agarwal et al., 2024), MiniLLM (Gu et al., 2024) · EvalPlus
(Liu et al., 2023) · Qwen2.5-Coder (Hui et al., 2024) · reward gaming (Skalse et al.,
2022) · error bars for evals (Miller, 2024).
