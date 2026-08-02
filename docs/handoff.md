# Project handoff & interview prep

A single reference for the project so far: what it is, the results, the concepts to
know cold, the engineering problems solved, and likely interview questions.

---

## 1. One-liner

An empirical study of the **reward-density spectrum** in small-model code
post-training on **Qwen2.5-Coder-1.5B**, evaluated on HumanEval+ / MBPP+. Three
training signals on one axis (reward granularity): **L0 GRPO+binary → L1
GRPO+partial-credit → L2 on-policy distillation (token-level)**. The project
measures *signal availability* (a custom metric, EPR), the *Goodhart cost* of dense
reward, and the *diversity* cost — the same questions frontier labs face at scale.

Target roles: LLM post-training / applied research / evals; and (with an engineering
framing) new-grad MLE.

## 2. Pipeline / arms

`A0` student baseline · `A0'` 7B-teacher baseline · `A1` SFT (shared init) ·
`A2` GRPO-binary · `A3` GRPO-partial · `A3'` +test-subsampling · `A4` OPD · `A5` OPD→GRPO.

Status: **A0, A0', A1 (seed 0) done and in git.** A2–A5 not started (the core RL
comparison is still ahead).

## 3. Results so far (all in `results/`, `docs/results.md`)

pass@1 (plus, greedy):

| | HumanEval+ | MBPP+ |
|---|:---:|:---:|
| A0 (1.5B base) | 65.2% | 59.0% |
| A1 (SFT) | 72.6% | 60.8% |
| A0' (7B teacher) | 87.2% | 72.0% |

Logic-error share (base tests): A0 25% → A1 18% → A0' 7% (HumanEval+); syntax ~0%
for all. **EPR@init: base 58.4% → A1 32.9%** (mean group reward 0.32 → 0.58).

### Key findings (interview gold)
1. **Syntax errors are already ~0%** even for the 1.5B → the textbook "SFT fixes
   syntax" story doesn't apply; the real headroom is **logic errors**.
2. **A1 (SFT) raises competence but LOWERS EPR@init** (58%→33%). A stronger
   cold-start saturates easy prompts into all-pass (zero-variance) groups and
   sharpens the distribution (entropy 0.25→0.07), *shrinking* the gradient-producing
   set for RL. Competence ↑ vs gradient-availability/diversity ↓ — the through-line.
3. **HumanEval+ gain (+7.4) ≫ MBPP+ gain (+1.8)** — plausibly because the
   contamination audit removed MBPP-train near-dups of MBPP+, so the MBPP+ number is
   uninflated (evidence the audit works); HE+ gains are pure generalization.

## 4. Concepts to know cold (things asked this session)

- **GRPO & gradient starvation.** GRPO advantage is group-normalized:
  `A_i=(r_i−mean)/std`. If a prompt's G rollouts all share a reward (all-pass or
  all-fail) → std=0 → zero advantage → **zero gradient**. Binary reward triggers this
  at *both* ends (hard=all-fail, easy=all-pass), so the effective training set shrinks
  from both ends. DAPO (Yu 2025) *filters* zero-variance groups; partial credit
  *densifies* the reward so groups have variance.
- **EPR (Effective Prompt Ratio)** = fraction of prompts whose rollout group has
  reward variance (produces gradient). `EPR@init` = measured for a starting policy;
  our signature metric — measures "how much learnable signal RL will actually have."
- **Reward-density spectrum.** L0 binary (whole-trajectory {0,1}) → L1 partial
  (test-level fraction) → L2 OPD (token-level teacher log-ratio). OPD = dense
  KL-constrained RL where the teacher's per-token log-ratio is an implicit reward.
- **Off-policy vs on-policy distillation.** Off-policy (=SFT here): imitate teacher
  outputs on a fixed dataset — the model never sees its own mistakes (exposure bias).
  On-policy (=OPD): the student generates, the teacher scores the student's *own*
  rollouts → learns to recover from its own errors (DAgger, Ross 2011). We use the
  *same* teacher for both so `A4−A1` isolates the value of on-policyness.
- **Rejection-sampling fine-tuning / execution-verified distillation.** Sample k
  solutions, keep only those passing all tests, SFT on them. Lineage: STaR (Zelikman
  2022), RFT (Yuan 2023), ReST/ReST^EM, Llama-2 rejection sampling, Code Llama
  unit-test filtering. Execution verification is the antidote to "false-promise"
  imitation (Gudibande 2023).
- **LoRA vs full-FT.** LoRA = small adapter, cheap, consistent across arms; but
  "LoRA learns less and forgets less" (Biderman 2024) — stated as a limitation.
- **Contamination audit.** n-gram overlap (n=13, GPT-3 style) between train pool and
  eval statements; function-name match alone over-fires on MBPP (generic names).
- **Completion-only loss.** Mask the prompt; compute loss only on assistant tokens
  (TRL `assistant_only_loss=True`) — else capacity is wasted modeling the problem.
- **greedy vs sampling / pass@1 vs pass@k.** Greedy = model's single most-confident
  answer (pass@1). Sampling (temp>0, n samples) → pass@k = "at least one of k is
  right" (potential/diversity). RL often raises pass@1 but may not raise pass@large-k
  ("sharpening not extending", Yue 2025).
- **Frontier anchor: Kimi K3 (arXiv:2607.24653).** Its post-training is exactly
  SFT(cold-start) → RL → Multi-Teacher On-Policy Distillation; the MOPD per-token
  reward is the *clipped teacher/student log-ratio* — the L2 signal at the dense end
  of our spectrum. DeepSeek-R1: distillation beats small-model RL; GRPO from
  DeepSeekMath.

## 5. Engineering problems solved (great "tell me about a bug" material)

- **evalplus 0.2.1 vs 0.3.1**: bare `evalplus.codegen` missing in 0.2.1 → pinned
  `evalplus==0.3.1`.
- **macOS `setrlimit` / `libnvrtc.so.13`**: evalplus executor + vLLM shutdown crash
  on the box's torch 2.11+cu130 (CUDA-13 lib absent). Our own sandbox guards rlimit
  in try/except; the vLLM shutdown traceback is cosmetic (samples already written).
- **wandb → pathtools → `imp` on Py3.12** broke `pip install` → split requirements,
  pinned `wandb>=0.18`, disable wandb unless `WANDB_API_KEY` set.
- **HF Xet 401** (hf-mirror doesn't proxy the Xet CAS) → `HF_HUB_DISABLE_XET=1`.
- **System disk full on the 7B** → `HF_HOME=/root/autodl-tmp/hf` (data disk); then
  switched to **ModelScope** for a faster CN download.
- **MBPP dataset id**: `load_dataset("mbpp")` rejected → `google-research-datasets/mbpp`
  config `full`.
- **sentence-transformers → torchcodec** import crash → replaced the embedding
  contamination filter with pure-Python **n-gram overlap**.
- **Contamination over-removal**: exact function-name match nuked ~half the pool
  (MBPP reuses generic names); short n-grams collide on templated phrasing → switch to
  **13-gram, remove on n-gram only**, name-match kept as a diagnostic.
- **vLLM can't load a LoRA adapter dir** → merge LoRA into base and save a full model.
  Merging the live TRL model left a `base_model.` prefix → merge onto a *fresh* base.
  Stray adapter safetensors beside the merged model broke vLLM → **isolate the trainer
  output dir** so `args.out` holds only the merged model.
- **TRL 1.x API**: `DataCollatorForCompletionOnlyLM` removed → use
  `SFTConfig(assistant_only_loss=True)`.
- **Shell stdout-capture bug**: `$(gen_eval …)` captured vLLM logs as the "path"
  (File name too long) → redirect subprocess output to stderr, only the path to stdout.
- **Unattended runs**: `nohup … ; /usr/bin/shutdown` + an `( sleep N; shutdown )`
  watchdog so long jobs auto-stop billing; AutoDL billing = while "运行中",
  disconnect ≠ stop; only 关机 stops (数据盘 kept), 释放 wipes data.

## 6. Likely interview questions (prepare answers)

- Why is code a clean RL domain? (verifiable, low-noise binary reward from tests)
- Explain GRPO's zero-variance/zero-gradient failure and how binary reward triggers
  it at both ends. How does partial credit help? How does DAPO differ (filter vs
  densify)?
- What is EPR and why did you invent it? Why did SFT *lower* it — is that bad?
- Off-policy (SFT) vs on-policy (OPD) distillation — what does `A4−A1` isolate?
  Why the same teacher?
- How did you build SFT data? Why execution-verified? How did you avoid the
  "false promise of imitation" critique?
- How did you prevent train/test contamination? Why n-gram not function-name?
- Partial credit is a proxy reward — how would you measure/ mitigate Goodhart
  (hacking_gap on held-out tests; IRT-weighted tests; subsampling)?
- Does RL extend capability or just sharpen it? How would pass@k show that?
- Why LoRA? What does it confound? (Biderman 2024)
- Scope: what can/can't a 1.5B function-level study tell you? (state limits proudly)
- Relate your setup to Kimi K3 / DeepSeek-R1.

## 7. Resume bullet (fill final numbers)

> Built and open-sourced an end-to-end code post-training + evaluation system
> (contamination-safe data → execution-verified SFT → planned GRPO/OPD) on
> Qwen2.5-Coder-1.5B; introduced **Effective Prompt Ratio** to measure gradient
> starvation under GRPO; SFT improved HumanEval+ pass@1 65%→73% while EPR@init fell
> 58%→33%, quantifying a competence-vs-learnable-signal trade-off. Ran distributed
> vLLM inference on cloud GPUs with reproducible configs and a hardened execution
> sandbox.

## 8. AutoDL ops cheatsheet

- Update code on the box: `git fetch origin && git reset --hard origin/main`
  (leaves untracked models/results alone).
- Env for GPU runs: `export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=8`
- Teacher cached at `/root/autodl-tmp/Qwen2.5-Coder-7B-Instruct` (use as a local path).
- Unattended: `nohup bash -c '( sleep 28800; /usr/bin/shutdown ) & …; /usr/bin/shutdown' > log 2>&1 &`
- Results live on the data disk; **download the small `results/*.json` and commit** —
  raw generations (`evalplus_results/`) and checkpoints stay off git.
- Stop billing: 控制台 → **关机** (not just disconnect; not 释放).

## 9. Next steps

1. A1 rigor: pass@k (confirm diversity collapse), seeds 1–2, diversity/learnability
   ablations.
2. **A2/A3 GRPO (the core)**: binary vs partial-credit reward; log EPR curves over
   training (RQ1); this is where the reward-density thesis is actually tested.
3. A3' Goodhart (visible vs held-out tests), A4 OPD (teacher log-ratio), stats
   (paired bootstrap, MDE), blog.
