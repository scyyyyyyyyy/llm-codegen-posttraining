# Results log

Running record of experimental results. Raw per-run summaries live in
`results/a0_<tag>_<dataset>.json`; this file is the human-readable digest.

## A0 — student baseline (Qwen2.5-Coder-1.5B-Instruct, zero-shot)

Greedy decoding, pass@1. Difficulty = terciles of canonical-solution LOC (proxy;
EvalPlus has no official labels). Difficulty pass@1 is on the `+` test set.

| Benchmark | pass@1 (base) | pass@1 (plus) | easy | medium | hard |
|-----------|:---:|:---:|:---:|:---:|:---:|
| HumanEval+ | 71.3% | 65.2% | 70.6% | 66.7% | 54.8% |
| MBPP+      | 69.6% | 59.0% | 70.4% | 62.1% | 39.8% |

Error breakdown of the greedy sample (base tests):

| Benchmark | correct | syntax | runtime | logic | timeout |
|-----------|:---:|:---:|:---:|:---:|:---:|
| HumanEval+ | 69.5% | 0.0% | 5.5% | 25.0% | 0.0% |
| MBPP+      | 69.0% | 0.0% | 5.0% | 25.9% | 0.0% |

Difficulty counts — HumanEval+: easy 68 / medium 54 / hard 42. MBPP+: 378 tasks.

**pass@k** (temperature 0.8, n=64; evalplus estimator), plus set, k=1/4/16/64:
HumanEval+ 61.8/80.4/88.8/**92.1**%, MBPP+ 56.0/71.4/79.3/**84.7**%. Full
A0→A1→A2→A3 table in the pass@k section below. (The batched rerun reproduced greedy
pass@1 at 65.9 / 59.5% HE+/MBPP+, vs. the first run's 65.2 / 59.0 — a <1% run-to-run
wobble; the headline table above keeps the first-run figures.)

### Findings
1. Steep difficulty gradient (MBPP+ hard 39.8% vs. easy 70.4%) — the sparse-reward
   regime that motivates partial credit (RQ1).
2. Syntax errors ≈ 0% on both benchmarks — the "SFT fixes syntax" story does not
   apply to a strong instruct base.
3. Logic errors are the dominant, stable failure mode (~25%) — the real headroom for
   RL / distillation (RQ2).

## A0' — teacher baseline (Qwen2.5-Coder-7B-Instruct, zero-shot)

Greedy pass@1 + pass@k (temperature 0.8, n=64).

| Benchmark | pass@1 (base) | pass@1 (plus) | easy | medium | hard |
|-----------|:---:|:---:|:---:|:---:|:---:|
| HumanEval+ | 91.5% | 87.2% | 92.6% | 88.9% | 76.2% |
| MBPP+      | 82.8% | 72.0% | 84.1% | 69.7% | 54.5% |

Error breakdown (greedy, base tests):

| Benchmark | correct | syntax | runtime | logic | timeout |
|-----------|:---:|:---:|:---:|:---:|:---:|
| HumanEval+ | 90.2% | 0.0% | 2.4% | 7.3% | 0.0% |
| MBPP+      | 82.3% | 0.0% | 2.9% | 14.8% | 0.0% |

pass@k (plus): HumanEval+ 82.3 / 91.0 / 93.7 / 94.5% (k=1/4/16/64);
MBPP+ 67.2 / 80.5 / 86.3 / 89.4%.

## Student vs. teacher (pass@1 plus, greedy)

| | 1.5B (A0) | 7B (A0') | gap |
|---|:---:|:---:|:---:|
| HumanEval+ | 65.2% | 87.2% | +22.0 |
| MBPP+      | 59.0% | 72.0% | +13.0 |

The gap is the ceiling that post-training targets — and it is almost entirely
**logic errors**: the 7B cuts logic errors from ~25% to 7–15% while syntax stays
at 0% for both. This directly frames the project: the headroom is logic, not syntax.

## A1 — SFT cold-start (seed 0, first result)

Training pool after pre-filter + contamination audit: 455 → 392 clean prompts
(63 removed on 13-gram overlap). Teacher rejection sampling yielded 255 verified
SFT examples. SFT (LoRA r=32, 3 epochs) trained cleanly: loss 0.18 → 0.07,
token accuracy 0.98, entropy 0.25 → 0.07.

**EPR@init (headline metric), on the 392-prompt pool, G=8, temp 1.0:**

| | mean group reward | EPR@init |
|---|:---:|:---:|
| base (pre-SFT) | 0.319 | 58.4% |
| A1 (post-SFT)  | 0.584 | 32.9% |

**Finding (refutes H1, and is more interesting than H1).** SFT raised competence
(mean reward 0.32 → 0.58) but **lowered** EPR@init (58% → 33%). A stronger cold-start
does not monotonically increase the gradient-producing prompt fraction: SFT
saturates easy prompts into all-pass (zero-variance) groups faster than it rescues
hard prompts from all-fail, and it sharpens the output distribution (entropy
0.25 → 0.07), reducing rollout diversity — both push prompts out of the learnable
zone. This is the reward-density spectrum's easy-end zero-gradient + diversity-
collapse mechanism, observed directly. Implication: a better SFT init can *shrink*
the learnable set for the subsequent RL.

**Held-out eval (greedy pass@1, A0 / A1 / A0').**

| pass@1 (plus) | HumanEval+ | MBPP+ |
|---|:---:|:---:|
| A0 (base 1.5B) | 65.2% | 59.0% |
| A1 (SFT) | **72.6%** | **60.8%** |
| A0' (7B teacher) | 87.2% | 72.0% |

| logic error (base) | HumanEval+ | MBPP+ |
|---|:---:|:---:|
| A0 | 25.0% | 25.9% |
| A1 | **17.7%** | **22.2%** |
| A0' | 7.3% | 14.8% |

A1 difficulty (plus): HumanEval+ easy 77.9 / med 75.9 / hard 59.5; MBPP+ easy 73.0
/ med 62.1 / hard 41.5.

**H2 supported:** SFT raises competence and cuts logic errors (HE+ 25→17.7, MBPP+
26→22) while syntax stays 0%. A1 closes ~34% of the student→teacher gap on
HumanEval+ (+7.4) but only ~14% on MBPP+ (+1.8) — the asymmetry is plausibly
because the contamination audit removed MBPP-train items near the MBPP+ eval set,
so the MBPP+ gain is uninflated (a sign the audit worked), while HumanEval+ gains
are pure generalization from the distilled data.

**Net A1 story:** SFT works (competence ↑, logic errors ↓) but at a cost — it
*lowers* EPR@init (easy-prompt saturation + distribution sharpening), shrinking the
learnable set and diversity for the subsequent RL. This tension (competence vs
gradient-availability/diversity) is the through-line into A2/A3 (RL) and A4 (OPD).

Caveats: single seed; pass@k (to confirm diversity collapse), seeds 1–2, and the
diversity/learnability ablations still to run.

## A2 / A3 — GRPO binary vs partial credit (RQ1)

Both arms: GRPO from the A1 SFT checkpoint, 1 epoch over the 392-prompt pool,
G=8, temp 1.0, LoRA. The reward function logs EPR every step
(`results/epr_curve_{binary,partial}[_s1].jsonl` for seeds 0–1 and
`results/epr_curve_{a2-binary-s2,a3-partial-s2}.jsonl` for seed 2).

### RQ1a — training dynamics (EPR), replicated over 3 seeds

Mean EPR over the 98 training steps:

| | seed 0 | seed 1 | seed 2 | mean ± SD |
|---|:---:|:---:|:---:|:---:|
| A2 GRPO-binary  | 0.319 | 0.319 | 0.337 | 0.325 ± 0.010 |
| A3 GRPO-partial | **0.459** | **0.492** | **0.487** | **0.480 ± 0.018** |

**Finding (supported, n=3).** Partial-credit reward raises EPR from ~32% to
~46–49% across training, on all three seeds — densifying the reward moves substantially
more prompts into the gradient-producing (some-pass-some-fail) zone, exactly the
reward-density thesis. Binary reward wastes both ends (all-fail hard prompts,
all-pass easy prompts); partial credit rescues the middle. (Mean reward is not
comparable across arms — partial is continuous [0,1] with compile/runtime bonuses,
so it is higher by construction; EPR is the clean comparison.)

### RQ1b — does higher EPR convert to held-out accuracy? (No, in this regime)

Held-out greedy pass@1 of the GRPO checkpoints (seed 0), vs the A1 start:

| pass@1 (plus) | HumanEval+ | MBPP+ |
|---|:---:|:---:|
| A1 SFT (start) | 72.6% | 60.8% |
| A2 GRPO-binary | 71.3% | 60.6% |
| A3 GRPO-partial | 72.6% | 60.1% |

Seed-2 replication:

| pass@1 (plus) | HumanEval+ | MBPP+ |
|---|:---:|:---:|
| A2 GRPO-binary | 73.8% | 59.5% |
| A3 GRPO-partial | 71.3% | 60.1% |

**Finding (null, and the honest headline).** One epoch of LoRA GRPO — with *either*
reward — does **not** move held-out pass@1 off the SFT start (all deltas ≤1.5%, some
negative). The higher EPR of partial credit does **not** translate into higher
accuracy here. All three arms are statistically indistinguishable on held-out
pass@1: the conservative (independent-sample) MDE at 80% power is ~15% on
HumanEval+ (n=164) and ~10% on MBPP+ (n=378), so sub-1.5% gaps are far below what
these eval sizes resolve. The reproducible effect of reward density is in the
*training dynamics* (EPR), not the endpoint accuracy — consistent with the plan's
pre-registered risk note ("if binary vs partial is not significant, report it with
MDE and move the story to the mechanism").

Why so flat: from an already-strong distilled SFT init, GRPO is a light touch —
`frac_reward_zero_std` ends at ~0.5–0.66 (half the groups produce no gradient),
reward rises within training (0.32→~0.68) but the policy barely moves on held-out.
This also echoes DeepSeek-R1 (distillation > RL at small scale) and Yue 2025 (RL
sharpens rather than extends — see pass@k below).

The exact seed-2 paired comparison is small and inconsistent across benchmarks:
A2−A3 is +2.44 points on HumanEval+ (paired bootstrap 95% CI +0.61 to +4.88;
McNemar p=0.125) and −0.53 on MBPP+ (95% CI −2.12 to +1.06; McNemar p=0.754).
It does not establish a consistent endpoint winner after considering both
benchmarks and multiplicity. The serialized comparison is in
`results/a2_a3_seed2_comparison.json`.

## pass@k — sharpening vs. extending (RQ4)

pass@k on the `+` set (temperature 0.8, n=64; evalplus estimator):

| pass@k (plus) | k=1 | k=4 | k=16 | k=64 |
|---|:---:|:---:|:---:|:---:|
| **A0** base 1.5B | 61.8 / 56.0 | 80.4 / 71.4 | 88.8 / 79.3 | **92.1 / 84.7** |
| **A1** SFT | 70.8 / 59.1 | 82.2 / 69.8 | 88.3 / 76.5 | 91.5 / 80.2 |
| **A2** GRPO-binary | 70.6 / 59.4 | 82.2 / 69.7 | 88.2 / 76.3 | 90.2 / 81.0 |
| **A3** GRPO-partial | 70.5 / 59.3 | 81.9 / 69.7 | 87.6 / 76.6 | 90.2 / 81.0 |
| **A0'** 7B teacher | 82.3 / 67.2 | 91.0 / 80.5 | 93.7 / 86.3 | 94.5 / 89.4 |

(Each cell is HumanEval+ / MBPP+.)

Seed-2 pass@k replication (each cell is again HumanEval+ / MBPP+):

| pass@k (plus) | k=1 | k=4 | k=16 | k=64 |
|---|:---:|:---:|:---:|:---:|
| **A2** binary | 70.1 / 59.7 | 82.1 / 71.1 | 86.9 / 77.7 | 89.0 / 81.7 |
| **A3** partial | 69.9 / 59.5 | 82.4 / 70.9 | 87.4 / 77.8 | 90.2 / 83.1 |

**Finding (RQ4).** Post-training **sharpens but does not extend**. SFT/RL lift the
low-k end (HumanEval+ pass@1 61.8 → ~71) by concentrating probability mass on the
top solution, but the high-k ceiling does **not** rise — pass@64 goes 92.1 (base) →
91.5 (SFT) → 90.2 (GRPO), flat-to-slightly-*down*. The base model already reaches
those solutions given 64 tries; post-training reranks rather than expands the
reachable set. This is a textbook instance of the "RL sharpens, not extends"
argument (Yue et al., 2025), now visible on the student itself. Only the 7B teacher
sits on a genuinely higher pass@k curve — the ceiling OPD (A4) targets.

## A4 — OPD

_Pending._ The dense-signal limit and the only arm expected to lift the pass@k
ceiling rather than just sharpen it.
