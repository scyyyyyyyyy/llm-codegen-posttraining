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

**Preliminary pass@k** (HumanEval+, temperature 0.8, n=64; evalplus estimator):
base pass@1 66.4% / pass@10 90.6%; plus pass@1 60.3% / pass@10 85.2%.
(Full pass@k table for k=1,4,16,64 across both benchmarks pending a batched rerun.)

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

## A2+ — RL / OPD arms

_Pending._
