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

## A1+ — training arms

_Pending._

## A1+ — training arms

_Pending._
