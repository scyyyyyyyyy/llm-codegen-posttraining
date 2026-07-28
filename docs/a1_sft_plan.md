# A1 — Supervised Fine-Tuning: experiment plan

Status: planned. This spec is written before running (pre-registration style) so
the analysis is not chosen after seeing results.

## 1. Framing

A1 is **not** a generic SFT run. It is *off-policy distillation* from the 7B
teacher: all training targets are teacher-generated, execution-verified solutions.
This is deliberate — later, OPD (A4) is *on-policy* distillation from the **same**
teacher, so `A4 − A1` isolates the value of on-policyness alone (an exposure-bias
correction; cf. DAgger, Ross et al. 2011). A1 is the shared init for every RL/OPD arm.

**Reframing forced by A0.** The baselines show syntax errors are already ~0% for
both student and teacher, so the textbook "SFT removes syntax errors" story does
not apply here. A1's real job is to **close part of the student→teacher logic-error
gap** by imitating verified-correct solutions.

Anchors from A0/A0' (greedy pass@1, plus):

| | HumanEval+ | MBPP+ | logic-err (HE+/MBPP+) |
|---|:---:|:---:|:---:|
| A0 student 1.5B | 65.2% | 59.0% | 25.0% / 25.9% |
| A0' teacher 7B  | 87.2% | 72.0% | 7.3% / 14.8% |
| gap | +22.0 | +13.0 | — |

## 2. Hypotheses (falsifiable)

- **H1 (primary).** A1 recovers 30–50% of the student→teacher pass@1(plus) gap:
  HumanEval+ → ~72–76%, MBPP+ → ~63–66%. Reported with problem-level paired
  bootstrap 95% CI vs A0.
- **H2.** The gain is concentrated in **logic errors** (25% → 12–18%); syntax stays
  ~0%; runtime roughly flat.
- **H3 (diversity cost).** A1 raises pass@1 more than pass@64, i.e. it sharpens the
  distribution rather than extending capability (sets up RQ4). Falsified if
  pass@64 rises proportionally.
- **H4 (CoT ablation).** CoT targets help hard problems (+3–8 pts) but not easy.

## 3. Data — contamination-safe distilled SFT

Pipeline: `build_prompt_pool → contamination_audit → build_sft_data`.

**Prompt pool (target 800–1500, each with executable tests):**
- MBPP *train* split (374; 3 asserts each).
- TACO / APPS intro–medium **function-style** subset (own tests); drop stdin/stdout
  IO problems.
- Pre-filter: a problem is kept only if its ground-truth solution passes all its
  own tests (guards against bad TACO/APPS tests).

**Contamination audit (mandatory, reported):**
- Exact function-signature match **and** problem-statement embedding cosine ≥ 0.90
  against HumanEval+ / MBPP+.
- Drop any match; report `n_before / n_removed / n_after` in `results/`.

**Teacher rejection sampling → targets:**
- For each prompt, sample **k = 4** from the 7B teacher (temp 0.7, top_p 0.95).
- Keep at most **1** solution per prompt that passes **all** visible tests
  (verifiable filter — zero label noise). Expected ~700–1200 examples.
- Optional CoT variant (for H4): teacher generates reasoning + code under a strict
  format prompt; keep only if the code still passes.

**Formatting:**
- Chat format: system (expert Python) / user (problem + signature) / assistant
  (` ```python … ``` `). Normalize fences to match evalplus extraction exactly.
- Near-duplicate dedup on normalized code to avoid style collapse.
- Record per-source counts and solution-length distribution.

## 4. Training recipe

- Base `Qwen2.5-Coder-1.5B-Instruct`; LoRA r=16, α=32, targets = attn {q,k,v,o} +
  MLP {gate,up,down}; TRL `SFTTrainer`.
- lr 2e-4, cosine, warmup 0.05, **3 epochs**, effective batch 16 (bs 4 × accum 4),
  max_len 2048, packing on, bf16.
- **Completion-only loss** (mask the prompt; `DataCollatorForCompletionOnlyLM`).
  Non-negotiable — otherwise capacity is wasted modelling the problem text.
- **3 seeds** {0,1,2}. Main table reports mean ± std.
- Checkpoint every 200 steps; keep all (best is often mid-training).
- Monitor: train loss, dev-split eval loss, and a 50-problem quick pass@1 every
  200 steps. **Early-stop trigger:** dev pass@1 falls while train loss falls.
- Save best checkpoint to HF Hub (private) + record the commit/config hash.

## 5. Evaluation protocol

Same harness as A0 (`scripts/run_a0.sh` analogue). For the best checkpoint of each
seed:
- HumanEval+ / MBPP+ greedy pass@1 (base + plus), difficulty-stratified.
- Error breakdown (syntax/runtime/logic/timeout), **per saved checkpoint** → error
  dynamics vs step (RQ2 figure).
- pass@k (k=1,4,16,64) for the diversity/ceiling check (RQ4, H3).
- Three-way comparison A0 vs A1 vs A0', with paired bootstrap CIs and per-difficulty
  breakdown.

## 6. Ablations

- **CoT vs no-CoT** (H4): identical hyperparameters, two data variants.
- (Deferred) data-scale curve if compute allows: 250 / 500 / 1000 examples.

## 7. Acceptance criteria (quantitative gates)

- Pipeline: contamination audit run and reported; SFT data are 100% execution-verified.
- Training healthy: loss drops within 1 epoch; no reward/format collapse; a best
  checkpoint identified before overfit.
- Result gate: A1 pass@1(plus) beats A0 by a **statistically significant** margin
  on at least one benchmark (paired bootstrap, Holm–Bonferroni). If not, report the
  MDE and treat it as a (still publishable) negative result with mechanism analysis.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Prompt-loss not masked | completion-only collator; unit-test the masking |
| Overfit on ~1k examples | dev monitor, early stop, keep intermediate ckpts |
| Format drift at inference | normalize training format to evalplus extractor |
| Too few verified targets (hard prompts) | raise k to 8 on prompts with 0 passes |
| Contamination | audit + report; spot-check 20 nearest pairs by hand |
| Distribution mismatch (train vs eval) | report per-source transfer; keep function-style only |

## 9. Deliverables

- `data/prompt_pool.clean.jsonl`, `data/sft.jsonl` (+ CoT variant)
- `results/contamination_report.json`
- `results/a1_qwen1.5b_{humaneval,mbpp}.json` (× seeds) + error-dynamics table
- SFT checkpoint(s) on HF Hub; config + commit hash recorded
- 3-way comparison table (A0 / A1 / A0') with CIs

## 10. Execution & compute (vGPU-48GB, ~¥2.88/hr)

Teacher already cached on the data disk (no re-download).

1. `pip install -r requirements-train.txt`
2. build pool → audit → rejection-sample SFT data (~0.5–1 h GPU)
3. `train/sft.py --seed {0,1,2}` (~0.5–1 h GPU each)
4. eval best checkpoints (~0.5 h)
5. analysis notebook: 3-way table + error dynamics

Rough budget ~4–6 GPU-hr ≈ ¥12–20; keep ~¥40–50 for debugging headroom.
Cost-saver: run seed 0 end-to-end first (validate loss/eval), then seeds 1–2.
