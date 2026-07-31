# A1 — Supervised Fine-Tuning: experiment plan (redesigned)

Status: planned. Pre-registered before running so the analysis isn't chosen after
seeing results.

## 0. The reframe: A1 is a cold-start for RL, not an end in itself

Kimi K3 (arXiv:2607.24653) states its SFT stage establishes *"a high-quality
cold-start policy for the subsequent RL stage"* — SFT's job is to prepare the
policy, then RL does the heavy lifting. We adopt this framing, and it connects A1
directly to this project's core metric.

GRPO's failure mode is the **zero-variance group → zero gradient**: a prompt whose
rollouts all pass (or all fail) contributes nothing. So the real value of a good
SFT cold-start is to **move all-fail (hard, zero-gradient) prompts into the
"some-pass-some-fail" learnable zone — i.e. to raise EPR before RL even starts.**

**A1's headline metric is therefore `EPR@init`**, not raw pass@1: after A1, what
fraction of training prompts have a GRPO group with reward variance? This is cheap
(the sandbox already computes it), sits on the project's spine (EPR = RQ1), is
validated by K3's cold-start framing, and is differentiating (most SFT writeups
report only pass@1; we report "how much learnable signal did SFT create for RL").

This also unifies the pipeline under one ruler: **SFT raises EPR → RL then has
gradient to learn from → OPD is the dense limit.**

Anchors from A0/A0' (greedy pass@1, plus):

| | HumanEval+ | MBPP+ | logic-err (HE+/MBPP+) |
|---|:---:|:---:|:---:|
| A0 student 1.5B | 65.2% | 59.0% | 25.0% / 25.9% |
| A0' teacher 7B  | 87.2% | 72.0% | 7.3% / 14.8% |

## 1. Framing: off-policy distillation

A1 is *off-policy distillation* from the 7B teacher — every target is a
teacher-generated, execution-verified solution. This mirrors K3's SFT (trajectories
synthesized by prior domain-specialist models, then verified). Because OPD (A4) is
*on-policy* distillation from the **same** teacher, `A4 − A1` isolates the value of
on-policyness (exposure-bias fix; DAgger, Ross et al. 2011). A1 is the shared init
for every RL/OPD arm.

## 2. Hypotheses (falsifiable)

- **H1 (headline).** A1 raises `EPR@init` over the base student, especially by
  rescuing hard-tier all-fail prompts into the learnable zone.
- **H2.** A1 raises pass@1(plus) with the gain concentrated in **logic errors**
  (25% → 12–18%); syntax stays ~0%.
- **H3 (diversity cost).** Diversity-preserving SFT raises EPR@init without
  collapsing pass@64. Falsified if pass@64 drops materially.
- **H4 (format readiness).** Format-normalized SFT yields near-100% parseable RL
  rollouts; un-normalized SFT degrades reward computability.

## 3. Data — contamination-safe distilled SFT

Pipeline: `build_prompt_pool → contamination_audit → build_sft_data`.

**Prompt pool (target ~1000, each with executable tests):**
- MBPP `train` split (374; disjoint from MBPP+ eval which derives from the test split).
- KodCode (`KodCode/KodCode-V1`) filtered to Python function-style problems with a
  runnable test; verify field names on load.
- **Pre-filter:** keep a problem only if its reference solution passes its own
  tests (guards bad tests); report drop count.

**Contamination audit (mandatory, reported):**
- Exact function-signature/entry-point match **and** statement-embedding cosine
  ≥ 0.90 vs HumanEval+ / MBPP+ → drop. Report `n_before / removed / n_after`.

**Teacher rejection sampling → targets:**
- Sample **k = 4** from the 7B teacher (temp 0.7, top_p 0.95), keep solutions that
  pass **all** visible tests.
- **Format discipline (K3's XTML lesson):** normalize every target to a single
  ` ```python … ``` ` block that matches the evalplus extractor **and** the future
  RL reward parser — mis-formatted SFT → unparseable RL rollouts → broken reward.
- **Diversity:** keep ≤ 2 dissimilar verified solutions per problem (normalized-code
  distance) so the cold-start preserves exploration for RL.
- **Learnability tag:** record whether the base student solves the prompt (greedy);
  emphasize the "student-fails, teacher-solves" frontier.

Emit variants for the ablations: `sft_base` (1 sol/problem), `sft_div`
(≤2 diverse sols), `sft_learn` (frontier-only).

## 4. Training recipe

- `Qwen2.5-Coder-1.5B-Instruct`; LoRA r=32, α=64, targets = attn {q,k,v,o} + MLP
  {gate,up,down}; TRL `SFTTrainer`.
- lr 2e-4, cosine, warmup 0.05, 3 epochs, effective batch 16, max_len 2048,
  packing, bf16.
- **Completion-only loss** (mask the prompt; `DataCollatorForCompletionOnlyLM`).
- **3 seeds** {0,1,2}; checkpoint every 200 steps (best is often mid-training).
- Monitor train loss + dev eval loss + 50-problem quick pass@1; early-stop when dev
  pass@1 falls while train loss falls.
- Push best checkpoint to HF Hub (private); record config + commit hash.
- LoRA is a compute choice; LoRA ≠ full-FT (Biderman et al. 2024) is stated as a
  limitation. Optional: one full-FT A1 as a robustness check.

## 5. Evaluation

For the best checkpoint of each seed, same harness as A0:
- pass@1 (base+plus), difficulty-stratified; error breakdown per checkpoint (RQ2
  error dynamics); pass@k (k=1,4,16,64).
- **EPR@init** on the training pool (`eval/epr_init.py`): base vs A1, stratified by
  difficulty — the headline A1 result.
- Three-way table A0 / A1 / A0' with paired-bootstrap CIs.

## 6. Ablations

- **Diversity** (`sft_base` vs `sft_div`) → EPR@init and pass@64 (H3).
- **Format normalization** on/off → RL rollout parse rate (H4).
- (Deferred) CoT vs no-CoT; full-FT robustness check.

## 7. Acceptance gates

- Contamination audit run + reported; SFT targets 100% execution-verified.
- Training healthy; a best checkpoint identified before overfit.
- **Primary gate:** A1 raises EPR@init over base by a clear margin (this is the
  cold-start's job), and pass@1(plus) beats A0 significantly on ≥1 benchmark
  (paired bootstrap, Holm–Bonferroni). A null result is reported honestly with MDE.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Prompt loss not masked | completion-only collator; unit-test the mask |
| Overfit on ~1k examples | dev monitor, early stop, keep intermediate ckpts |
| Format drift → broken RL reward | normalize to the evalplus/RL parser; measure parse rate |
| Too few verified targets on hard prompts | raise k to 8 where 0 passes |
| Diversity collapse (hurts RL) | keep ≤2 diverse sols; track pass@64 |
| Contamination | audit + report; hand-check 20 nearest pairs |

## 9. Deliverables

- `data/prompt_pool.clean.jsonl`, `data/sft_{base,div,learn}.jsonl`
- `results/contamination_report.json`, `results/epr_init_*.json`
- `results/a1_qwen1.5b_{humaneval,mbpp}.json` (× seeds) + error-dynamics
- SFT checkpoint(s) on HF Hub; config + commit hash
- A0 / A1 / A0' comparison table with CIs

## 10. Execution & compute (vGPU-48GB, ~¥2.88/hr; teacher already cached)

1. `pip install -r requirements-train.txt`
2. `python -m data.build_prompt_pool` → `python -m data.contamination_audit`
3. `python -m data.build_sft_data` (7B teacher sampling; ~0.5–1 h GPU)
4. `python train/sft.py --seed {0,1,2}` (~0.5–1 h each)
5. `python -m eval.epr_init` (base vs A1) + eval best checkpoints
6. analysis: 3-way table + EPR@init + error dynamics

Rough budget ~4–6 GPU-hr ≈ ¥12–20; keep ~¥40–50 headroom. Validate seed 0
end-to-end before seeds 1–2.

## Related work anchor

The SFT→RL→OPD structure and the OPD-as-dense-token-reward view are exactly Kimi
K3's SFT → RL → Multi-Teacher On-Policy Distillation pipeline (arXiv:2607.24653);
K3's per-token OPD reward is the clipped teacher/student log-ratio — the L2 signal
this project places at the dense end of the reward-density spectrum. GRPO: DeepSeek
(Shao et al. 2024; Guo et al. 2025). Distillation-beats-small-model-RL: DeepSeek-R1.
