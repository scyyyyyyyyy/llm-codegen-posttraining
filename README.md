# LLM Post-training for Code Generation

Post-training pipeline for **Qwen2.5-Coder-1.5B-Instruct** on **HumanEval+ / MBPP+**:
`SFT → Rejection Sampling → Iterative DPO`.

**Research question:** does a *partial-credit* reward (score by fraction of tests
passed) beat a *binary* reward (all-or-nothing) on hard problems, where binary
reward gives an all-zero gradient? Paired with an error-type analysis of where
SFT vs DPO actually help.

## Pipeline

```
data/   prepare_sft.py → rejection_sample.py → prepare_dpo.py
train/  sft.py · dpo.py · grpo.py (optional)
eval/   safe_execute.py · error_classify.py · compute_metrics.py
configs/  sft.yaml · dpo.yaml
analysis/ ablation notebooks
```

- `eval/safe_execute.py` — sandboxed subprocess execution with timeout
- `eval/error_classify.py` — syntax / runtime / logic / timeout taxonomy
- `eval/compute_metrics.py` — binary vs partial-credit reward, pass@1, pass@k

## Quickstart

```bash
pip install -r requirements.txt

python data/prepare_sft.py --with-cot
python train/sft.py
python data/rejection_sample.py --model checkpoints/sft --problems data/problems.jsonl
python data/prepare_dpo.py --candidates data/candidates.json --strategy partial
python train/dpo.py --model checkpoints/sft
```

## Ablations

Reward design (binary vs partial), preference strategy (binary / partial /
error-aware), DPO `beta` sensitivity, iterative rounds (R0–R3), and CoT vs no-CoT.
