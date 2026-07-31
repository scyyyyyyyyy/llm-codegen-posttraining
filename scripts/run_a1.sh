#!/usr/bin/env bash
# A1 SFT pipeline, to run on an AutoDL GPU instance (teacher already cached).
#
#   data pool -> contamination audit -> teacher distilled SFT data
#   -> EPR@init(base) -> SFT (seed 0) -> EPR@init(a1) -> eval
#
# Validate seed 0 end-to-end first; add seeds 1,2 once it looks healthy.
set -euo pipefail

TEACHER="${TEACHER:-/root/autodl-tmp/Qwen2.5-Coder-7B-Instruct}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATA="${DATA:-data/sft_base.jsonl}"
SEED="${SEED:-0}"
CKPT="${CKPT:-checkpoints/sft-s${SEED}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "### 1. prompt pool"
python -m data.build_prompt_pool --out data/prompt_pool.jsonl

echo "### 2. contamination audit"
python -m data.contamination_audit --pool data/prompt_pool.jsonl \
    --out data/prompt_pool.clean.jsonl --report results/contamination_report.json

echo "### 3. distilled SFT data (teacher rejection sampling)"
python -m data.build_sft_data --pool data/prompt_pool.clean.jsonl \
    --teacher "$TEACHER" --student "$STUDENT" --out-prefix data/sft

echo "### 4. EPR@init (base student, before SFT)"
python -m eval.epr_init --model "$STUDENT" --tag base --pool data/prompt_pool.clean.jsonl

echo "### 5. SFT (seed ${SEED})"
python train/sft.py --data "$DATA" --base-model "$STUDENT" --seed "$SEED" --out "$CKPT"

echo "### 6. EPR@init (A1, after SFT)"
python -m eval.epr_init --model "$CKPT" --tag "a1-s${SEED}" --pool data/prompt_pool.clean.jsonl

echo "Done. Next: eval the checkpoint with the A0 harness and compare A0/A1/A0'."
