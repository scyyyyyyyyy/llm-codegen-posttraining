#!/usr/bin/env bash
# A0 / A0' zero-shot baselines, to run on an AutoDL GPU instance (Linux + CUDA).
#
# A0  = student  Qwen2.5-Coder-1.5B-Instruct
# A0' = teacher  Qwen2.5-Coder-7B-Instruct   (OPD ceiling reference)
#
# Produces, per dataset: greedy pass@1 (base + plus) and pass@k (k=1,4,16,64),
# plus our error breakdown + difficulty stratification via eval.run_a0.
#
# --- AutoDL notes ---------------------------------------------------------
#   HuggingFace is often unreachable from CN; use the mirror:
#     export HF_ENDPOINT=https://hf-mirror.com
#   evalplus pulls its datasets from GitHub releases; if slow, enable AutoDL's
#   academic acceleration in the SSH session BEFORE running:
#     source /etc/network_turbo
#   One 24GB GPU (3090/4090) is enough for 1.5B and 7B in bf16.
# --------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
TAG="${TAG:-qwen1.5b}"            # use qwen7b for A0'
ROOT="${ROOT:-evalplus_results}"
NSAMPLES="${NSAMPLES:-64}"
TEMP="${TEMP:-0.8}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

newest_results() {   # newest *_eval_results.json under $ROOT/$1
  ls -t "${ROOT}/$1"/*eval_results.json 2>/dev/null | head -1
}

for DATASET in humaneval mbpp; do
  echo "############ ${DATASET} :: ${TAG} ############"

  # 1) greedy -> pass@1 (evalplus generates AND evaluates in one call)
  python -m evalplus.evaluate --model "${MODEL}" --dataset "${DATASET}" \
      --backend vllm --greedy --root "${ROOT}"
  GREEDY_RESULTS="$(newest_results "${DATASET}")"
  echo "greedy results: ${GREEDY_RESULTS}"

  # 2) n-sample -> pass@k
  python -m evalplus.evaluate --model "${MODEL}" --dataset "${DATASET}" \
      --backend vllm --n_samples "${NSAMPLES}" --temperature "${TEMP}" --root "${ROOT}"
  PASSK_RESULTS="$(newest_results "${DATASET}")"
  echo "pass@k results: ${PASSK_RESULTS}"

  # 3) collect: pass@1/pass@k + error breakdown + difficulty stratification
  python -m eval.run_a0 --dataset "${DATASET}" --tag "${TAG}" \
      --greedy-results "${GREEDY_RESULTS}" \
      --passk-results  "${PASSK_RESULTS}"
done

echo "Done. Summaries in results/a0_${TAG}_*.json"
