#!/usr/bin/env bash
# A0 / A0' zero-shot baselines, to run on an AutoDL GPU instance (Linux + CUDA).
#
# A0  = student  Qwen2.5-Coder-1.5B-Instruct
# A0' = teacher  Qwen2.5-Coder-7B-Instruct   (OPD ceiling reference)
#
# Flow per dataset (evalplus 0.3.1): codegen (vLLM) -> sanitize -> evaluate,
# for BOTH greedy (pass@1) and n-sample (pass@k). Then eval.run_a0 adds our
# error breakdown + difficulty stratification -> results/a0_<tag>_<dataset>.json.
#
# --- AutoDL notes ---------------------------------------------------------
#   Run these ONCE in the SSH session before this script:
#     source /etc/network_turbo          # academic acceleration (model/dataset dl)
#   HF mirror is exported below. One 24-48GB GPU handles 1.5B and 7B in bf16.
# --------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
TAG="${TAG:-qwen1.5b}"            # use qwen7b for A0'
ROOT="${ROOT:-evalplus_results}"
NSAMPLES="${NSAMPLES:-64}"
TEMP="${TEMP:-0.8}"
TP="${TP:-1}"                    # tensor-parallel; 1 GPU
DO_PASSK="${DO_PASSK:-1}"        # set 0 to skip pass@k (faster / cheaper)
DATASETS="${DATASETS:-humaneval mbpp}"   # override e.g. DATASETS=mbpp
PARALLEL="${PARALLEL:-8}"        # evalplus eval workers; too high OOMs the pool on pass@k

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# hf-mirror does not proxy HF's Xet/CAS server (xethub.hf.co) -> 401 on newer
# repos (e.g. the 7B). Disable Xet so downloads fall back to classic LFS via mirror.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# newest samples .jsonl in a dir, ignoring sanitized / results files
newest_samples() { ls -t "$1"/*.jsonl 2>/dev/null | grep -v -e sanitized -e eval_results | head -1; }
newest_results() { ls -t "$1"/*eval_results.json 2>/dev/null | head -1; }

gen_eval() {   # $1=dataset  $2=mode(greedy|passk)  $3=root
  local dataset="$1" mode="$2" root="$3"
  # IMPORTANT: everything except the final path must go to stderr, otherwise the
  # caller's $(gen_eval ...) captures all the vLLM/codegen logs as the "path".
  {
    # vLLM's engine can throw during SHUTDOWN teardown (e.g. libnvrtc.so.13 missing
    # on CUDA 12 images) AFTER the samples file is written; don't let that abort us.
    set +e
    if [ "$mode" = "greedy" ]; then
      python -m evalplus.codegen "$MODEL" "$dataset" --greedy --backend vllm --tp "$TP" --root "$root"
    else
      python -m evalplus.codegen "$MODEL" "$dataset" --n_samples "$NSAMPLES" \
          --temperature "$TEMP" --backend vllm --tp "$TP" --root "$root"
    fi
    set -e
    local raw; raw="$(newest_samples "${root}/${dataset}")"
    if [ -z "$raw" ]; then
      echo "ERROR: codegen produced no samples in ${root}/${dataset}" >&2
      return 1
    fi
    python -m evalplus.sanitize --samples "$raw" >/dev/null
    SAN="${raw%.jsonl}-sanitized.jsonl"
    [ -f "$SAN" ] || SAN="$raw"       # fall back if sanitize named it differently
    python -m evalplus.evaluate --dataset "$dataset" --samples "$SAN" --parallel "$PARALLEL"
  } 1>&2
  newest_results "${root}/${dataset}"   # the ONLY thing on stdout
}

for DATASET in $DATASETS; do
  echo "############ ${DATASET} :: ${TAG} ############"
  # Tag-scoped roots so different models never share a samples dir (a failed run
  # must NOT silently pick up another model's leftover samples).
  GREEDY_RESULTS="$(gen_eval "$DATASET" greedy "${ROOT}/${TAG}/greedy")"
  echo "greedy results: ${GREEDY_RESULTS}"

  PASSK_ARG=()
  if [ "$DO_PASSK" = "1" ]; then
    PASSK_RESULTS="$(gen_eval "$DATASET" passk "${ROOT}/${TAG}/passk")"
    echo "pass@k results: ${PASSK_RESULTS}"
    PASSK_ARG=(--passk-results "${PASSK_RESULTS}")
  fi

  python -m eval.run_a0 --dataset "$DATASET" --tag "$TAG" \
      --greedy-results "${GREEDY_RESULTS}" "${PASSK_ARG[@]}"
done

echo "Done. Summaries in results/a0_${TAG}_*.json"
