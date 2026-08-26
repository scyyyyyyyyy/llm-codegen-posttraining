# Running on AutoDL (GPU) — verified setup

Generation/scoring runs on a rented AutoDL GPU instance (Linux + CUDA). The
local Mac can only run the CPU gate (`eval/verify_pipeline.py`), because evalplus's
executor uses `setrlimit`, which fails on macOS.

## 1. Rent an instance

- GPU: a 24–48 GB card is enough for 1.5B and 7B in bf16. Verified on **vGPU-48GB**;
  RTX 4090 / 4090D (24 GB) also fine. Avoid RTX 5090 (Blackwell) — too new for the
  pinned torch/vLLM. Avoid <16 GB cards for the 7B teacher.
- Image (镜像): **PyTorch 2.x / Python 3.11 or 3.12 / CUDA 12.x**. Verified on
  `PyTorch 2.5.1 / Python 3.12 / CUDA 12.4`.
- Billing: 按量计费 (pay-as-you-go). **Power off (关机) when idle** or it keeps billing.

## 2. Connect

Console → 容器实例 → copy the SSH command + password. Then either:
- **VSCode**: install the *Remote - SSH* extension → bottom-left `><` → Connect to
  Host → paste the `ssh -p <port> root@<host>` line → enter password → Open Folder
  `/root/autodl-tmp`.
- **Terminal**: `ssh -p <port> root@<host>` and paste the password (invisible when typing).

## 3. Install (once per instance)

```bash
cd ~/autodl-tmp
git clone https://github.com/scyyyyyyyyy/llm-codegen-posttraining.git
cd llm-codegen-posttraining
source /etc/network_turbo          # AutoDL academic acceleration (model/dataset dl)
pip install -r requirements.txt    # vllm + evalplus==0.3.1
```

**Put the HF cache on the DATA disk.** AutoDL's system disk (`/`, ~30 GB) is small;
the default `~/.cache/huggingface` lives there and fills up on the 7B model (~15 GB),
stalling downloads with "Not enough free disk space". Point it at the 50 GB data disk
in every shell that downloads or runs a model:

```bash
export HF_HOME=/root/autodl-tmp/hf
```

Notes / gotchas that already bit us:
- **`evalplus` must be 0.3.1** (pinned). Unpinned, pip resolves to 0.2.1, which has
  no `codegen` CLI → `No module named evalplus.codegen`.
- **Do NOT install `requirements-train.txt` for A0.** `wandb`'s old deps pull
  `pathtools`, which needs the removed `imp` module on Python 3.12 and aborts the
  whole install. Training deps are only needed for A1+.
- **HF Xet 401 on the 7B model.** hf-mirror does not proxy HF's Xet/CAS server
  (`xethub.hf.co`), so newer repos fail with `401 Unauthorized`. `scripts/run_a0.sh`
  exports `HF_HUB_DISABLE_XET=1` to force classic LFS downloads via the mirror; set it
  yourself for manual `evalplus.codegen` / `hf download` runs.

## 4. Verify the eval pipeline (CPU, ~1 min)

```bash
python -m eval.verify_pipeline --n 164
```
Expect `GATE: PASS ✅` (164/164 canonical solutions pass, classifier 5/5).

## 5. Run A0 / A0'

```bash
export HF_ENDPOINT=https://hf-mirror.com
ARM=A0 MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct TAG=qwen1.5b bash scripts/run_a0.sh
ARM="A0'" MODEL=Qwen/Qwen2.5-Coder-7B-Instruct TAG=qwen7b bash scripts/run_a0.sh
```
- Greedy pass@1 is quick/cheap; `pass@k` (n=64) is the time sink. To skip it:
  `DO_PASSK=0 ARM=... MODEL=... TAG=... bash scripts/run_a0.sh`.
- Output: `results/a0_<tag>_<dataset>.json` (pass@1 base/plus, pass@k, error
  breakdown, difficulty stratification). Download via the VSCode file tree.

## 6. Power off

Console → 关机 (keeps the data disk; billing stops). SSH host/port may change next
boot — re-copy the login command.
