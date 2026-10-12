#!/usr/bin/env bash
# Usage: sbatch -A <ACCOUNT> --job-name=adsv5 --time=02:30:00 scripts/sbatch_clause_label.sh <11_clause_label.py args...>
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=outputs/agent_view_v3/logs/%x_%j.out
#SBATCH --error=outputs/agent_view_v3/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/work/b12901015/addressee-labeling}"
export HF_HOME="${HF_HOME:-/work/b12901015/hf_cache}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# vLLM workers must be spawned, not forked (see scripts/05_benchmark_speed.py); fork -> "CUDA driver initialization failed"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY="${PY:-/work/b12901015/envs/qwen3-vllm/bin/python}"
"$PY" -c 'import torch,sys; ok=torch.cuda.is_available(); print("cuda", ok, torch.cuda.get_device_name(0) if ok else "-"); sys.exit(0 if ok else 75)' || { echo "no usable CUDA; abort" >&2; exit 75; }
echo "start: job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-?} args=$*"
"$PY" scripts/11_clause_label.py "$@"
echo "done: job=${SLURM_JOB_ID:-local}"
