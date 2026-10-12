#!/usr/bin/env bash
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/agent_view_v3/logs/%x_%j.out
#SBATCH --error=outputs/agent_view_v3/logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/work/b12901015/addressee-labeling}"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
/work/b12901015/venvs/nemo_asr_tiny_v2/bin/python scripts/09_asr_check_ttv2.py "$@"
