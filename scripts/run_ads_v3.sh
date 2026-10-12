#!/usr/bin/env bash
# ADS v3 labeling pipeline, all steps in one GPU job (see docs/ADS_V3_PIPELINE.md).
#
#   TTV2_ROOT=/path/to/turn_taking_2000_v2 OUT=outputs/ads_v3_v5/my_run IDS_FILE=ids.txt \
#   PY_VLLM=/path/to/vllm-env/bin/python PY_NEMO=/path/to/nemo-env/bin/python \
#   sbatch -A <ACCOUNT> --job-name=adsv3 --time=04:00:00 scripts/run_ads_v3.sh
#
# Required: TTV2_ROOT, OUT, PY_VLLM, PY_NEMO, and one of IDS_FILE / ALL=1 / CONVS (an existing conversations JSONL).
# Optional: MODEL (Qwen/Qwen3-32B), ASR_MODEL (nvidia/parakeet-tdt-0.6b-v2, or a .nemo path), K (5), SEED (20261011),
#           FIXED (fixed cases JSON), GREEDY=1, FULL=1 (also a full-context comparison run), STEPS ("prepare label align
#           frames report"), AUDIO_PATTERN, PY_CPU (python3), HF_HOME, LOCK (gates file for the report),
#           BACKEND=mock (CPU dry run of the labeling step).
# Every step can be re-run: labeling resumes (--resume), alignment skips finished dialogues.
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
# relative paths are relative to where the command was run
for v in TTV2_ROOT OUT IDS_FILE CONVS FIXED LOCK; do
  if [[ -n "${!v:-}" && "${!v}" != /* ]]; then printf -v "$v" '%s' "$PWD/${!v}"; fi
done
# repo root: next to this script when run with bash; sbatch runs a spooled copy, so fall back to the submit dir
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$here/scripts/00c_prepare_ttv2.py" ]]; then cd "$here"
elif [[ -f "${SLURM_SUBMIT_DIR:-.}/scripts/00c_prepare_ttv2.py" ]]; then cd "$SLURM_SUBMIT_DIR"
else echo "cannot find the addressee-labeling repo: sbatch from the repo root, or run with bash" >&2; exit 2; fi
: "${TTV2_ROOT:?set TTV2_ROOT}" "${OUT:?set OUT}" "${PY_VLLM:?set PY_VLLM}" "${PY_NEMO:?set PY_NEMO}"
PY_CPU="${PY_CPU:-python3}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
ASR_MODEL="${ASR_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
K="${K:-5}"; SEED="${SEED:-20261011}"
DEFAULT_AUDIO_PATTERN='audio_qwen3-tts-1.7b-clone_{seat}.wav'   # (a literal } inside \${..:-..} would end it)
AUDIO_PATTERN="${AUDIO_PATTERN:-$DEFAULT_AUDIO_PATTERN}"
STEPS="${STEPS:-prepare label align frames report}"
BACKEND="${BACKEND:-vllm}"   # mock = CPU dry run of the labeling step (random answers)
CONVS="${CONVS:-$OUT/convs.jsonl}"
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn   # fork -> "CUDA driver initialization failed"
mkdir -p "$OUT" logs
has() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "[$(date '+%F %T')] $*"; }

if { has label && [[ "$BACKEND" != mock ]]; } || has align; then
  "$PY_VLLM" -c 'import torch,sys; ok=torch.cuda.is_available(); print("cuda", ok); sys.exit(0 if ok else 75)' \
    || { echo "no usable CUDA on ${SLURMD_NODENAME:-this node}; abort" >&2; exit 75; }
fi

if has prepare && [[ ! -s "$CONVS" ]]; then
  log "1/5 prepare conversations -> $CONVS"
  sel=()
  if [[ -n "${IDS_FILE:-}" ]]; then sel=(--ids-file "$IDS_FILE"); elif [[ "${ALL:-0}" == 1 ]]; then sel=(--all);
  else echo "set IDS_FILE or ALL=1 (or CONVS to an existing file)" >&2; exit 2; fi
  "$PY_CPU" scripts/00c_prepare_ttv2.py --root "$TTV2_ROOT" "${sel[@]}" --out-convs "$CONVS" --out-ref "$OUT/generator_ref.jsonl"
fi

if has label; then
  log "2/5 clause votes (causal) with $MODEL"
  extra=(); [[ -n "${FIXED:-}" ]] && extra+=(--fixed "$FIXED"); [[ "${GREEDY:-0}" == 1 ]] && extra+=(--greedy)
  "$PY_VLLM" scripts/11_clause_label.py --convs "$CONVS" --mode causal --backend "$BACKEND" --model "$MODEL" --k "$K" --seed "$SEED" \
    --out-dir "$OUT/causal" --resume "${extra[@]}"
  if [[ "${FULL:-0}" == 1 ]]; then
    log "2b/5 clause votes (full context, comparison only)"
    "$PY_VLLM" scripts/11_clause_label.py --convs "$CONVS" --mode full --backend "$BACKEND" --model "$MODEL" --k "$K" --seed "$SEED" \
      --out-dir "$OUT/full" --resume ${FIXED:+--fixed "$FIXED"}
  fi
fi

if has align; then
  log "3/5 word timestamps with $ASR_MODEL"
  "$PY_NEMO" scripts/12_align_ttv2.py --root "$TTV2_ROOT" --convs "$CONVS" --out-dir "$OUT/align" \
    --asr-model "$ASR_MODEL" --audio-pattern "$AUDIO_PATTERN"
fi

if has frames; then
  log "4/5 frame labels (12.5 Hz)"
  "$PY_NEMO" scripts/13_build_frames_v5.py --convs "$CONVS" --votes "$OUT/causal/clause_votes.jsonl" \
    --align-dir "$OUT/align" --root "$TTV2_ROOT" --out-dir "$OUT/frames" --audio-pattern "$AUDIO_PATTERN"
fi

if has report; then
  log "5/5 report"
  rep=(--causal "$OUT/causal" --frames "$OUT/frames" --out "$OUT/report.json")
  [[ -d "$OUT/full" ]] && rep+=(--full "$OUT/full")
  [[ -s "$OUT/generator_ref.jsonl" ]] && rep+=(--ref "$OUT/generator_ref.jsonl")
  [[ -n "${FIXED:-}" ]] && rep+=(--fixed "$FIXED")
  [[ -n "${LOCK:-}" ]] && rep+=(--lock "$LOCK")
  "$PY_CPU" scripts/14_report_v5.py "${rep[@]}"
fi
log "done: $OUT"
