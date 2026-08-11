#!/usr/bin/env bash
# Download a few AMI meetings (Mix-Headset audio) + the manual annotations, then
# convert them into data/ami/conversations.jsonl.
#
# Usage:   bash scripts/download_ami.sh ES2002a ES2002b
# Default: ES2002a ES2002b   (same 4 participants; they self-introduce by name)
#
# AMI corpus © University of Edinburgh, released under CC-BY 4.0.
# Data is large + license-attributed — it is gitignored, never committed.
set -euo pipefail
cd "$(dirname "$0")/.."

MEETINGS=("${@:-}")
if [ -z "${MEETINGS[*]}" ]; then MEETINGS=(ES2002a ES2002b); fi

AUDIO_BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANN_URL="https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"

mkdir -p data/ami/audio data/ami/annotations

echo ">> audio"
for M in "${MEETINGS[@]}"; do
  if [ ! -f "data/ami/audio/$M.Mix-Headset.wav" ]; then
    echo "   downloading $M.Mix-Headset.wav"
    curl -sSL -o "data/ami/audio/$M.Mix-Headset.wav" "$AUDIO_BASE/$M/audio/$M.Mix-Headset.wav"
  else
    echo "   have $M.Mix-Headset.wav"
  fi
done

echo ">> annotations"
if [ ! -f data/ami/ami_manual.zip ]; then
  curl -sSL -o data/ami/ami_manual.zip "$ANN_URL"
fi
# extract only the words+segments for the requested meetings
for M in "${MEETINGS[@]}"; do
  unzip -o -q data/ami/ami_manual.zip \
    "words/$M.*.words.xml" "segments/$M.*.segments.xml" -d data/ami/annotations/
done

echo ">> convert"
python3 scripts/00_prepare_ami.py --meetings "${MEETINGS[@]}" --out data/ami/conversations.jsonl
echo ">> done -> data/ami/conversations.jsonl"
