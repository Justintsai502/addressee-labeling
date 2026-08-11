#!/usr/bin/env python3
"""Step 1 — build the GOLDEN set with Gemini 3 Pro (audio + transcript).

RUN ON THE SERVER (needs google-genai + GEMINI_API_KEY + the audio files).

    export GEMINI_API_KEY=...
    python scripts/01_build_golden.py \
        --conversations data/golden_sample.jsonl \
        --out outputs/golden.jsonl \
        --model gemini-3-pro

Cost control: run this only on the sampled golden subset (see docs on stratified
sampling), not the full corpus.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (sets sys.path)
from src.io_utils import load_conversations
from src.labelers import get_labeler
from src.pipeline import run_labeler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--out", default="outputs/golden.jsonl")
    ap.add_argument("--model", default="gemini-3-pro")
    ap.add_argument("--max-turns-per-window", type=int, default=40)
    ap.add_argument("--context-turns", type=int, default=10)
    ap.add_argument("--no-audio", action="store_true",
                    help="ablation: run Gemini on transcript only (isolates audio's value)")
    args = ap.parse_args()

    convs = load_conversations(args.conversations)
    labeler = get_labeler(
        "gemini",
        model=args.model,
        max_turns_per_window=args.max_turns_per_window,
        context_turns=args.context_turns,
        send_audio=not args.no_audio,
    )
    run_labeler(labeler, convs, args.out)


if __name__ == "__main__":
    main()
