#!/usr/bin/env python3
"""Step 1 — build the GOLDEN set: strongest available audio+transcript model.

Two backends; pick with --backend:

  gemini (default): Gemini via Google API.
      export GEMINI_API_KEY=...
      python scripts/01_build_golden.py --conversations data/golden_sample.jsonl --out outputs/golden.jsonl
      # Pro-tier models need billing enabled on the AI Studio project (free tier
      # grants 0 quota for them) — see README for the gemini-2.x/3.x-flash fallback.

  openai: an audio-capable OpenAI chat model (e.g. gpt-4o-audio-preview).
      export OPENAI_API_KEY=...
      python scripts/01_build_golden.py --backend openai --conversations data/golden_sample.jsonl --out outputs/golden_openai.jsonl

Both write the same label format, so you can run BOTH on the same sample and
diff them (via 03_evaluate.py, treating one as "gold" and the other as "pred")
as a cross-check: if two independent strong models agree, that's a much
stronger signal than either one alone, since neither is verified human truth.

Cost control: run this only on a small stratified sample, not the full corpus —
audio tokens aren't free on the Pro/flagship tier.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.io_utils import load_conversations
from src.labelers import get_labeler
from src.pipeline import run_labeler

DEFAULT_MODELS = {
    "gemini": "gemini-3.1-pro-preview",
    "openai": "gpt-4o-audio-preview",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--out", default="outputs/golden.jsonl")
    ap.add_argument("--backend", choices=["gemini", "openai"], default="gemini")
    ap.add_argument("--model", default=None,
                    help=f"default per backend: {DEFAULT_MODELS}")
    ap.add_argument("--max-turns-per-window", type=int, default=40)
    ap.add_argument("--context-turns", type=int, default=10)
    ap.add_argument("--max-output-tokens", type=int, default=8192,
                    help="reasoning models spend part of this on invisible "
                         "thinking before the visible answer — raise if you "
                         "see truncated-output errors on long windows")
    ap.add_argument("--no-audio", action="store_true",
                    help="ablation: run on transcript only (isolates audio's value)")
    args = ap.parse_args()

    convs = load_conversations(args.conversations)
    model = args.model or DEFAULT_MODELS[args.backend]
    labeler = get_labeler(
        args.backend,
        model=model,
        max_turns_per_window=args.max_turns_per_window,
        context_turns=args.context_turns,
        max_output_tokens=args.max_output_tokens,
        send_audio=not args.no_audio,
    )
    run_labeler(labeler, convs, args.out)


if __name__ == "__main__":
    main()
