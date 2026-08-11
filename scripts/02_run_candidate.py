#!/usr/bin/env python3
"""Step 2 — run the CANDIDATE labeler: Qwen, transcript only.

RUN ON THE SERVER. Either self-host Qwen with vLLM or point at a hosted endpoint.

    # self-hosted (weights download happens here, on the server):
    vllm serve Qwen/Qwen3-32B --port 8000
    python scripts/02_run_candidate.py \
        --conversations data/golden_sample.jsonl \
        --out outputs/candidate.jsonl \
        --model Qwen3-32B \
        --base-url http://localhost:8000/v1 --api-key EMPTY
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.io_utils import load_conversations
from src.labelers import get_labeler
from src.pipeline import run_labeler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--out", default="outputs/candidate.jsonl")
    ap.add_argument("--model", default="Qwen3-32B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--max-turns-per-window", type=int, default=40)
    ap.add_argument("--context-turns", type=int, default=10)
    args = ap.parse_args()

    convs = load_conversations(args.conversations)
    labeler = get_labeler(
        "qwen",
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_turns_per_window=args.max_turns_per_window,
        context_turns=args.context_turns,
    )
    run_labeler(labeler, convs, args.out)


if __name__ == "__main__":
    main()
