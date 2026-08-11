#!/usr/bin/env python3
"""Step 2 — run the CANDIDATE labeler (transcript only). RUN ON THE SERVER.

Two backends; pick with --backend:

  local  (download the weights, load in-process — swap models by id):
      pip install vllm                      # or: transformers torch accelerate
      python scripts/02_run_candidate.py --backend local \
          --model Qwen/Qwen3-32B \
          --conversations data/ami/conversations.jsonl --out outputs/candidate.jsonl
      # try another model by changing --model:
      #   --model Qwen/Qwen3-8B
      #   --model meta-llama/Llama-3.1-8B-Instruct
      #   --engine hf   (universal fallback if vllm can't load a model)

  endpoint  (talk to a running OpenAI-compatible server / hosted API):
      vllm serve Qwen/Qwen3-32B --port 8000
      python scripts/02_run_candidate.py --backend endpoint \
          --model Qwen3-32B --base-url http://localhost:8000/v1 --api-key EMPTY \
          --conversations data/ami/conversations.jsonl --out outputs/candidate.jsonl
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
    ap.add_argument("--backend", choices=["local", "endpoint"], default="local",
                    help="'local' downloads+loads weights in-process; "
                         "'endpoint' calls a running OpenAI-compatible server")
    ap.add_argument("--model", default="Qwen/Qwen3-32B",
                    help="HF id (local) or served model name (endpoint)")
    ap.add_argument("--max-turns-per-window", type=int, default=40)
    ap.add_argument("--context-turns", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    # local backend
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm",
                    help="local backend engine")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="#GPUs (vllm)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="thinking models (Qwen3) spend part of this on a "
                         "<think> trace before the answer — raise further if "
                         "you see truncated/unclosed <think> or unbalanced "
                         "JSON errors")
    # endpoint backend
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    convs = load_conversations(args.conversations)
    if args.backend == "local":
        labeler = get_labeler(
            "local",
            model=args.model,
            backend=args.engine,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_turns_per_window=args.max_turns_per_window,
            context_turns=args.context_turns,
        )
    else:
        labeler = get_labeler(
            "qwen",
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_turns_per_window=args.max_turns_per_window,
            context_turns=args.context_turns,
        )
    run_labeler(labeler, convs, args.out)


if __name__ == "__main__":
    main()
