#!/usr/bin/env python3
"""Benchmark processing time across models on the same transcript.

Answers "how long does each model take to label the same material?" — run each
model over one identical slice of conversation and report load time, generation
time, throughput, and time relative to the audio's real duration.

    python3 scripts/05_benchmark_speed.py \
        --conversations data/ami/bench_5min.jsonl \
        --models Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B \
        --engine hf --max-turns-per-window 10 --max-new-tokens 16384 \
        --out outputs/benchmark_speed.json

Every model gets IDENTICAL settings (same data, window size, token budget,
thinking mode) — otherwise the timings aren't comparable. Models are loaded and
freed one at a time so a large one can't fail for lack of GPU memory held by a
previous one.

Labels are still written (one file per model, --labels-dir) so the same run can
be scored for accuracy afterwards — speed alone isn't the whole picture.
"""
from __future__ import annotations

import os

# vLLM spawns worker processes for its engine; with the default "fork" start
# method any CUDA state in this parent process makes the children fail with
# "Cannot re-initialize CUDA in forked subprocess". Set before importing torch
# or vllm so it actually takes effect.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from src.io_utils import load_conversations, save_labels
from src.labelers import get_labeler


def free_gpu() -> None:
    """Release the previous model's GPU memory before loading the next.

    Must NOT initialize CUDA as a side effect: vLLM forks subprocesses for its
    engine, and CUDA already initialized in the parent makes those children die
    with "Cannot re-initialize CUDA in forked subprocess". So only touch the
    allocator when CUDA is *already* initialized (i.e. a previous HF model
    loaded it) — never as the first CUDA call of the process.

    Best-effort only: this is cleanup, so it must never kill a benchmark run.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  (gpu cleanup skipped: {type(e).__name__}: {str(e)[:80]})",
              file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--engine", choices=["vllm", "hf"], default="hf")
    ap.add_argument("--max-turns-per-window", type=int, default=10)
    ap.add_argument("--context-turns", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--out", default="outputs/benchmark_speed.json")
    ap.add_argument("--labels-dir", default="outputs/benchmark_labels",
                    help="per-model label files, so accuracy can be scored too")
    args = ap.parse_args()

    convs = load_conversations(args.conversations)
    n_turns = sum(len(c.turns) for c in convs)
    audio_seconds = sum(
        max((t.end for t in c.turns), default=0.0) - min((t.start for t in c.turns), default=0.0)
        for c in convs
    )
    print(f"benchmark data: {len(convs)} conversation(s), {n_turns} turns, "
          f"{audio_seconds/60:.1f} min of audio\n", file=sys.stderr)

    Path(args.labels_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for model in args.models:
        print(f"{'='*60}\n{model}\n{'='*60}", file=sys.stderr)
        free_gpu()
        labeler = get_labeler(
            "local",
            model=model,
            backend=args.engine,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=not args.disable_thinking,
            max_turns_per_window=args.max_turns_per_window,
            context_turns=args.context_turns,
        )
        t_wall = time.time()
        labels = {c.conversation_id: labeler.label_conversation(c) for c in convs}
        wall = time.time() - t_wall

        n_failed = sum(1 for v in labels.values() for l in v.values()
                       if (l.labeler or "").endswith("(failed)"))
        safe = model.replace("/", "_")
        save_labels(labels, Path(args.labels_dir) / f"{safe}.jsonl")

        s = labeler.stats
        row = {
            "model": model,
            "engine": args.engine,
            "load_seconds": round(s["load_seconds"], 1),
            "generate_seconds": round(s["generate_seconds"], 1),
            "wall_seconds": round(wall, 1),
            "n_calls": s["n_calls"],
            "input_tokens": s["input_tokens"],
            "output_tokens": s["output_tokens"],
            "tokens_per_second": round(
                s["output_tokens"] / max(s["generate_seconds"], 0.01), 1),
            "seconds_per_turn": round(wall / max(n_turns, 1), 2),
            "realtime_factor": round(wall / max(audio_seconds, 0.01), 2),
            "failed_windows": n_failed,
        }
        results.append(row)
        print(f"\n-> {model}: {row['wall_seconds']}s total "
              f"({row['load_seconds']}s load + {row['generate_seconds']}s generate), "
              f"{row['tokens_per_second']} tok/s, {row['failed_windows']} failed\n",
              file=sys.stderr)

        del labeler
        free_gpu()

    # ---- summary table ----
    print("\n" + "=" * 96)
    print(f"SPEED BENCHMARK — {n_turns} turns / {audio_seconds/60:.1f} min audio, "
          f"engine={args.engine}, window={args.max_turns_per_window}, "
          f"thinking={'off' if args.disable_thinking else 'on'}")
    print("=" * 96)
    hdr = (f"{'model':<24}{'load s':>8}{'gen s':>9}{'total s':>9}"
           f"{'tok/s':>8}{'s/turn':>8}{'xRT':>7}{'fail':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['model']:<24}{r['load_seconds']:>8.1f}{r['generate_seconds']:>9.1f}"
              f"{r['wall_seconds']:>9.1f}{r['tokens_per_second']:>8.1f}"
              f"{r['seconds_per_turn']:>8.2f}{r['realtime_factor']:>7.2f}"
              f"{r['failed_windows']:>6}")
    print("=" * 96)
    print("xRT = wall time / audio duration (1.0 = realtime; lower is faster)")
    print("fail = windows that errored and were recorded UNKNOWN "
          "(a fast model with failures is not actually doing the work)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "conversations": args.conversations,
                "n_turns": n_turns,
                "audio_seconds": round(audio_seconds, 1),
                "engine": args.engine,
                "max_turns_per_window": args.max_turns_per_window,
                "context_turns": args.context_turns,
                "max_new_tokens": args.max_new_tokens,
                "thinking": not args.disable_thinking,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
