#!/usr/bin/env python3
"""Label conversations K times (sampling) and save the votes; optional greedy pass.

    # CPU dry run (no model)
    python3 scripts/06_vote_label.py --convs data/ami/dacts_pilot.jsonl --backend mock \
        --out-dir outputs/agent_view_v3/mock

    # server (vLLM, local weights)
    python3 scripts/06_vote_label.py --convs data/ami/dacts_pilot.jsonl --backend vllm \
        --model Qwen/Qwen3-32B --k 5 --temperature 0.7 --greedy \
        --out-dir outputs/agent_view_v3/qwen3_32b

Writes <out-dir>/votes.jsonl ({conversation_id, labeler, k, temperature, turns:[{turn_id, samples}]}),
<out-dir>/greedy.jsonl (repo label format, if --greedy) and <out-dir>/run.json (settings + stats).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from src.io_utils import load_conversations, save_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backend", choices=("vllm", "mock"), default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--prompt", choices=("v3", "v4"), default="v3", help="v4 adds the reply/addressee rules (2026-10-10)")
    ap.add_argument("--causal", action="store_true", help="one request per turn with only past context (no later turns)")
    ap.add_argument("--greedy", action="store_true", help="also run one temperature-0 pass (same prompt)")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-turns-per-window", type=int, default=40)
    ap.add_argument("--context-turns", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    convs = load_conversations(args.convs)
    win = dict(max_turns_per_window=args.max_turns_per_window, context_turns=args.context_turns)

    if args.backend == "mock":
        from src.labelers.voting_labeler import MockVotingLabeler

        labeler = MockVotingLabeler(k=args.k, **win)
    else:
        from src.labelers.voting_labeler import CausalVotingLabeler, VotingLocalLabeler

        labeler = (CausalVotingLabeler if args.causal else VotingLocalLabeler)(k=args.k, sample_temperature=args.temperature, prompt=args.prompt, model=args.model, backend="vllm",
                                     temperature=0.0, max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
                                     enable_thinking=not args.no_think, **win)
    t0 = time.time()
    for i, conv in enumerate(convs, 1):
        labeler.label_conversation(conv)
        print(f"[vote {i}/{len(convs)}] {conv.conversation_id}: {len(conv.turns)} turns ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    with open(out / "votes.jsonl", "w", encoding="utf-8") as f:
        for conv in convs:
            v = labeler.votes.get(conv.conversation_id, {})
            f.write(json.dumps({"conversation_id": conv.conversation_id, "labeler": labeler.name, "k": args.k,
                                "temperature": args.temperature,
                                "turns": [{"turn_id": tid, "samples": v[tid]} for tid in sorted(v)]}, ensure_ascii=False) + "\n")

    run = {"args": vars(args), "labeler": labeler.name, "vote_seconds": time.time() - t0,
           "stats": getattr(labeler, "stats", None)}
    if args.greedy:
        t1 = time.time()
        if args.backend == "mock":
            from src.labelers.mock_labeler import MockHeuristicLabeler

            g = MockHeuristicLabeler(**win)
        else:
            labeler.k, labeler.name = 1, labeler.name.split("-vote")[0] + "-greedy"
            g = labeler
        greedy = {c.conversation_id: g.label_conversation(c) for c in convs}
        save_labels(greedy, out / "greedy.jsonl")
        run["greedy_seconds"] = time.time() - t1
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    print(f"wrote {out}/votes.jsonl" + (" and greedy.jsonl" if args.greedy else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
