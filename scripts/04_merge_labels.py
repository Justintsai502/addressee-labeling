#!/usr/bin/env python3
"""Step 4 — merge addressee labels back into the transcript.

This is the actual TODO-1 deliverable: input a transcript (conversations.jsonl),
output the SAME transcript with each turn carrying its addressee. Feed this into
the next data-prep step (turn-level addressee -> frame-level <spk>/<ads> training
tokens). Runs fully offline — pure Python, no model calls.

Two outputs:
  --out-jsonl   conversations.jsonl where every turn gains "addressee",
                "addressee_confidence", "addressee_labeler". Same schema as the
                input, so it still loads with load_conversations().
  --out-text    optional: one human-readable "<conv_id>.txt" per conversation,
                e.g. "[50.42-50.99] B -> A: Okay."

    python3 scripts/04_merge_labels.py \
        --conversations data/ami/conversations.jsonl \
        --labels outputs/golden.jsonl \
        --out-jsonl outputs/ami_labeled.jsonl \
        --out-text outputs/transcripts_annotated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import _bootstrap  # noqa: F401
from src.io_utils import load_conversations, load_labels
from src.schema import Conversation, ConversationLabels


def merge_conversation(conv: Conversation, labels: ConversationLabels) -> dict:
    d = conv.to_dict()
    for t in d["turns"]:
        lab = labels.get(t["turn_id"])
        if lab is None:
            t["addressee"] = None
            t["addressee_confidence"] = None
            t["addressee_labeler"] = None
            continue
        norm = lab.normalized()
        t["addressee"] = norm.addressees
        t["addressee_confidence"] = norm.confidence
        t["addressee_labeler"] = norm.labeler
    return d


def render_text(conv: Conversation, labels: ConversationLabels) -> str:
    lines = []
    for t in conv.turns:
        lab = labels.get(t.turn_id)
        addr = "+".join(lab.normalized().addressees) if lab else "?"
        lines.append(f"[{t.start:7.2f}-{t.end:7.2f}] {t.speaker:>3} -> {addr:<10}: {t.text}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--labels", required=True,
                    help="a labels file from 01_build_golden.py or 02_run_candidate.py")
    ap.add_argument("--out-jsonl", default="outputs/labeled_conversations.jsonl")
    ap.add_argument("--out-text", default=None,
                    help="optional directory: write one <conv_id>.txt per conversation")
    args = ap.parse_args()

    convs = load_conversations(args.conversations)
    labels_by_conv: Dict[str, ConversationLabels] = load_labels(args.labels)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n_turns = n_labeled = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for conv in convs:
            labels = labels_by_conv.get(conv.conversation_id, {})
            f.write(json.dumps(merge_conversation(conv, labels), ensure_ascii=False) + "\n")
            n_turns += len(conv.turns)
            n_labeled += sum(1 for t in conv.turns if t.turn_id in labels)
    print(f"wrote {args.out_jsonl}  ({n_labeled}/{n_turns} turns carry an addressee)")

    if args.out_text:
        out_dir = Path(args.out_text)
        out_dir.mkdir(parents=True, exist_ok=True)
        for conv in convs:
            labels = labels_by_conv.get(conv.conversation_id, {})
            (out_dir / f"{conv.conversation_id}.txt").write_text(
                render_text(conv, labels), encoding="utf-8"
            )
        print(f"wrote per-conversation .txt -> {out_dir}/")


if __name__ == "__main__":
    main()
