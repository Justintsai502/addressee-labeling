#!/usr/bin/env python3
"""Step 3 — compare candidate labels against the golden set.

Runs fully offline (no LLM). This is the piece that answers the research question:
"is transcript-only Qwen close enough to the audio+transcript golden set?"

    python scripts/03_evaluate.py \
        --conversations data/golden_sample.jsonl \
        --gold outputs/golden.jsonl \
        --pred outputs/candidate.jsonl \
        --accept-kappa 0.75 --accept-exact 0.80 \
        --report outputs/report.json
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from src.evaluate import acceptance_check, evaluate, format_report
from src.io_utils import load_conversations, load_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--accept-kappa", type=float, default=0.75)
    ap.add_argument("--accept-exact", type=float, default=0.80)
    ap.add_argument("--report", default=None, help="optional path to write report JSON")
    args = ap.parse_args()

    convs = load_conversations(args.conversations)
    gold = load_labels(args.gold)
    pred = load_labels(args.pred)

    # Windows the labeler failed on are stored as UNKNOWN with a "(failed)"
    # labeler tag. They are NOT model predictions — silently scoring them drags
    # the metrics down in a way that looks like real model behaviour, so say so
    # loudly rather than letting them pass as data.
    for name, labels_by_conv in (("gold", gold), ("pred", pred)):
        n_failed = sum(
            1 for labels in labels_by_conv.values() for lab in labels.values()
            if (lab.labeler or "").endswith("(failed)")
        )
        n_total = sum(len(labels) for labels in labels_by_conv.values())
        if n_failed:
            print(f"WARNING: {n_failed}/{n_total} {name} labels come from FAILED "
                  f"windows (recorded as UNKNOWN, not real predictions). "
                  f"Scores below are distorted by these — re-run those windows "
                  f"before trusting the numbers.\n")

    result = evaluate(gold, pred, convs)
    acc = acceptance_check(result, args.accept_kappa, args.accept_exact)
    print(format_report(result, acc))

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"result": result, "acceptance": acc}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nwrote report -> {args.report}")


if __name__ == "__main__":
    main()
