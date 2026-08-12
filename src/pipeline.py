"""Glue: run a labeler over conversations and persist, with a progress print.

Checkpoints after every window (not just at the end), and resumes from an
existing --out file if one is already there — a killed job (SLURM time limit,
Ctrl+C) loses at most one in-flight window's work, and a retry with the same
--out path picks up where it left off instead of re-labeling (and re-paying
API cost or GPU time for) turns already done.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

from .io_utils import load_labels, save_labels
from .labelers import AddresseeLabeler
from .schema import Conversation, ConversationLabels


def run_labeler(
    labeler: AddresseeLabeler,
    convs: List[Conversation],
    out_path: str | Path,
    verbose: bool = True,
) -> Dict[str, ConversationLabels]:
    out_path = Path(out_path)
    labels_by_conv: Dict[str, ConversationLabels] = {}
    if out_path.exists():
        labels_by_conv = load_labels(out_path)
        n_done = sum(len(v) for v in labels_by_conv.values())
        if verbose and n_done:
            print(f"resuming from {out_path}: {n_done} turns already labeled",
                  file=sys.stderr)

    for i, conv in enumerate(convs, 1):
        t0 = time.time()
        existing = labels_by_conv.get(conv.conversation_id, {})

        def checkpoint(labels: ConversationLabels, conv=conv) -> None:
            labels_by_conv[conv.conversation_id] = labels
            save_labels(labels_by_conv, out_path)

        labels_by_conv[conv.conversation_id] = labeler.label_conversation(
            conv, existing=existing, on_progress=checkpoint
        )
        if verbose:
            dt = time.time() - t0
            print(f"[{i}/{len(convs)}] {conv.conversation_id}: "
                  f"{len(conv.turns)} turns labeled by {labeler.name} ({dt:.1f}s)",
                  file=sys.stderr)
    save_labels(labels_by_conv, out_path)
    if verbose:
        print(f"wrote labels -> {out_path}", file=sys.stderr)
    return labels_by_conv
