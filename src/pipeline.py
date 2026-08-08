"""Glue: run a labeler over conversations and persist, with a progress print."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

from .io_utils import save_labels
from .labelers import AddresseeLabeler
from .schema import Conversation, ConversationLabels


def run_labeler(
    labeler: AddresseeLabeler,
    convs: List[Conversation],
    out_path: str | Path,
    verbose: bool = True,
) -> Dict[str, ConversationLabels]:
    labels_by_conv: Dict[str, ConversationLabels] = {}
    for i, conv in enumerate(convs, 1):
        t0 = time.time()
        labels_by_conv[conv.conversation_id] = labeler.label_conversation(conv)
        if verbose:
            dt = time.time() - t0
            print(f"[{i}/{len(convs)}] {conv.conversation_id}: "
                  f"{len(conv.turns)} turns labeled by {labeler.name} ({dt:.1f}s)",
                  file=sys.stderr)
    save_labels(labels_by_conv, out_path)
    if verbose:
        print(f"wrote labels -> {out_path}", file=sys.stderr)
    return labels_by_conv
