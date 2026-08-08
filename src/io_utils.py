"""Loading and saving conversations and labels (JSONL on disk)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .schema import AddresseeLabel, Conversation, ConversationLabels


def load_conversations(path: str | Path) -> List[Conversation]:
    """Read a .jsonl file, one Conversation object per line."""
    convs: List[Conversation] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            convs.append(Conversation.from_dict(json.loads(line)))
    return convs


def save_conversations(convs: Iterable[Conversation], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def save_labels(labels_by_conv: Dict[str, ConversationLabels], path: str | Path) -> None:
    """Persist labels as .jsonl: one line per conversation.

    {"conversation_id": "...", "labels": [ {turn_id, addressees, ...}, ... ]}
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conv_id, labels in labels_by_conv.items():
            row = {
                "conversation_id": conv_id,
                "labels": [labels[t].to_dict() for t in sorted(labels)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_labels(path: str | Path) -> Dict[str, ConversationLabels]:
    out: Dict[str, ConversationLabels] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conv_id = str(row["conversation_id"])
            labels: ConversationLabels = {}
            for d in row["labels"]:
                lab = AddresseeLabel.from_dict(d)
                labels[lab.turn_id] = lab
            out[conv_id] = labels
    return out
