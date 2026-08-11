"""Robustly turn a model's raw text output into AddresseeLabel objects.

LLMs wrap JSON in ```json fences, add prose, or emit a bare list instead of the
requested object. This normalizes all of that and validates the labels against the
closed label set for the turn's speaker.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from .schema import AddresseeLabel, Conversation, UNKNOWN


def _extract_json_blob(text: str) -> str:
    """Pull the first balanced {...} or [...] out of a raw model response."""
    text = text.strip()
    # Reasoning models (Qwen3, and similar) emit a <think>...</think> block
    # before the real answer. Strip it first so stray braces in the reasoning
    # prose are never mistaken for the JSON answer.
    think_close = text.rfind("</think>")
    if think_close != -1:
        text = text[think_close + len("</think>"):].strip()
    elif text.lstrip().startswith("<think>"):
        raise ValueError(
            "model output is an unclosed <think> block with no answer after it — "
            "likely ran out of max_new_tokens before finishing; got: " + text[:200]
        )
    # Strip code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Find the first opening bracket and scan to its match.
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise ValueError("no JSON object/array found in model output")
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]
    raise ValueError(
        f"unbalanced JSON in model output ({len(text) - start} chars scanned "
        f"from the opening bracket, never closed) — almost always means "
        f"generation got cut off by max_new_tokens before finishing; "
        f"tail of raw output: ...{text[-150:]!r}"
    )


def parse_label_response(
    raw: str,
    conv: Conversation,
    target_turn_ids: List[int],
    labeler_name: str,
) -> Dict[int, AddresseeLabel]:
    """Parse and validate a labeler response for a set of target turns.

    Invalid addressee ids are dropped; a turn with no valid addressee left becomes
    UNKNOWN. Turns the model forgot are filled with UNKNOWN so every target turn
    always gets a label (so evaluation coverage is well-defined).
    """
    blob = _extract_json_blob(raw)
    data = json.loads(blob)
    if isinstance(data, dict) and "labels" in data:
        items = data["labels"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    target_set = set(target_turn_ids)
    by_id: Dict[int, AddresseeLabel] = {}
    for it in items:
        try:
            tid = int(it["turn_id"])
        except (KeyError, ValueError, TypeError):
            continue
        if tid not in target_set:
            continue
        turn = conv.turn_by_id(tid)
        if turn is None:
            continue
        allowed = set(conv.allowed_addressees(turn.speaker))
        addr = it.get("addressees", it.get("addressee", []))
        if isinstance(addr, str):
            addr = [addr]
        addr = [str(a).strip() for a in addr]
        # Keep only labels valid for this speaker; drop self-address & unknown ids.
        addr = [a for a in addr if a in allowed]
        if not addr:
            addr = [UNKNOWN]
        conf = it.get("confidence")
        by_id[tid] = AddresseeLabel(
            turn_id=tid,
            addressees=addr,
            confidence=(float(conf) if isinstance(conf, (int, float)) else None),
            rationale=(str(it["rationale"]) if it.get("rationale") is not None else None),
            labeler=labeler_name,
        ).normalized()

    # Fill any missing target turn with UNKNOWN so coverage == 100%.
    for tid in target_turn_ids:
        if tid not in by_id:
            by_id[tid] = AddresseeLabel(
                turn_id=tid, addressees=[UNKNOWN], confidence=0.0,
                rationale="missing from model output", labeler=labeler_name,
            )
    return by_id
