"""Render a conversation (or a window of it) into the text a labeler reads.

Both the golden (Gemini, audio+transcript) and candidate (Qwen, transcript-only)
labelers must be shown *exactly the same* speaker ids and turn ids, so their
outputs are directly comparable. This module is the single source of truth for
that rendering.

Long conversations are processed in windows: a block of `target` turns to label,
preceded by a few read-only `context` turns so the model can see who was talking
before the window starts. `[CTX]` lines are context-only; `[>>]` lines are to be
labeled.
"""

from __future__ import annotations

from typing import List, Tuple

from .schema import Conversation, Turn


def format_turn(t: Turn, marker: str) -> str:
    return f"{marker} (turn {t.turn_id}) [{t.start:.2f}-{t.end:.2f}] {t.speaker}: {t.text}"


def render_window(conv: Conversation, context: List[Turn], target: List[Turn]) -> str:
    """Render one labeling window as plain text."""
    lines: List[str] = []
    lines.append(f"Speakers in this conversation: {', '.join(conv.speakers)}")
    lines.append("")
    lines.append("Legend: [CTX] = context only, do NOT label.  [>>] = label this turn.")
    lines.append("")
    for t in context:
        lines.append(format_turn(t, "[CTX]"))
    for t in target:
        lines.append(format_turn(t, "[>> ]"))
    return "\n".join(lines)


def iter_windows(
    conv: Conversation,
    max_turns_per_window: int,
    context_turns: int,
) -> List[Tuple[List[Turn], List[Turn]]]:
    """Split a conversation's turns into (context, target) windows.

    Windows tile the conversation with no target overlap; each carries up to
    `context_turns` preceding turns as read-only context.
    """
    turns = conv.turns
    windows: List[Tuple[List[Turn], List[Turn]]] = []
    i = 0
    n = len(turns)
    step = max(1, max_turns_per_window)
    while i < n:
        target = turns[i : i + step]
        ctx_start = max(0, i - context_turns)
        context = turns[ctx_start:i]
        windows.append((context, target))
        i += step
    return windows
