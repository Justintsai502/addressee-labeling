"""Abstract labeler + a windowed template method shared by all backends."""

from __future__ import annotations

import abc
from typing import Dict, List

from ..schema import AddresseeLabel, Conversation
from ..transcript_format import iter_windows, render_window


class AddresseeLabeler(abc.ABC):
    """Label every turn of a conversation with its addressee(s).

    Subclasses implement `_label_window`. `label_conversation` handles the
    windowing (so long conversations don't overflow context) and stitches the
    per-window results back together. `_prepare` / `_cleanup` let a backend do
    per-conversation setup such as uploading the audio file once.
    """

    #: Human-readable id stored on every label (e.g. "gemini-3-pro").
    name: str = "base"

    def __init__(self, max_turns_per_window: int = 40, context_turns: int = 10) -> None:
        self.max_turns_per_window = max_turns_per_window
        self.context_turns = context_turns

    # -- backend hooks --------------------------------------------------------
    def _prepare(self, conv: Conversation) -> None:  # noqa: B027 - optional hook
        """Per-conversation setup (default: nothing)."""

    def _cleanup(self, conv: Conversation) -> None:  # noqa: B027 - optional hook
        """Per-conversation teardown (default: nothing)."""

    @abc.abstractmethod
    def _label_window(
        self, conv: Conversation, context: List, target: List, window_text: str
    ) -> Dict[int, AddresseeLabel]:
        """Return {turn_id: AddresseeLabel} for the target turns of one window."""
        raise NotImplementedError

    # -- public API -----------------------------------------------------------
    def label_conversation(self, conv: Conversation) -> Dict[int, AddresseeLabel]:
        self._prepare(conv)
        try:
            labels: Dict[int, AddresseeLabel] = {}
            for context, target in iter_windows(
                conv, self.max_turns_per_window, self.context_turns
            ):
                window_text = render_window(conv, context, target)
                labels.update(self._label_window(conv, context, target, window_text))
            return labels
        finally:
            self._cleanup(conv)

    def label_all(self, convs: List[Conversation]) -> Dict[str, Dict[int, AddresseeLabel]]:
        return {c.conversation_id: self.label_conversation(c) for c in convs}
