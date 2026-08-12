"""Abstract labeler + a windowed template method shared by all backends."""

from __future__ import annotations

import abc
from typing import Callable, Dict, List, Optional

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
    def label_conversation(
        self,
        conv: Conversation,
        existing: Optional[Dict[int, AddresseeLabel]] = None,
        on_progress: Optional[Callable[[Dict[int, AddresseeLabel]], None]] = None,
    ) -> Dict[int, AddresseeLabel]:
        """Label every turn, skipping ones already present in `existing`.

        `on_progress(labels)` fires after every window with the accumulated
        labels so far — the caller can checkpoint to disk, so a killed job
        (SLURM time limit, Ctrl+C) loses at most one in-flight window, not
        everything. Pass the same `existing` back in on retry to resume
        instead of re-labeling (and re-paying for) turns already done.
        """
        self._prepare(conv)
        try:
            labels: Dict[int, AddresseeLabel] = dict(existing or {})
            for context, target in iter_windows(
                conv, self.max_turns_per_window, self.context_turns
            ):
                if all(t.turn_id in labels for t in target):
                    continue  # this whole window was already done on a prior run
                window_text = render_window(conv, context, target)
                labels.update(self._label_window(conv, context, target, window_text))
                if on_progress is not None:
                    on_progress(labels)
            return labels
        finally:
            self._cleanup(conv)

    def label_all(self, convs: List[Conversation]) -> Dict[str, Dict[int, AddresseeLabel]]:
        return {c.conversation_id: self.label_conversation(c) for c in convs}
