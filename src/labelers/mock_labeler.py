"""Offline heuristic labeler — NO LLM, NO network.

Purpose: let the whole pipeline (windowing -> labeling -> evaluation -> report)
run end-to-end on a laptop with zero dependencies, for development and for the unit
test. It is a rule-based straw man, not a serious labeler.

The `noise` parameter randomly perturbs a fraction of decisions, which is handy for
producing a *candidate* stream that visibly disagrees with a *golden* stream so the
evaluation metrics exercise real (non-100%) numbers in the demo.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

from ..schema import AddresseeLabel, Conversation, GROUP, UNKNOWN
from .base import AddresseeLabeler

_BACKCHANNELS = {
    "yeah", "yep", "yes", "mm", "mm-hm", "mmhm", "uh-huh", "right", "ok", "okay",
    "sure", "exactly", "true", "haha", "hmm", "oh", "wow", "i see", "got it",
}


def _is_backchannel(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > 3:
        return False
    return any(w in _BACKCHANNELS for w in words) or len(words) <= 2


class MockHeuristicLabeler(AddresseeLabeler):
    def __init__(self, noise: float = 0.0, seed: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.name = f"mock-heuristic(noise={noise})"
        self.noise = noise
        self._rng = random.Random(seed)

    def _label_window(
        self, conv: Conversation, context: List, target: List, window_text: str
    ) -> Dict[int, AddresseeLabel]:
        all_turns = list(context) + list(target)
        out: Dict[int, AddresseeLabel] = {}
        for t in target:
            others = [s for s in conv.speakers if s != t.speaker]
            addressee = self._guess(conv, all_turns, t, others)
            # Inject noise: flip to a different plausible addressee.
            if self.noise and self._rng.random() < self.noise and others:
                choices = [o for o in others if o not in addressee] + [GROUP]
                addressee = [self._rng.choice(choices)]
            out[t.turn_id] = AddresseeLabel(
                turn_id=t.turn_id, addressees=addressee, confidence=0.5,
                rationale="heuristic", labeler=self.name,
            ).normalized()
        return out

    def _guess(self, conv, all_turns, turn, others: List[str]) -> List[str]:
        if not others:
            return [UNKNOWN]
        # 1. Explicit vocative: another speaker's id appears in the text.
        for o in others:
            if re.search(rf"\b{re.escape(o)}\b", turn.text, re.IGNORECASE):
                return [o]
        # 2. Two-party stretch -> the only other person.
        if len(others) == 1:
            return [others[0]]
        # 3. Backchannel or reply -> whoever spoke most recently before this turn.
        prev = self._prev_speaker(all_turns, turn)
        if prev and prev != turn.speaker:
            return [prev]
        # 4. Fallback: addressing the group.
        return [GROUP]

    @staticmethod
    def _prev_speaker(all_turns, turn) -> Optional[str]:
        prev = None
        for t in all_turns:
            if t.turn_id == turn.turn_id:
                break
            if t.speaker != turn.speaker:
                prev = t.speaker
        return prev
