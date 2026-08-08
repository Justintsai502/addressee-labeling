"""Core data structures for addressee labeling.

A *conversation* is a diarized, time-aligned transcript (plus an audio file) with
two or more speakers. For every turn we want an *addressee label*: to whom is the
speaker talking? The addressee is drawn from the other speakers in the
conversation, plus two special labels:

  - GROUP   : the speaker is addressing everyone / the whole group.
  - UNKNOWN : the addressee cannot be determined from the available signal.

The label is a *set* of speaker ids (a speaker can address more than one person at
once), never including the speaker themselves.

This schema is dataset-agnostic. AMI, PersonaPlex, or any other multi-party corpus
is converted into a list of `Conversation` objects before labeling; the downstream
Moshi/Mimi token stream (<spk>/<ads>) is produced from these labels in a separate
data-prep step (not part of this validation harness).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Special (non-speaker) addressee labels.
GROUP = "GROUP"
UNKNOWN = "UNKNOWN"
SPECIAL_LABELS = (GROUP, UNKNOWN)


@dataclass
class Turn:
    """A single diarized speaking turn."""

    turn_id: int
    speaker: str
    start: float  # seconds
    end: float    # seconds
    text: str
    # True if this turn overlaps in time with another speaker's turn. Overlap
    # turns are the hardest case for text-only labeling, so we track them for
    # stratified evaluation.
    overlap: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Turn":
        return cls(
            turn_id=int(d["turn_id"]),
            speaker=str(d["speaker"]),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            text=str(d.get("text", "")),
            overlap=bool(d.get("overlap", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Conversation:
    """A diarized conversation with its audio and speaker roster."""

    conversation_id: str
    turns: List[Turn]
    speakers: List[str] = field(default_factory=list)
    audio_path: Optional[str] = None  # absolute or config-relative path to the audio file
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Derive the speaker roster from the turns if not given explicitly, so
        # both labelers see exactly the same closed label set.
        if not self.speakers:
            seen: List[str] = []
            for t in self.turns:
                if t.speaker not in seen:
                    seen.append(t.speaker)
            self.speakers = seen
        self.turns = sorted(self.turns, key=lambda t: (t.start, t.turn_id))

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)

    def allowed_addressees(self, speaker: str) -> List[str]:
        """Valid addressee labels for a turn spoken by `speaker`."""
        others = [s for s in self.speakers if s != speaker]
        return others + list(SPECIAL_LABELS)

    def turn_by_id(self, turn_id: int) -> Optional[Turn]:
        for t in self.turns:
            if t.turn_id == turn_id:
                return t
        return None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Conversation":
        return cls(
            conversation_id=str(d["conversation_id"]),
            turns=[Turn.from_dict(t) for t in d["turns"]],
            speakers=[str(s) for s in d.get("speakers", [])],
            audio_path=d.get("audio_path"),
            meta=dict(d.get("meta", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "audio_path": self.audio_path,
            "speakers": list(self.speakers),
            "meta": self.meta,
            "turns": [t.to_dict() for t in self.turns],
        }


@dataclass
class AddresseeLabel:
    """The addressee decision for one turn, produced by a labeler."""

    turn_id: int
    addressees: List[str]                 # subset of allowed_addressees(speaker)
    confidence: Optional[float] = None    # 0..1, optional (used to flag weak golden labels)
    rationale: Optional[str] = None       # short free-text justification, optional
    labeler: Optional[str] = None         # name of the labeler that produced this

    def normalized(self) -> "AddresseeLabel":
        """Deduplicate + sort addressees for stable comparison."""
        addr = sorted(set(a for a in self.addressees if a))
        # UNKNOWN is exclusive: if the labeler is unsure it should not also name
        # concrete addressees. Collapse to UNKNOWN when mixed.
        if UNKNOWN in addr and len(addr) > 1:
            addr = [UNKNOWN]
        return AddresseeLabel(
            turn_id=self.turn_id,
            addressees=addr,
            confidence=self.confidence,
            rationale=self.rationale,
            labeler=self.labeler,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AddresseeLabel":
        addr = d.get("addressees", d.get("addressee", []))
        if isinstance(addr, str):
            addr = [addr]
        return cls(
            turn_id=int(d["turn_id"]),
            addressees=[str(a) for a in addr],
            confidence=(float(d["confidence"]) if d.get("confidence") is not None else None),
            rationale=d.get("rationale"),
            labeler=d.get("labeler"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "addressees": list(self.addressees),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "labeler": self.labeler,
        }


# A conversation's labels: turn_id -> AddresseeLabel
ConversationLabels = Dict[int, AddresseeLabel]
