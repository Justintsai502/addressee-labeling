"""Agent-view ("is the AGENT addressed?") projection of set-valued addressee labels.

The repo's labels are sets over the other speakers plus GROUP / UNKNOWN. A full-duplex
agent needs one question answered: is *it* being addressed? On human-human corpora
(AMI) we simulate this by treating each seat in turn as the agent and projecting the
set label to one of

  to_agent      the agent seat is among the addressees, or GROUP
  not_to_agent  addressees are other participants only
  uncertain     UNKNOWN
  self          the agent seat is the speaker (no addressee question for the agent)

(`none` = no human speech exists only at the frame level and is produced downstream.)

Votes: a turn labelled K times (sampling) becomes a distribution over
{to_agent, not_to_agent, uncertain} per agent seat.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .schema import GROUP, UNKNOWN

TO, NOT, UNC, SELF = "to_agent", "not_to_agent", "uncertain", "self"
VOTE_CLASSES = (TO, NOT, UNC)


def agent_view(addressees: Iterable[str], speaker: str, agent: str) -> str:
    addr = {a for a in addressees if a}
    if speaker == agent:
        return SELF
    if not addr or UNKNOWN in addr:
        return UNC
    if GROUP in addr or agent in addr:
        return TO
    return NOT


def is_group(addressees: Iterable[str], speaker: str, speakers: Sequence[str]) -> bool:
    """GROUP, or every other participant (AMI encodes whole-group address as the full set)."""
    addr = {a for a in addressees if a}
    others = {s for s in speakers if s != speaker}
    return GROUP in addr or (len(others) >= 2 and others <= addr)


def vote_distribution(samples: List[Iterable[str]], speaker: str, agent: str) -> Dict[str, float]:
    """Fraction of K sampled label sets that project to each class (self excluded upstream)."""
    views = [agent_view(s, speaker, agent) for s in samples]
    c = Counter(v for v in views if v != SELF)
    k = max(1, sum(c.values()))
    return {cls: c.get(cls, 0) / k for cls in VOTE_CLASSES}


def argmax_class(dist: Dict[str, float]) -> str:
    """Majority class; ties broken toward uncertain, then not_to_agent (conservative)."""
    order = {UNC: 0, NOT: 1, TO: 2}
    return max(dist, key=lambda c: (round(dist[c], 9), -order[c]))
