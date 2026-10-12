"""ADS v3 / prompt v5: clause-level, causal addressee labels with K-slot votes (CPU-only logic).

Spec: mpmoshi `artifacts/ads_v3_labeling_1011/TODO.md` (rev 3) and `docs/ADS_LABEL_SPEC.md` §v3.

- `split_clauses`: rule-based clause splitter with character offsets (every character covered once).
- `render_causal` / `render_full`: the user prompt for one target clause.
- `parse_vote`: one raw LLM answer -> normalized addressee set ("ALL" or sorted names) or a fail reason.
- `class_for` / `aggregate`: seat-set -> 5-class agent view, and K votes -> soft target + mask.

No UNKNOWN class: an UNKNOWN answer is an invalid vote. A clause with < MIN_VALID valid votes is masked.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .parsing import _extract_json_blob

ALL = "ALL"
CLASSES = ("none", "self", "to_agent_itself", "to_everyone", "not_to_agent")
SPEECH_CLASSES = ("to_agent_itself", "to_everyone", "not_to_agent")
MIN_VALID = 3
FAIL_REASONS = ("unparsable", "illegal_name", "self_address", "missing_target", "unknown")

# ---------------------------------------------------------------------------- clauses
_WORD = re.compile(r"[A-Za-z0-9'\[\]-]+")
# words that may stand next to a name in a vocative piece ("hey guys,", "okay Theo,")
VOC_FILLER = {"hey", "hi", "hello", "okay", "ok", "so", "well", "and", "guys", "everyone", "everybody", "folks",
              "all", "oh", "um", "uh", "right", "alright", "yo", "listen", "look", "you", "two", "both"}
MIN_CLAUSE_WORDS = 3


@dataclass
class Clause:
    idx: int
    start: int  # char offset, inclusive
    end: int    # char offset, exclusive (includes trailing whitespace before the next clause)
    text: str   # text[start:end].strip()


def _words(s: str) -> List[str]:
    return [w.strip("[]'-").lower() for w in _WORD.findall(s)]


def _is_vocative(piece: str, names: set) -> bool:
    ws = [w for w in _words(piece) if w]
    return bool(ws) and any(w in names for w in ws) and all(w in names or w in VOC_FILLER for w in ws)


def _split_at(text: str, pattern: str, lo: int, hi: int) -> List[Tuple[int, int]]:
    """Split text[lo:hi] after each match of `pattern` followed by whitespace; spans tile [lo, hi)."""
    cuts = [m.end() for m in re.finditer(pattern + r"\s+", text[lo:hi])]
    bounds = [lo] + [lo + c for c in cuts if lo + c < hi] + [hi]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def split_clauses(text: str, names: Iterable[str] = ()) -> List[Clause]:
    """Sentences at [.?!;]; inside a sentence, pieces at commas, merged so that
    (1) a vocative piece ("Theo,", "hey guys,") attaches to the clause it opens (or, at the end, closes), and
    (2) no clause inside a sentence has fewer than MIN_CLAUSE_WORDS words (short pieces merge forward;
        a short last piece merges back). Sentences are never merged with each other."""
    nameset = {n.lower() for n in names}
    out: List[Tuple[int, int]] = []
    for s_lo, s_hi in _split_at(text, r"[.?!;]+[\"')\]]*", 0, len(text)):
        pieces = _split_at(text, r",", s_lo, s_hi)
        sent: List[Tuple[int, int]] = []
        buf: Optional[Tuple[int, int]] = None
        for p in pieces:
            buf = p if buf is None else (buf[0], p[1])
            seg = text[buf[0]:buf[1]]
            is_last = p is pieces[-1]
            if not is_last and (len(_words(seg)) < MIN_CLAUSE_WORDS or _is_vocative(seg, nameset)):
                continue  # carry forward
            sent.append(buf)
            buf = None
        if buf is not None:
            sent.append(buf)
        # a short or vocative-only last piece ("..., Bob?") goes back to the previous clause
        if len(sent) > 1:
            last = text[sent[-1][0]:sent[-1][1]]
            if len(_words(last)) < MIN_CLAUSE_WORDS or _is_vocative(last, nameset):
                sent[-2:] = [(sent[-2][0], sent[-1][1])]
        out.extend(sent)
    if not out:
        out = [(0, len(text))]
    return [Clause(i, a, b, text[a:b].strip()) for i, (a, b) in enumerate(out)]


# ---------------------------------------------------------------------------- prompts
_RULES = """\
LABEL SET (closed)
- One or more of the OTHER participants, by name (never the speaker themselves).
- "GROUP": the speaker addresses everyone else present (all other participants).
There is no "unknown" option: if you are unsure, give the best-supported answer and lower your confidence.

HOW TO DECIDE (label WHO is addressed, not whether a reply is expected)
1. Vocatives / names: "Bob, can you ...", "... right, Dr. Lee?" -> that person. A name said earlier in the
   same turn still holds for later clauses of that turn unless the speaker turns to someone else.
2. "you guys", "everyone", "all of you", "anyone", or a question / announcement / proposal to the room -> GROUP.
3. Answering a question or request -> the person who asked it, especially when it was put to this speaker.
4. Reacting to someone's specific proposal or claim (agreeing, objecting, adding a caveat) -> that person.
5. Answering a question that was asked to the whole room can be either the asker or GROUP; choose the one the
   wording supports ("you" singular / the asker's point -> asker; "we", "let's", "everyone" -> GROUP) and lower
   your confidence.
6. A person mentioned in the third person ("Sam and I will handle it") is NOT the addressee.
7. Short reactions and backchannels ("yeah", "sure", "got it") -> the person they react to.
"""

_CUE = """\
CUE
Also give "cue": the shortest exact words, copied from the speaker's current turn as shown, that most decide
the addressee (e.g. "Bob", "you guys", "what do you think"), or "none_needed" if the earlier turns alone decide
it (e.g. a plain answer to a question that was put to this speaker).

OUTPUT
Return ONLY a JSON object, no other text:
{"addressees": ["<name>" | "GROUP", ...], "cue": "<exact words>" | "none_needed", "confidence": <0.0-1.0>}
"""

_HEAD = """\
You are an expert conversation analyst labeling addressee information for a multi-speaker spoken-dialogue
dataset. You get the transcript only (no audio). The speaker's current turn is split into clauses; decide
WHO the speaker is talking TO in the clause marked ⟦like this⟧.

"""

SYSTEM_V5_CAUSAL = _HEAD + _RULES + """
CAUSAL SETTING
You see the conversation only up to the END OF THE MARKED CLAUSE: earlier turns, and the current turn up to
and including the marked clause. Later clauses and later turns are not shown. Decide as a listener would at
the moment the marked clause ends. Do not guess from who might answer next.

""" + _CUE

SYSTEM_V5_FULL = _HEAD + _RULES + """
FULL CONTEXT
You see the whole conversation, including the rest of the current turn and later turns. You may use them
(for example, who answers next).

""" + _CUE

PROMPTS_V5 = {"causal": SYSTEM_V5_CAUSAL, "full": SYSTEM_V5_FULL}


def _marked(text: str, clauses: Sequence[Clause], k: int, upto: bool) -> str:
    c = clauses[k]
    head = text[:c.start]
    body = text[c.start:c.end].rstrip()
    tail = "" if upto else text[c.end:].strip()
    return (head + "⟦" + body + "⟧" + (" " + tail if tail else "")).strip()


def render_causal(speakers: Sequence[str], turns: Sequence[Tuple[str, str]], t: int, clauses: Sequence[Clause],
                  k: int, context_turns: int = 10) -> str:
    """turns = [(speaker, text)]; target = clause k of turn t; shows <= context_turns earlier turns."""
    spk, text = turns[t]
    lines = [f"Participants: {', '.join(speakers)}", "", "Earlier turns:"]
    lo = max(0, t - context_turns)
    lines += [f"(turn {i}) {s}: {x}" for i, (s, x) in enumerate(turns[lo:t], lo)] or ["(none)"]
    lines += ["", f"Current turn (turn {t}, speaker: {spk}):", f"{spk}: {_marked(text, clauses, k, upto=True)}", "",
              f"Who is {spk} talking to in the marked clause? Return the JSON object now."]
    return "\n".join(lines)


def render_full(speakers: Sequence[str], turns: Sequence[Tuple[str, str]], t: int, clauses: Sequence[Clause],
                k: int) -> str:
    spk, _ = turns[t]
    lines = [f"Participants: {', '.join(speakers)}", "", "Conversation (the current turn is marked >>):"]
    for i, (s, x) in enumerate(turns):
        if i == t:
            lines.append(f">> (turn {i}) {s}: {_marked(x, clauses, k, upto=False)}")
        else:
            lines.append(f"   (turn {i}) {s}: {x}")
    lines += ["", f"Who is {spk} talking to in the marked clause of turn {t}? Return the JSON object now."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- votes
_GROUP_WORDS = {"group", "everyone", "everybody", "all", "all others", "the group", "the room"}


def parse_vote(raw: str, speakers: Sequence[str], speaker: str) -> Dict:
    """-> {"norm": "ALL" | "name1|name2" | None, "fail": reason | None, "cue": str | None, "confidence": float | None}.
    Any illegal / self / UNKNOWN entry invalidates the whole vote (no silent dropping)."""
    res = {"norm": None, "fail": None, "cue": None, "confidence": None}
    try:
        data = json.loads(_extract_json_blob(raw))
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {"addressees": data}
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except Exception:
        res["fail"] = "unparsable"
        return res
    cue = data.get("cue")
    res["cue"] = str(cue).strip() if cue is not None else None
    conf = data.get("confidence")
    res["confidence"] = float(conf) if isinstance(conf, (int, float)) else None
    addr = data.get("addressees", data.get("addressee"))
    if isinstance(addr, str):
        addr = [addr]
    if not isinstance(addr, list) or not [a for a in addr if str(a).strip()]:
        res["fail"] = "missing_target"
        return res
    by_lower = {s.lower(): s for s in speakers}
    others = {s for s in speakers if s != speaker}
    got, group = set(), False
    for a in addr:
        a = str(a).strip().strip("[]").strip()
        al = a.lower()
        if al == "unknown":
            res["fail"] = "unknown"
            return res
        if al in _GROUP_WORDS:
            group = True
        elif al in by_lower:
            if by_lower[al] == speaker:
                res["fail"] = "self_address"
                return res
            got.add(by_lower[al])
        else:
            res["fail"] = "illegal_name"
            return res
    res["norm"] = ALL if group or got == others else "|".join(sorted(got))
    return res


def class_for(norm: str, speaker: str, agent: str) -> str:
    """Mapping table (spec §1.1) for one valid vote and agent seat (both by display name)."""
    if speaker == agent:
        return "self"
    if norm == ALL:
        return "to_everyone"
    return "to_agent_itself" if agent in norm.split("|") else "not_to_agent"


def aggregate(norms: Sequence[Optional[str]], speaker: str, agent: str) -> Dict:
    """K final votes (None = invalid) -> soft target over SPEECH_CLASSES + mask; self if the agent speaks."""
    if speaker == agent:
        return {"soft": None, "valid": sum(n is not None for n in norms), "mask": 1, "hard": "self"}
    valid = [n for n in norms if n is not None]
    if len(valid) < MIN_VALID:
        return {"soft": None, "valid": len(valid), "mask": 0, "hard": None}
    c = Counter(class_for(n, speaker, agent) for n in valid)
    soft = {k: c.get(k, 0) / len(valid) for k in SPEECH_CLASSES}
    # ties: fixed order (to_everyone, to_agent_itself, not_to_agent) only for the reported hard label
    order = {"to_everyone": 2, "to_agent_itself": 1, "not_to_agent": 0}
    hard = max(soft, key=lambda k: (soft[k], order[k]))
    return {"soft": soft, "valid": len(valid), "mask": 1, "hard": hard}


def majority_norm(norms: Sequence[Optional[str]]) -> Tuple[Optional[str], int]:
    valid = [n for n in norms if n is not None]
    if not valid:
        return None, 0
    (n, c), = Counter(valid).most_common(1)
    return n, c
