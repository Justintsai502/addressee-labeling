#!/usr/bin/env python3
"""Tests for ADS v3 / prompt v5 logic (rev 3 contract). Run: `python3 tests/test_clause_v5.py` (numpy needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clause_v5 import ALL, aggregate, class_for, parse_vote, render_causal, split_clauses  # noqa: E402
from src.frames_v5 import CLASS_IDX, FRAME, assign_span, build_view, clause_spans  # noqa: E402

_fail = 0


def check(name, cond, extra=""):
    global _fail
    if not cond:
        _fail += 1
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + str(extra)) if extra and not cond else ''}")


SPK = ["Mara", "Theo", "Priya", "Jonas"]

# --- splitter -------------------------------------------------------------------------------
fc = json.loads((ROOT / "data/ttv2/fixed_cases_v5.json").read_text())
for c in fc["cases"]:
    got = [x.text for x in split_clauses(c["turns"][-1][1], fc["speakers"])]
    check(f"splitter fixed case {c['case_id']}", got == c["expect_clauses"], got)
for t in ["Yeah, I'm down but I have to head out in about forty minutes.", "Sure.", "", "a, b, c, d.",
          "Right, so we still need to melt wax, pour them into molds, and add wicks."]:
    cl = split_clauses(t, SPK)
    check(f"offsets tile the text: {t[:30]!r}", "".join(t[c.start:c.end] for c in cl) == t and
          all(a.end == b.start for a, b in zip(cl, cl[1:])))

# --- causal rendering hides later clauses ----------------------------------------------------
turns = [("Theo", "I think we should leave at six."), ("Mara", "I agree with you, Theo. Priya, what do you think?")]
cl = split_clauses(turns[1][1], SPK)
u0 = render_causal(SPK, turns, 1, cl, 0)
check("causal prompt of clause 0 does not show clause 1", "Priya, what" not in u0 and "⟦I agree with you, Theo.⟧" in u0, u0)
u1 = render_causal(SPK, turns, 1, cl, 1)
check("causal prompt of clause 1 shows clause 0 unmarked", "I agree with you, Theo. ⟦Priya" in u1, u1)

# --- vote parsing / normalization -----------------------------------------------------------
pv = lambda a, spk="Mara": parse_vote(json.dumps({"addressees": a, "cue": "x", "confidence": 0.5}), SPK, spk)
check("GROUP -> ALL", pv(["GROUP"])["norm"] == ALL)
check("all other seats -> ALL", pv(["Theo", "Priya", "Jonas"])["norm"] == ALL)
check("subset -> sorted names", pv(["Theo", "Jonas"])["norm"] == "Jonas|Theo")
check("case-insensitive names", pv(["theo"])["norm"] == "Theo")
check("UNKNOWN -> invalid unknown", pv(["UNKNOWN"])["fail"] == "unknown")
check("self -> invalid self_address", pv(["Mara"])["fail"] == "self_address")
check("illegal name invalidates the vote", pv(["Theo", "Bob"])["fail"] == "illegal_name")
check("empty -> missing_target", pv([])["fail"] == "missing_target")
check("garbage -> unparsable", parse_vote("<think>x</think> nope", SPK, "Mara")["fail"] == "unparsable")
check("think block stripped", parse_vote('<think>{"a":1}</think>{"addressees":["Theo"]}', SPK, "Mara")["norm"] == "Theo")

# --- mapping table (spec §1.1) ---------------------------------------------------------------
check("ALL -> to_everyone", class_for(ALL, "Mara", "Theo") == "to_everyone")
check("agent alone -> to_agent_itself", class_for("Theo", "Mara", "Theo") == "to_agent_itself")
check("agent + human -> to_agent_itself", class_for("Priya|Theo", "Mara", "Theo") == "to_agent_itself")
check("other human -> not_to_agent", class_for("Priya", "Mara", "Theo") == "not_to_agent")
check("all humans except agent -> not_to_agent", class_for("Jonas|Priya", "Mara", "Theo") == "not_to_agent")
check("agent speaking -> self", class_for("Priya", "Theo", "Theo") == "self")

# --- aggregation / masking ------------------------------------------------------------------
g = aggregate(["Theo", "Theo", ALL, None, None], "Mara", "Theo")
check("3 valid of 5 -> soft over 3", g["mask"] == 1 and g["valid"] == 3 and abs(g["soft"]["to_agent_itself"] - 2 / 3) < 1e-9, g)
g = aggregate(["Theo", ALL, None, None, None], "Mara", "Theo")
check("2 valid -> masked, no soft", g["mask"] == 0 and g["soft"] is None, g)
g = aggregate(["Theo", "Theo", ALL, "Priya"], "Mara", "Theo")
check("4 valid -> quarter steps", g["soft"] == {"to_agent_itself": 0.5, "to_everyone": 0.25, "not_to_agent": 0.25}, g)
g = aggregate(["Theo"] * 5, "Mara", "Mara")
check("speaker is agent -> self", g["hard"] == "self")

# --- frame -> span assignment ---------------------------------------------------------------
check("largest coverage wins (b)", assign_span(0, [(0.0, 0.03, "a"), (0.03, 0.2, "b")]) == "b")
check("exact tie -> earlier span", assign_span(0, [(0.0, 0.04, "a"), (0.04, 0.2, "b")]) == "a")
check("< 50% covered -> None", assign_span(0, [(0.05, 0.2, "a")]) is None)

# --- frame precedence -----------------------------------------------------------------------
seats = ["A", "B", "C", "D"]
T = 6
act = np.zeros((4, T), bool)
act[0, 0] = True                 # frame 0: agent A alone -> self
# frame 1: silence -> none
act[1, 2] = True                 # frame 2: B inside span, clause ok
act[1, 3] = True                 # frame 3: B outside any span -> masked no_span
act[1, 4] = act[2, 4] = True     # frame 4: overlap; C starts later (incoming) but has no span here
act[2, 5] = True                 # frame 5: C inside a masked clause
spans = {"A": [], "B": [(2 * FRAME, 3 * FRAME, "b0")], "C": [(5 * FRAME, 6 * FRAME, "c0")], "D": []}
lab = {"b0": {"mask": 1, "soft": {"to_agent_itself": 0.6, "to_everyone": 0.4, "not_to_agent": 0.0}},
       "c0": {"mask": 0, "soft": None}}
soft, mask, reason = build_view(act, seats, "A", spans, lab, T)
check("agent alone -> self", reason[0] == "self" and soft[0, CLASS_IDX["self"]] == 1)
check("silence -> none", reason[1] == "none" and soft[1, CLASS_IDX["none"]] == 1 and mask[1] == 1)
check("speech in span -> soft label", reason[2] == "speech" and abs(soft[2, CLASS_IDX["to_agent_itself"]] - 0.6) < 1e-6)
check("speech without span -> masked, not none", mask[3] == 0 and reason[3] == "no_span" and soft[3].sum() == 0)
check("overlap follows incoming speaker (no span -> masked)", mask[4] == 0 and reason[4] == "overlap_no_span", reason[4])

check("masked clause stays masked on the timeline", mask[5] == 0 and reason[5] == "masked_clause")

# --- overlap: incoming speaker rule (user, 2026-10-12) ---------------------------------------
T2 = 5
act = np.zeros((4, T2), bool)
act[1, 0:3] = True               # B talks frames 0-2 (outgoing)
act[2, 2:5] = True               # C starts at frame 2 -> frame 2 is overlap, C incoming
spans2 = {"A": [], "B": [(0.0, 3 * FRAME, "b0")], "C": [(2 * FRAME, 5 * FRAME, "c0")], "D": []}
lab2 = {"b0": {"mask": 1, "soft": {"to_agent_itself": 0.0, "to_everyone": 1.0, "not_to_agent": 0.0}},
       "c0": {"mask": 1, "soft": {"to_agent_itself": 1.0, "to_everyone": 0.0, "not_to_agent": 0.0}}}
s2, m2, r2 = build_view(act, seats, "A", spans2, lab2, T2)
check("overlap frame takes the incoming speaker's clause label", r2[2] == "overlap_new" and m2[2] == 1
      and s2[2, CLASS_IDX["to_agent_itself"]] == 1, (r2[2], s2[2]))
check("before the overlap the outgoing speaker's label holds", s2[1, CLASS_IDX["to_everyone"]] == 1)
s2, m2, r2 = build_view(act, seats, "C", spans2, lab2, T2)
check("incoming speaker is the agent -> self", r2[2] == "overlap_self" and s2[2, CLASS_IDX["self"]] == 1)
act2 = np.zeros((4, 2), bool)
act2[1, 0:2] = act2[2, 0:2] = True  # B and C start in the same frame
s2, m2, r2 = build_view(act2, seats, "A", {"A": [], "B": [], "C": [], "D": []}, {}, 2)
check("simultaneous onset -> masked overlap_tie", m2[0] == 0 and r2[0] == "overlap_tie")

# --- alignment: unmatched clause -> low quality fallback ------------------------------------
utts = [{"turn_id": 0, "text": "Theo, can you book it? Thanks a lot.", "start": 0.0, "end": 3.0,
         "clauses": [(0, 23), (23, 36)]}]
words = [(0.1, 0.3, "Theo,"), (0.4, 0.5, "can"), (0.5, 0.6, "you"), (0.6, 0.8, "book"), (0.8, 0.9, "it?"),
         (1.5, 1.7, "Thanks"), (1.7, 1.8, "a"), (1.8, 2.0, "lot.")]
sp = clause_spans(utts, words)
check("aligned clauses are contiguous inside the utterance", sp[0]["t1"] == sp[1]["t0"] and not sp[0]["low_quality"], sp)
check("utterance edges from first/last word", sp[0]["t0"] == 0.1 and sp[1]["t1"] == 2.0)
sp = clause_spans(utts, words[:5] + [(1.5, 1.7, "zzz")])
check("unaligned clause -> whole utterance proportional fallback, flagged", all(c["low_quality"] for c in sp), sp)

print(f"\n{_fail} failure(s)")
sys.exit(1 if _fail else 0)
