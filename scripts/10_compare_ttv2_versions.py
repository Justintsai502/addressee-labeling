#!/usr/bin/env python3
"""Compare labeler versions on the ttv2 pilot against the generator labels and Claude's 25 readings.

    python3 scripts/10_compare_ttv2_versions.py v3=outputs/agent_view_v3/ttv2_qwen3_32b \
        v4=outputs/agent_view_v3/ttv2_qwen3_32b_v4 v4causal=outputs/agent_view_v3/ttv2_qwen3_32b_v4_causal
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from src.agent_view import is_group
from src.io_utils import load_conversations, load_labels

CONVS = "data/ttv2/convs_pilot.jsonl"
REF = "outputs/agent_view_v3/ttv2_ref_pilot.jsonl"
READ25 = "outputs/agent_view_v3/ttv2_claude_read_25.json"


def norm(s, sp, spk):
    s = sorted(set(s))
    return ("GROUP",) if is_group(s, sp, spk) else tuple(s)


def main() -> None:
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    convs = {c.conversation_id: c for c in load_conversations(CONVS)}
    raw = {json.loads(l)["conversation_id"]: {t["turn_id"]: t for t in json.loads(l)["turns"]} for l in open(CONVS)}
    ref = load_labels(REF)
    votes = {v: {json.loads(l)["conversation_id"]: {t["turn_id"]: t["samples"] for t in json.loads(l)["turns"]}
                 for l in open(Path(p) / "votes.jsonl")} for v, p in runs.items()}
    top = {}
    for v, vv in votes.items():
        for cid, labs in ref.items():
            c = convs[cid]
            for t in c.turns:
                cnt = Counter(norm(s, t.speaker, c.speakers) for s in vv[cid][t.turn_id])
                top[(v, cid, t.turn_id)] = cnt.most_common(1)[0]

    def cat(r, g):
        if r["group_cue_x"]:
            return "[X] on guys/everyone line"
        if r["x_unnamed_next"]:
            return "[X] unnamed, X speaks next"
        if r["x_named"]:
            return "[X] named in text"
        return "[X] other" if g != ("GROUP",) else "EVERYONE"

    stats = defaultdict(Counter)
    for cid, labs in ref.items():
        c = convs[cid]
        ts = c.turns
        for i, t in enumerate(ts):
            g = norm(labs[t.turn_id].addressees, t.speaker, c.speakers)
            k = cat(raw[cid][t.turn_id], g)
            prev = (ts[i - 1].speaker,) if i else None
            for v in runs:
                tp, n = top[(v, cid, t.turn_id)]
                for key in (k, "ALL"):
                    s = stats[(key, v)]
                    s["n"] += 1
                    s["agree_generator"] += tp == g
                    s["pred_GROUP"] += tp == ("GROUP",)
                    s["pred_previous_speaker"] += tp == prev
                    s["pred_UNKNOWN"] += tp == ("UNKNOWN",)
                    s["unanimous_5of5"] += n == 5
                    s["split_le_3of5"] += n <= 3
    print("fractions per category (top vote of 5):")
    for key in ["[X] named in text", "[X] unnamed, X speaks next", "[X] on guys/everyone line", "EVERYONE", "ALL"]:
        for v in runs:
            s = stats[(key, v)]
            if not s["n"]:
                continue
            print(f"  {key:28s} {v:9s} n={s['n']:3d} " + " ".join(f"{m}={s[m] / s['n']:.2f}" for m in s if m != "n"))
    # pairwise disagreement between versions
    names = list(runs)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            d = sum(top[(names[a], cid, t.turn_id)][0] != top[(names[b], cid, t.turn_id)][0]
                    for cid in ref for t in convs[cid].turns)
            tot = sum(len(convs[cid].turns) for cid in ref)
            print(f"top-vote differs {names[a]} vs {names[b]}: {d}/{tot} ({d / tot:.1%})")
    # Claude's 25 readings
    items = json.load(open(READ25))["items"]
    for v in runs:
        c25 = Counter()
        for it in items:
            c = convs[it["conversation_id"]]
            tp = top[(v, it["conversation_id"], it["turn_id"])][0]
            lab = "prev" if tp == (it["previous_speaker"],) else "group" if tp == ("GROUP",) else "other"
            ok = it["claude_reading"] == "either" and lab in ("prev", "group") or it["claude_reading"] == lab
            c25["consistent"] += ok
            c25[f"{it['claude_reading']}->{lab}"] += 1
        print(f"Claude-25 {v}: consistent {c25['consistent']}/25  {dict((k, x) for k, x in c25.items() if k != 'consistent')}")


if __name__ == "__main__":
    main()
