#!/usr/bin/env python3
"""ADS v3 frame labels (12.5 Hz) for ttv2 dialogues from v5 clause votes + per-seat word timestamps (CPU).

Inputs: conversations (00c_prepare_ttv2.py output), clause votes (11_clause_label.py, causal), ASR words
(12_align_ttv2.py). Run with a python that has numpy + soundfile (e.g. the NeMo venv).

    python scripts/13_build_frames_v5.py --convs data/ttv2/convs_pilot.jsonl \
        --votes outputs/ads_v3_v5/dev20_causal/clause_votes.jsonl --align-dir outputs/ads_v3_v5/align \
        --root /work/b12901015/mpmoshi/data/ttv2/turn_taking_2000_v2 --out-dir outputs/ads_v3_v5/dev20_frames

Writes <out>/frames/<cid>.npz (per agent seat X: soft_X [T,5], mask_X [T], reason_X [T] codes; activity [4,T]),
<out>/clauses.jsonl (spans, alignment flags, per-agent labels, cue diagnostics), <out>/summary.json.
Classes: none, self, to_agent_itself, to_everyone, not_to_agent. human_audited=false.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import _bootstrap  # noqa: F401

from src.clause_v5 import aggregate, self_vocative
from src.frames_v5 import (FRAME, build_view, clause_spans, frames_from_sub, norm_tokens, vad_subframes)
from src.io_utils import load_conversations

STAGE = re.compile(r"\[to\s[^\]]*\]", re.I)
REASONS = ("none", "self", "speech", "speech_lq", "overlap_tie", "no_span", "masked_clause",
           "overlap_new", "overlap_self", "overlap_no_span", "overlap_masked_clause")


def cue_diag(cue, clause_text, char_start, word_times):
    """Where a vote's cue sits: none_needed / not_in_clause / in clause (+ estimated_cue_end if its words aligned)."""
    if cue is None:
        return {"kind": "missing"}
    if cue.strip().lower() in ("none_needed", "none needed"):
        return {"kind": "none_needed"}
    ct = [t for t, _, _ in norm_tokens(cue)]
    toks = norm_tokens(clause_text)
    for i in range(len(toks) - len(ct) + 1):
        if ct and [t for t, _, _ in toks[i:i + len(ct)]] == ct:
            last = toks[i + len(ct) - 1]
            rel = (last[2]) / max(1, len(clause_text))
            starts = [char_start + a for _, a, _ in toks[i:i + len(ct)]]
            t = [word_times.get(s) for s in starts]
            return {"kind": "in_clause", "rel_char_end": round(rel, 3),
                    "estimated_cue_end": (round(t[-1][1], 3) if all(t) else None)}
    return {"kind": "not_in_clause"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", required=True)
    ap.add_argument("--votes", required=True)
    ap.add_argument("--align-dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--audio-pattern", default="audio_qwen3-tts-1.7b-clone_{seat}.wav")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    votes = defaultdict(list)
    for l in open(args.votes, encoding="utf-8"):
        r = json.loads(l)
        if not r["conversation_id"].startswith("fixed:"):
            votes[r["conversation_id"]].append(r)
    summ = {"frames": Counter(), "frames_by_class": defaultdict(Counter), "clauses": 0,
            "clauses_low_quality": 0, "clauses_masked": 0, "clauses_stage_direction": 0, "clauses_self_vocative": 0, "dialogues": 0, "cue": Counter(),
            "script_overlap_pairs": 0}
    fc = open(out / "clauses.jsonl", "w", encoding="utf-8")
    for conv in load_conversations(args.convs):
        cid = conv.conversation_id
        if cid not in votes:
            continue
        d = Path(args.root) / cid
        names = conv.meta["names"]                       # seat -> display name
        seats = sorted(names)
        src_folder = json.loads((d / "meta.json").read_text())["setup"]["source_folder"]
        script = [json.loads(l) for l in (d / "script.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        align = json.loads((Path(args.align_dir) / f"{cid}.json").read_text())
        rows = sorted(votes[cid], key=lambda r: (r["turn_id"], r["clause_idx"]))
        by_turn = defaultdict(list)
        for r in rows:
            by_turn[r["turn_id"]].append(r)
        # activity layer
        acts, dur = [], 0.0
        subs = []
        for s in seats:
            x, sr = sf.read(str(d / args.audio_pattern.format(seat=s)))
            if x.ndim > 1:
                x = x.mean(1)
            dur = max(dur, len(x) / sr)
            subs.append(vad_subframes(x, sr))
        n_frames = math.ceil(dur / FRAME)
        active = np.stack([frames_from_sub(a, n_frames) for a in subs])
        # span layer
        spans_by_seat, cinfo = {}, {}
        for s in seats:
            utts = []
            for t in conv.turns:
                if script[t.turn_id]["speaker"] != s:
                    continue
                cl = by_turn[t.turn_id]
                utts.append({"turn_id": t.turn_id, "text": t.text, "start": t.start, "end": t.end,
                             "clauses": [(c["char_start"], c["char_end"]) for c in cl]})
            sp = clause_spans(utts, [tuple(w) for w in align["seats"][s]])
            spans_by_seat[s] = sorted((c["t0"], c["t1"], (c["turn_id"], c["clause_idx"])) for c in sp)
            for c in sp:
                cinfo[(c["turn_id"], c["clause_idx"])] = c
        for a, b in zip(script, script[1:]):
            summ["script_overlap_pairs"] += int(b["start"] < a["end"] - 1e-6)
        npz = {"activity": active.astype(np.uint8)}
        labels_by_agent = {}
        for agent in seats:
            lab = {}
            for r in rows:
                key = (r["turn_id"], r["clause_idx"])
                g = aggregate([v["norm"] for v in r["final"]], r["speaker"], names[agent])
                g["low_quality"] = cinfo[key]["low_quality"]
                # stage directions in the script text ("[to C] ...") leak the addressee and are spoken by the
                # TTS ("To see, ..."); mask those clauses (user, 2026-10-12). The agent's own turn stays `self`.
                if STAGE.search(r["turn_text"]) and g["hard"] != "self":
                    g.update(mask=0, soft=None, hard=None, stage_direction=True)
                # the speaker calls their own name: the addressee is unknowable from the audio; mask (2026-10-12)
                if self_vocative(r["turn_text"], r["speaker"]) and g["hard"] != "self":
                    g.update(mask=0, soft=None, hard=None, self_vocative=True)
                lab[key] = g
            labels_by_agent[agent] = lab
            # spans are indexed by speaking seat; build_view looks up the active seat's spans
            soft, mask, reason = build_view(active, list(seats), agent, spans_by_seat, lab, n_frames)
            npz[f"soft_{agent}"], npz[f"mask_{agent}"] = soft, mask
            npz[f"reason_{agent}"] = np.array([REASONS.index(x) for x in reason], np.uint8)
            summ["frames"].update(f"{agent}:{x}" for x in reason)
            hard = np.where(mask == 1, soft.argmax(1), -1)
            summ["frames_by_class"][agent].update(int(h) for h in hard)
        np.savez_compressed(out / "frames" / f"{cid}.npz", **npz)
        for r in rows:
            key = (r["turn_id"], r["clause_idx"])
            ci = cinfo[key]
            cues = [cue_diag(v["cue"], r["clause"], r["char_start"], ci["word_times"]) if v["norm"] else None
                    for v in r["final"]]
            summ["cue"].update(c["kind"] for c in cues if c)
            summ["clauses"] += 1
            summ["clauses_low_quality"] += int(ci["low_quality"])
            summ["clauses_masked"] += int(r["valid_vote_count"] < 3)
            summ["clauses_stage_direction"] += int(bool(STAGE.search(r["turn_text"])))
            summ["clauses_self_vocative"] += int(self_vocative(r["turn_text"], r["speaker"]))
            fc.write(json.dumps({
                "conversation_id": cid, "turn_id": r["turn_id"], "clause_idx": r["clause_idx"],
                "seat": script[r["turn_id"]]["speaker"], "speaker": r["speaker"],
                "function": script[r["turn_id"]].get("function"), "source_folder": src_folder, "clause": r["clause"],
                "t0": ci["t0"], "t1": ci["t1"], "low_quality_alignment": ci["low_quality"],
                "matched_frac": ci["matched_frac"], "valid_vote_count": r["valid_vote_count"],
                "majority": r["majority"], "majority_count": r["majority_count"],
                "agent_view": {a: {k: labels_by_agent[a][key][k] for k in ("hard", "soft", "mask")} for a in seats},
                "cues": cues, "human_audited": False}, ensure_ascii=False) + "\n")
        summ["dialogues"] += 1
        print(cid, n_frames, "frames", flush=True)
    fc.close()
    meta_src = {}
    for cid in votes:
        p = Path(args.root) / cid / "meta.json"
        if p.exists():
            meta_src[cid] = json.loads(p.read_text())["setup"]["source_folder"]
    summ["source_folder"] = meta_src
    summ = {k: (dict(v) if isinstance(v, Counter) else ({a: dict(c) for a, c in v.items()} if k == "frames_by_class" else v))
            for k, v in summ.items()}
    summ["class_index"] = ["none", "self", "to_agent_itself", "to_everyone", "not_to_agent", "masked=-1"]
    summ["human_audited"] = False
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: summ[k] for k in ("dialogues", "clauses", "clauses_low_quality", "clauses_masked")}))


if __name__ == "__main__":
    main()
