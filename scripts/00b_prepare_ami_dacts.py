#!/usr/bin/env python3
"""AMI dialogue acts -> conversations.jsonl (one Turn per dialogue act) + HUMAN addressee labels.

AMI's manual dialogue-act layer carries a human `addressee` attribute on a subset of
meetings (22 meetings, ~8.9k dialogue acts in ami_public_manual_1.6.2), e.g.

    <dact nite:id="IS1008a.A.dialog-act.s9553330.1" addressee="D,C,B">
       <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_14)"/>
       <nite:child href="IS1008a.A.words.xml#id(IS1008a.A.words0)..id(IS1008a.A.words1)"/>

Unlike 00_prepare_ami.py (one Turn per transcription segment), the unit here is the
dialogue act, because that is where the human addressee lives. Outputs:

  --out-convs   conversations.jsonl  (repo schema; extra per-turn keys dact_id/da_type are
                ignored by Turn.from_dict but kept for analysis)
  --out-human   labels jsonl in the repo label format, labeler="AMI-human".
                Dialogue acts WITHOUT an addressee attribute are NOT written: in AMI these
                are mostly backchannels / fragments / stalls that were not annotated, so
                they are "unlabeled", not UNKNOWN.

Reads straight from the zip; pure stdlib.

    python3 scripts/00b_prepare_ami_dacts.py --meetings IS1008a TS3005a \
        --out-convs data/ami/dacts_pilot.jsonl --out-human outputs/agent_view_v3/human_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NITE = "{http://nite.sourceforge.net/}"
ID_RE = re.compile(r"id\(([^)]+)\)")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def da_types(z: zipfile.ZipFile) -> dict:
    root = ET.fromstring(z.read("ontologies/da-types.xml"))
    out = {}
    for el in root.iter():
        i = el.get(NITE + "id")
        if i and el.get("name"):
            out[i] = el.get("name")
    return out


def words(z: zipfile.ZipFile, meeting: str, spk: str):
    root = ET.fromstring(z.read(f"words/{meeting}.{spk}.words.xml"))
    ordered, info = [], {}
    for el in list(root):
        wid = el.get(NITE + "id")
        if wid is None:
            continue
        info[wid] = {"tag": el.tag.split("}")[-1], "text": (el.text or "").strip(),
                     "punc": el.get("punc") == "true", "start": _f(el.get("starttime")), "end": _f(el.get("endtime"))}
        ordered.append(wid)
    return ordered, info


def text_of(span, info) -> str:
    parts = []
    for wid in span:
        w = info.get(wid)
        if not w or w["tag"] != "w" or not w["text"]:
            continue
        if w["punc"] and parts:
            parts[-1] += w["text"]
        else:
            parts.append(w["text"])
    return " ".join(parts)


def build(z: zipfile.ZipFile, meeting: str, types: dict):
    names = set(z.namelist())
    speakers = sorted(s for s in "ABCDE" if f"dialogueActs/{meeting}.{s}.dialog-act.xml" in names)
    turns, human = [], {}
    for spk in speakers:
        ordered, info = words(z, meeting, spk)
        idx = {w: i for i, w in enumerate(ordered)}
        root = ET.fromstring(z.read(f"dialogueActs/{meeting}.{spk}.dialog-act.xml"))
        for d in root.iter("dact"):
            child = d.find(NITE + "child")
            if child is None:
                continue
            ids = ID_RE.findall(child.get("href", "").split("#", 1)[-1])
            if not ids or ids[0] not in idx or ids[-1] not in idx:
                continue
            i0, i1 = sorted((idx[ids[0]], idx[ids[-1]]))
            span = ordered[i0:i1 + 1]
            text = text_of(span, info)
            st = [info[w]["start"] for w in span if info[w]["start"] is not None]
            en = [info[w]["end"] for w in span if info[w]["end"] is not None]
            if not text or not st or not en:
                continue
            ptr = d.find(NITE + "pointer")
            da = ID_RE.findall(ptr.get("href", ""))[0] if ptr is not None and ID_RE.findall(ptr.get("href", "")) else None
            turns.append({"speaker": spk, "start": min(st), "end": max(en), "text": text,
                          "dact_id": d.get(NITE + "id"), "da_type": types.get(da, "unknown"),
                          "_addr": d.get("addressee")})
    turns.sort(key=lambda t: (t["start"], t["speaker"]))
    for i, t in enumerate(turns):
        t["turn_id"] = i
        t["overlap"] = any(o is not t and o["speaker"] != t["speaker"] and o["start"] < t["end"] and o["end"] > t["start"]
                           for o in turns[max(0, i - 30): i + 30])
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)
        addr = t.pop("_addr")
        if addr:
            human[i] = sorted(a.strip() for a in addr.split(",") if a.strip())
    conv = {"conversation_id": meeting, "audio_path": None, "speakers": speakers,
            "meta": {"source": "AMI", "meeting": meeting, "unit": "dialogue_act"}, "turns": turns}
    return conv, human


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/ami/ami_manual.zip")
    ap.add_argument("--meetings", nargs="+", required=True)
    ap.add_argument("--out-convs", required=True)
    ap.add_argument("--out-human", required=True)
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    types = da_types(z)
    for p in (args.out_convs, args.out_human):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_convs, "w", encoding="utf-8") as fc, open(args.out_human, "w", encoding="utf-8") as fh:
        for m in args.meetings:
            conv, human = build(z, m, types)
            fc.write(json.dumps(conv, ensure_ascii=False) + "\n")
            labels = [{"turn_id": tid, "addressees": a, "confidence": None, "rationale": "AMI manual addressee",
                       "labeler": "AMI-human"} for tid, a in sorted(human.items())]
            fh.write(json.dumps({"conversation_id": m, "labels": labels}, ensure_ascii=False) + "\n")
            print(f"{m}: {len(conv['speakers'])} speakers, {len(conv['turns'])} dialogue acts, "
                  f"{len(human)} with human addressee", file=sys.stderr)


if __name__ == "__main__":
    main()
