#!/usr/bin/env python3
"""Convert AMI manual annotations (NXT XML) into our conversations.jsonl schema.

AMI layout (from ami_public_manual_1.6.2.zip):
  words/{MEETING}.{SPK}.words.xml
      <w nite:id="ES2002a.A.words0" starttime="77.44" endtime="77.74">Hi</w>
      (punctuation carries punc="true"; non-lexical events are other tags)
  segments/{MEETING}.{SPK}.segments.xml
      <segment transcriber_start="77.4" transcriber_end="80.9">
        <nite:child href="ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)"/>
      </segment>

Each <segment> becomes one Turn. SPK is the AMI agent letter (A/B/C/D). A turn is
flagged overlap=True if it overlaps any other speaker's turn in time.

    python3 scripts/00_prepare_ami.py \
        --annotations-dir data/ami/annotations \
        --audio-dir data/ami/audio \
        --meetings ES2002a ES2002b \
        --out data/ami/conversations.jsonl

Pure stdlib; no LLM, no heavy deps. Safe to run on the laptop.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

NITE = "{http://nite.sourceforge.net/}"
ID_RE = re.compile(r"id\(([^)]+)\)")


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_words(path: Path):
    """Return (ordered_ids, id->info) where info = {tag,text,punc,start,end}."""
    root = ET.parse(path).getroot()
    ordered, info = [], {}
    for el in list(root):
        wid = el.get(NITE + "id")
        if wid is None:
            continue
        tag = el.tag.split("}")[-1]
        info[wid] = {
            "tag": tag,
            "text": (el.text or ""),
            "punc": el.get("punc") == "true",
            "start": _to_float(el.get("starttime")),
            "end": _to_float(el.get("endtime")),
        }
        ordered.append(wid)
    return ordered, info


def reconstruct_text(word_ids, info) -> str:
    parts = []
    for wid in word_ids:
        w = info.get(wid)
        if not w or w["tag"] != "w":
            continue  # skip vocalsound/gap/disfmarker/etc.
        t = w["text"].strip()
        if not t:
            continue
        if w["punc"] and parts:
            parts[-1] = parts[-1] + t  # attach punctuation to previous token
        else:
            parts.append(t)
    return " ".join(parts)


def parse_segments(seg_path: Path, ordered, id_index, info, speaker: str):
    """Yield turns (dicts without turn_id) for one speaker's segments file."""
    root = ET.parse(seg_path).getroot()
    turns = []
    for seg in root.findall("segment"):
        child = seg.find(NITE + "child")
        if child is None:
            continue
        href = child.get("href", "")
        ids = ID_RE.findall(href.split("#", 1)[-1])
        if not ids:
            continue
        first, last = ids[0], ids[-1]
        if first not in id_index or last not in id_index:
            continue
        i0, i1 = id_index[first], id_index[last]
        if i0 > i1:
            i0, i1 = i1, i0
        span = ordered[i0 : i1 + 1]
        text = reconstruct_text(span, info)
        if not text:
            continue  # silence / non-lexical only
        starts = [info[w]["start"] for w in span if info[w]["start"] is not None]
        ends = [info[w]["end"] for w in span if info[w]["end"] is not None]
        start = min(starts) if starts else _to_float(seg.get("transcriber_start"))
        end = max(ends) if ends else _to_float(seg.get("transcriber_end"))
        if start is None or end is None:
            continue
        turns.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return turns


def build_conversation(meeting: str, ann_dir: Path, audio_dir: Path):
    words_dir, seg_dir = ann_dir / "words", ann_dir / "segments"
    speakers = sorted(
        p.name.split(".")[1]
        for p in seg_dir.glob(f"{meeting}.*.segments.xml")
    )
    all_turns = []
    for spk in speakers:
        wpath = words_dir / f"{meeting}.{spk}.words.xml"
        spath = seg_dir / f"{meeting}.{spk}.segments.xml"
        if not (wpath.exists() and spath.exists()):
            continue
        ordered, info = parse_words(wpath)
        id_index = {wid: i for i, wid in enumerate(ordered)}
        all_turns.extend(parse_segments(spath, ordered, id_index, info, spk))

    all_turns.sort(key=lambda t: (t["start"], t["speaker"]))
    # overlap flag: intersects any other speaker's turn in time
    for i, t in enumerate(all_turns):
        t["overlap"] = any(
            o["speaker"] != t["speaker"]
            and o["start"] < t["end"]
            and o["end"] > t["start"]
            for o in all_turns
        )
    for i, t in enumerate(all_turns):
        t["turn_id"] = i
        t["start"] = round(t["start"], 2)
        t["end"] = round(t["end"], 2)

    audio_path = audio_dir / f"{meeting}.Mix-Headset.wav"
    return {
        "conversation_id": meeting,
        "audio_path": str(audio_path),
        "speakers": speakers,
        "meta": {"source": "AMI", "meeting": meeting},
        "turns": all_turns,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-dir", default="data/ami/annotations")
    ap.add_argument("--audio-dir", default="data/ami/audio")
    ap.add_argument("--meetings", nargs="+", required=True)
    ap.add_argument("--out", default="data/ami/conversations.jsonl")
    args = ap.parse_args()

    ann_dir, audio_dir = Path(args.annotations_dir), Path(args.audio_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for m in args.meetings:
            conv = build_conversation(m, ann_dir, audio_dir)
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            n = len(conv["turns"])
            n_ov = sum(t["overlap"] for t in conv["turns"])
            dur = max((t["end"] for t in conv["turns"]), default=0)
            print(f"{m}: {len(conv['speakers'])} speakers, {n} turns, "
                  f"{n_ov} overlap ({100*n_ov//max(n,1)}%), ~{dur/60:.1f} min",
                  file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
