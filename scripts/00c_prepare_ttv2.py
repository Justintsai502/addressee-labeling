#!/usr/bin/env python3
"""Senior's 4-speaker turn-taking scripts (Audio_synthesize feature/ttv2-prod, turn_taking_2000_v2)
-> conversations.jsonl + the GENERATOR's addressing as a reference label file.

Speakers are rendered by their display names (Glen, Marc, ...), because vocatives in the text
use those names; slot letters (A-D) are kept in meta. Generator `["EVERYONE"]` -> GROUP.
Text is kept verbatim, including unreplaced placeholders such as "[A]".

Per-turn flags (for stratified analysis): x_named (addressee [X] and X's name in text),
x_unnamed_next (addressee [X], name absent, X speaks next), group_cue_x ([X] but the text says
guys/everyone/anyone...), placeholder (text contains "[...]").

Selection (default, = the EXP-060 pilot): dialogues with group_cue_x, dialogues with placeholders, then a seeded
random fill. Or pass --ids / --ids-file / --all to convert exactly those dialogues (ADS v3 pipeline).

    python3 scripts/00c_prepare_ttv2.py --root /work/b12901015/mpmoshi/data/ttv2/turn_taking_2000_v2 \
        --n 20 --out-convs data/ttv2/convs_pilot.jsonl --out-ref outputs/agent_view_v3/ttv2_ref_pilot.jsonl
    python3 scripts/00c_prepare_ttv2.py --root <ttv2 root> --all --out-convs data/ttv2/convs_all.jsonl \
        --out-ref outputs/ttv2_ref_all.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

GROUP_CUE = re.compile(r"\b(guys|everyone|everybody|all of you|you all|y'all|folks|anyone|anybody)\b", re.I)
PLACEHOLDER = re.compile(r"\[[A-Za-z]+\]")


def load(d: Path):
    u = [json.loads(l) for l in (d / "script.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    names = u[0].get("display_names") or json.loads((d / "names.json").read_text())["canonical"]
    return u, names


def flags(u, i, names):
    x = u[i]
    a = x["addressing"]
    t = x["text"]
    out = {"placeholder": bool(PLACEHOLDER.search(t)), "x_named": False, "x_unnamed_next": False, "group_cue_x": False}
    if a != ["EVERYONE"]:
        X = a[0]
        named = bool(re.search(r"\b" + re.escape(names.get(X, "#")) + r"\b", t))
        nxt = u[i + 1]["speaker"] if i + 1 < len(u) else None
        out["x_named"] = named
        out["x_unnamed_next"] = (not named) and nxt == X
        out["group_cue_x"] = (not named) and bool(GROUP_CUE.search(t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out-convs", required=True)
    ap.add_argument("--out-ref", required=True)
    ap.add_argument("--ids", nargs="*", default=None, help="convert exactly these dialogue ids")
    ap.add_argument("--ids-file", default=None, help="file with dialogue ids (whitespace separated)")
    ap.add_argument("--all", action="store_true", help="convert every dialogue under --root")
    ap.add_argument("--audio-name", default="audio_qwen3-tts-1.7b-clone.wav", help="mix wav file name (meta only)")
    args = ap.parse_args()
    dirs = sorted(Path(p).parent for p in glob.glob(f"{args.root}/tr*/script.jsonl"))
    info = {}
    for d in dirs:
        u, names = load(d)
        f = [flags(u, i, names) for i in range(len(u))]
        info[d.name] = {"gc": any(x["group_cue_x"] for x in f), "ph": any(x["placeholder"] for x in f)}
    explicit = None
    if args.all:
        explicit = sorted(info)
    elif args.ids or args.ids_file:
        explicit = list(args.ids or []) + (Path(args.ids_file).read_text().split() if args.ids_file else [])
        missing = [i for i in explicit if i not in info]
        if missing:
            raise SystemExit(f"unknown dialogue ids: {missing[:10]}")
    rng = random.Random(args.seed)
    gc = sorted(k for k, v in info.items() if v["gc"])
    ph = sorted(k for k, v in info.items() if v["ph"] and not v["gc"])
    rng.shuffle(gc); rng.shuffle(ph)
    chosen = gc[: args.n // 3] + ph[: args.n // 3]
    rest = sorted(k for k in info if k not in chosen)
    rng.shuffle(rest)
    chosen += rest[: args.n - len(chosen)]
    if explicit is not None:
        chosen = explicit
    Path(args.out_convs).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_ref).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_convs, "w", encoding="utf-8") as fc, open(args.out_ref, "w", encoding="utf-8") as fr:
        for cid in sorted(chosen):
            d = Path(args.root) / cid
            u, names = load(d)
            turns, labels = [], []
            for i, x in enumerate(u):
                spk = names[x["speaker"]]
                turns.append({"turn_id": i, "speaker": spk, "start": x["start"], "end": x["end"], "text": x["text"],
                              "overlap": False, "slot": x["speaker"], "utt_id": x["utt_id"], **flags(u, i, names)})
                a = x["addressing"]
                labels.append({"turn_id": i, "addressees": ["GROUP"] if a == ["EVERYONE"] else [names[a[0]]],
                               "confidence": None, "rationale": "generator addressing (repair_dialogue applied)",
                               "labeler": "ttv2-generator"})
            setup = json.loads((d / "meta.json").read_text()).get("setup", {}) if (d / "meta.json").exists() else {}
            conv = {"conversation_id": cid, "audio_path": str(d / args.audio_name),
                    "speakers": [names[s] for s in sorted(names)],
                    "meta": {"source": "ttv2", "names": names, "source_folder": setup.get("source_folder")},
                    "turns": turns}
            fc.write(json.dumps(conv, ensure_ascii=False) + "\n")
            fr.write(json.dumps({"conversation_id": cid, "labels": labels}, ensure_ascii=False) + "\n")
    if explicit is not None:
        print(f"{len(chosen)} dialogues converted")
    else:
        print(f"{len(chosen)} dialogues: {len(gc[:args.n//3])} with group-cue [X], {len(ph[:args.n//3])} with placeholders, rest random")


if __name__ == "__main__":
    main()
