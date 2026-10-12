#!/usr/bin/env python3
"""Step 3 of the ADS v3 pipeline: word timestamps for ttv2 per-seat tracks. GPU, NeMo Parakeet (NeMo env python).

Each seat's whole track (default `audio_qwen3-tts-1.7b-clone_{seat}.wav`) is transcribed once with word timestamps;
script <-> ASR word matching happens later on CPU (13_build_frames_v5.py). Script `start`/`end` are not used
here: they differ from the audio by up to ~1.5 s at utterance ends (CPU check 2026-10-11).

    python scripts/12_align_ttv2.py --root <ttv2 root> --convs data/ttv2/convs_pilot.jsonl \
        --out-dir outputs/ads_v3_v5/align [--asr-model <path to parakeet-tdt-0.6b-v2.nemo or HF id>]
Dialogues already in --out-dir are skipped (resume).
"""
from __future__ import annotations

import argparse
import glob
import json
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def load_model(spec):
    import nemo.collections.asr as nemo_asr

    if spec.endswith(".nemo"):
        return nemo_asr.models.ASRModel.restore_from(spec)
    hits = sorted(glob.glob(f"{spec}/**/*.nemo", recursive=True)) if Path(spec).is_dir() else []
    if hits:
        return nemo_asr.models.ASRModel.restore_from(hits[-1])
    return nemo_asr.models.ASRModel.from_pretrained(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--convs", default=None, help="take dialogue ids (and seats) from this conversations JSONL")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--asr-model", default=DEFAULT_MODEL, help=".nemo file, a directory containing one, or an HF id")
    ap.add_argument("--audio-pattern", default="audio_qwen3-tts-1.7b-clone_{seat}.wav")
    args = ap.parse_args()
    todo = []  # (cid, seats)
    if args.convs:
        for l in open(args.convs, encoding="utf-8"):
            if l.strip():
                c = json.loads(l)
                todo.append((c["conversation_id"], sorted(c["meta"]["names"])))
    todo += [(i, list("ABCD")) for i in (args.ids or [])]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [(c, s) for c, s in todo if not (out / f"{c}.json").exists()]
    print(f"{len(todo)} dialogues to align", flush=True)
    if not todo:
        return
    model = load_model(args.asr_model).cuda().eval()
    ckpt = args.asr_model
    with tempfile.TemporaryDirectory() as tmp:
        for cid, seats in todo:
            d = Path(args.root) / cid
            paths = []
            for s in seats:
                audio, _ = librosa.load(str(d / args.audio_pattern.format(seat=s)), sr=16000, mono=True)
                p = Path(tmp) / f"{cid}_{s}.wav"
                sf.write(p, audio, 16000)
                paths.append(str(p))
            hyps = model.transcribe(paths, timestamps=True, verbose=False, batch_size=4)
            res = {"conversation_id": cid, "asr_model": ckpt, "seats": {}}
            for s, h in zip(seats, hyps):
                res["seats"][s] = [[round(float(w["start"]), 3), round(float(w["end"]), 3), str(w.get("word", ""))]
                                   for w in (h.timestamp or {}).get("word", [])]
            (out / f"{cid}.json").write_text(json.dumps(res, ensure_ascii=False) + "\n", encoding="utf-8")
            print(cid, {s: len(v) for s, v in res["seats"].items()}, flush=True)


if __name__ == "__main__":
    main()
