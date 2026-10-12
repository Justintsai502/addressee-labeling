#!/usr/bin/env python3
"""ASR spot-check of ttv2 audio: what did TTS actually say for utterances whose script text has an
unreplaced placeholder ("[A]", "[Jerome]")? (GPU, NeMo Parakeet; run with the NeMo venv python.)

For each dialogue: transcribe the mix wav with word timestamps, then print every placeholder
utterance next to the ASR words that fall inside its [start, end] window.

    python scripts/09_asr_check_ttv2.py --root /work/b12901015/mpmoshi/data/ttv2/turn_taking_2000_v2 \
        --ids tr5675 tr2181 tr5609 --out outputs/agent_view_v3/ttv2_placeholder_asr.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import tempfile
from pathlib import Path

import librosa
import soundfile as sf

MODEL = "/work/b12901015/datasets/hf/hub/models--nvidia--parakeet-tdt-0.6b-v2/snapshots"
PH = re.compile(r"\[[A-Za-z]+\]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(sorted(glob.glob(f"{MODEL}/*/parakeet-tdt-0.6b-v2.nemo"))[-1]).cuda().eval()
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        for cid in args.ids:
            d = Path(args.root) / cid
            audio, _ = librosa.load(str(d / "audio_qwen3-tts-1.7b-clone.wav"), sr=16000, mono=True)
            wav = Path(tmp) / "a.wav"
            sf.write(wav, audio, 16000)
            hyp = model.transcribe([str(wav)], timestamps=True, verbose=False)[0]
            words = [(float(w["start"]), float(w["end"]), str(w.get("word", ""))) for w in (hyp.timestamp or {}).get("word", [])]
            for l in (d / "script.jsonl").read_text(encoding="utf-8").splitlines():
                u = json.loads(l)
                if not PH.search(u["text"]):
                    continue
                heard = " ".join(w for s, e, w in words if s >= u["start"] - 0.3 and e <= u["end"] + 0.3)
                report.append({"dialogue": cid, "utt_id": u["utt_id"], "speaker": u["speaker"], "script": u["text"],
                               "asr": heard, "addressing": u["addressing"]})
                print(f"{cid} {u['utt_id']}\n  script: {u['text']}\n  asr:    {heard}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
