# Data format

Each corpus (AMI, PersonaPlex, ...) is converted to a `.jsonl` where **one line is
one conversation**:

```json
{
  "conversation_id": "ES2008a",
  "audio_path": "data/ami/audio/ES2008a.wav",
  "speakers": ["Alice", "Bob", "Carol", "Dave"],
  "turns": [
    {"turn_id": 0, "speaker": "Alice", "start": 0.0, "end": 3.0,
     "text": "Hey Bob, did you look at the numbers?", "overlap": false}
  ]
}
```

Rules:

- `speaker` ids must be **stable and shared** between the audio and the transcript.
  The models label addressees using exactly these ids, so if diarization renames a
  speaker the labels stop being comparable. Human-readable names (Alice/Bob) are
  fine and help vocative detection; anonymous ids (spk0/spk1) also work.
- `start`/`end` are seconds. `overlap: true` marks a turn that overlaps another
  speaker in time — these are the hard cases the evaluation reports separately.
- `audio_path` is only read by the Gemini golden labeler on the server. The
  offline demo, the evaluation, and the tests never open the audio.

## `sample/conversations.jsonl`

Three tiny synthetic conversations (dyadic / triad / quad) used by `run_demo.py`
and the tests. No audio files are shipped; the sample is transcript-only, which is
all the offline code needs.

## AMI corpus (`data/ami/`, gitignored)

Real multi-party meetings (4 speakers, lots of overlap) — a good stress test for
addressee labeling. The data is large and license-attributed (CC-BY 4.0, © Univ. of
Edinburgh), so it is **gitignored and never committed**; regenerate it with:

```bash
bash scripts/download_ami.sh ES2002a ES2002b
```

This downloads the Mix-Headset audio + manual annotations and runs
`scripts/00_prepare_ami.py`, which parses the AMI NXT XML (per-speaker `words/` +
`segments/`) into `data/ami/conversations.jsonl` in the format above. Speaker ids are
the AMI agent letters (A/B/C/D); note participants self-introduce by name
("I'm Laura, the project manager"), which the model can use to map names → ids.

Observed on the sample: ES2002a ≈ 236 turns / 75% overlap / 18.5 min; ES2002b ≈ 432
turns / 89% overlap / 36 min. The high overlap is exactly why the golden labeler
needs audio.

## Building the golden sample (server)

Sample **stratified** by number of active speakers and by overlap, so the golden
set contains enough genuinely hard turns — otherwise agreement is dominated by
trivial two-party turns and looks better than it is. See the top-level README.
