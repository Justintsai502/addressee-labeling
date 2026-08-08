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

## Building the golden sample (server)

Sample **stratified** by number of active speakers and by overlap, so the golden
set contains enough genuinely hard turns — otherwise agreement is dominated by
trivial two-party turns and looks better than it is. See the top-level README.
