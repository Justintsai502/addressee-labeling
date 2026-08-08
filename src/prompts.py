"""Prompt templates for the two labelers.

The *label schema* and *guidelines* are shared verbatim so that the only
controlled difference between the golden labeler and the candidate labeler is the
**modality** (Gemini sees audio+transcript; Qwen sees transcript only). Keeping the
instructions identical is what makes the agreement number interpretable: a gap
means "audio helped", not "the prompts differed".
"""

from __future__ import annotations

from .schema import GROUP, UNKNOWN

# --- Shared definition of the task and the output format ---------------------

LABEL_SCHEMA = f"""\
TASK
For every turn marked "[>> ]", decide WHO the speaker is talking TO (the addressee).

LABEL SET (closed)
- One or more of the OTHER speaker ids listed for this conversation
  (never label the speaker as addressing themselves).
- "{GROUP}": the speaker is addressing everyone / the whole room, not one person.
- "{UNKNOWN}": the addressee genuinely cannot be determined.
A turn may have MORE THAN ONE addressee (e.g. ["B", "C"]). Use "{UNKNOWN}" alone,
never mixed with concrete ids.

HOW TO DECIDE (use every cue available to you)
1. Vocatives / names: "Bob, can you...", "what do you think, Dr. Lee?" -> that person.
2. Question-and-answer adjacency: who answers next usually reveals who was addressed.
   Look at the surrounding [CTX] and [>> ] turns as a whole.
3. Second-person pronouns: "you" (singular) points at one prior/next speaker;
   "you all", "you guys", "everyone" -> {GROUP}.
4. Topic / reference continuity: replies that pick up another speaker's point are
   usually addressed to that speaker.
5. Backchannels ("mm-hm", "yeah", "right", laughter) are addressed to whoever
   currently holds the floor (the speaker they are reacting to).
6. In a strictly two-person stretch, the addressee is simply the other person.

OUTPUT
Return ONLY a JSON object of this exact shape, with one entry per "[>> ]" turn:
{{
  "labels": [
    {{"turn_id": <int>, "addressees": ["<id>" | "{GROUP}" | "{UNKNOWN}", ...],
      "confidence": <0.0-1.0>, "rationale": "<short reason>"}}
  ]
}}
Do not include context ("[CTX]") turns. Do not add any text outside the JSON.
"""


GOLDEN_SYSTEM = f"""\
You are an expert conversation analyst labeling addressee information for a
multi-speaker spoken-dialogue dataset. You are given BOTH the audio of the
conversation AND its diarized transcript. Use the audio to resolve cases the text
is ambiguous about: prosody and intonation (a directed question vs. a broadcast
remark), who a speaker turns toward vocally, emphasis, and who actually responds.
The transcript's speaker ids and turn ids are authoritative — align everything you
hear to those ids.

{LABEL_SCHEMA}"""


CANDIDATE_SYSTEM = f"""\
You are an expert conversation analyst labeling addressee information for a
multi-speaker spoken-dialogue dataset. You are given the diarized transcript only
(no audio). Infer the addressee from the words, the speaker ids, and the
turn-taking structure. When the text is genuinely ambiguous, prefer "{UNKNOWN}"
over guessing, and lower your confidence.

{LABEL_SCHEMA}"""


def build_user_prompt(window_text: str) -> str:
    return (
        "Here is the conversation window. Label every [>> ] turn.\n\n"
        + window_text
        + "\n\nReturn the JSON object now."
    )
