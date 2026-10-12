"""K-sample voting on top of the local (vLLM) candidate labeler.

One vLLM request per window with SamplingParams(n=K, temperature=T); each sample is
parsed with the same `parse_label_response`, so every turn gets K independent label
sets. `label_conversation` still returns one label per turn (sample 0, to keep the base
API); the K samples are kept in `self.votes[conversation_id][turn_id]`.

Prompt: identical to CANDIDATE_SYSTEM plus one guideline (V3_NOTE) that separates
"who is addressed" from "is a reply expected" — group statements that ask nothing are
still addressed to the group.

`MockVotingLabeler` gives the same interface without a model (K noisy heuristic runs),
for CPU dry runs of the whole pipeline.
"""

from __future__ import annotations

from typing import Dict, List

from ..parsing import parse_label_response
from ..prompts import CANDIDATE_SYSTEM, build_user_prompt
from ..schema import AddresseeLabel, Conversation
from .local_labeler import LocalLLMLabeler
from .mock_labeler import MockHeuristicLabeler

V3_NOTE = """

ADDITIONAL GUIDELINE (addressee only)
Label WHO the speaker is talking to, not whether anyone is expected to reply.
A statement to the whole room that asks for nothing ("okay, let's take a break")
is still addressed to the group. Do not use UNKNOWN just because no reply is
needed; use it only when you cannot tell who is being talked to.
"""

CANDIDATE_SYSTEM_V3 = CANDIDATE_SYSTEM + V3_NOTE

# v4 (2026-10-10, after reading ttv2 disagreements with the user): decide from meaning and context.
V4_NOTE = V3_NOTE + """
DECIDING WHEN NO ONE IS NAMED (read the surrounding turns)
- Answering a question or request -> addressed to the person who asked it, especially when
  that question was put to this speaker ("Sam, can you ...?" -> Sam's answer is to the asker).
- Reacting to someone's specific proposal or claim (agreeing, objecting, adding a caveat,
  "fine by me", "but that will take too long") -> addressed to that person.
- Opening a new topic, asking the room a question, announcing or proposing something to
  everyone -> GROUP.
- Answering a question that was asked to the whole room, or chiming in on a remark made to
  the whole room, can be either the asker or GROUP; choose the one the wording supports
  ("you" singular / the asker's point -> asker; "we", "let's", "everyone" -> GROUP) and lower
  your confidence.
- A person mentioned in the third person ("Sam and I will handle it") is NOT the addressee.
"""

CANDIDATE_SYSTEM_V4 = CANDIDATE_SYSTEM + V4_NOTE
PROMPTS = {"v3": CANDIDATE_SYSTEM_V3, "v4": CANDIDATE_SYSTEM_V4}


class VotingLocalLabeler(LocalLLMLabeler):
    def __init__(self, k: int = 5, sample_temperature: float = 0.7, prompt: str = "v3", **kw) -> None:
        super().__init__(**kw)
        self.system_prompt = PROMPTS[prompt]
        self.prompt_version = prompt
        # fail fast: an engine error must stop the run, not be recorded as UNKNOWN labels
        # (job 519722 produced an all-UNKNOWN "result" that way)
        self.skip_failed_windows = False
        self.k = int(k)
        self.sample_temperature = float(sample_temperature)
        self.name = f"{self.name}-vote{self.k}@T{self.sample_temperature}-prompt{prompt}"
        self.votes: Dict[str, Dict[int, List[List[str]]]] = {}

    def _label_window(self, conv: Conversation, context: List, target: List, window_text: str) -> Dict[int, AddresseeLabel]:
        from vllm import SamplingParams

        engine = self._get_engine()
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": build_user_prompt(window_text)}]
        sp = SamplingParams(n=self.k, temperature=self.sample_temperature if self.k > 1 else self.temperature,
                            max_tokens=self.max_new_tokens)
        out = engine.chat(messages, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": self.enable_thinking})
        target_ids = [t.turn_id for t in target]
        per_sample = []
        for o in out[0].outputs:
            try:
                per_sample.append(parse_label_response(o.text, conv, target_ids, self.name))
            except Exception:  # unparsable sample -> all UNKNOWN for this sample (counted downstream)
                per_sample.append({tid: AddresseeLabel(turn_id=tid, addressees=["UNKNOWN"], confidence=0.0,
                                                      rationale="unparsable sample", labeler=f"{self.name}(failed)")
                                   for tid in target_ids})
            self.stats["output_tokens"] += len(o.token_ids or [])
        self.stats["n_calls"] += 1
        store = self.votes.setdefault(conv.conversation_id, {})
        for tid in target_ids:
            store[tid] = [s[tid].normalized().addressees for s in per_sample]
        return per_sample[0]


class MockVotingLabeler(MockHeuristicLabeler):
    def __init__(self, k: int = 5, noise: float = 0.3, seed: int = 0, **kw) -> None:
        super().__init__(noise=noise, seed=seed, **kw)
        self.k = int(k)
        self.name = f"mock-vote{self.k}"
        self.votes: Dict[str, Dict[int, List[List[str]]]] = {}

    def _label_window(self, conv, context, target, window_text):
        samples = [super(MockVotingLabeler, self)._label_window(conv, context, target, window_text) for _ in range(self.k)]
        store = self.votes.setdefault(conv.conversation_id, {})
        for t in target:
            store[t.turn_id] = [s[t.turn_id].addressees for s in samples]
        return samples[0]


# --- causal ("listener at the end of the turn") variant -------------------------------
CAUSAL_NOTE = """

CAUSAL SETTING (overrides rule 2 above)
You see the conversation only UP TO AND INCLUDING the turn to label; later turns are not
shown. Decide who is being addressed as a listener would at the moment this turn ends.
Do NOT guess from who will answer next. If, at this moment, the turn could be for more
than one addressee, pick the best-supported one and lower your confidence, or use
UNKNOWN if you cannot tell.
"""


class CausalVotingLabeler(VotingLocalLabeler):
    """One request per turn: context = up to `context_turns` previous turns, target = this turn.
    All turns of a conversation go to vLLM as one batch (n=K samples each)."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.system_prompt = self.system_prompt + CAUSAL_NOTE
        self.name = self.name + "-causal"

    def label_conversation(self, conv: Conversation, existing=None, on_progress=None):
        from vllm import SamplingParams

        from ..transcript_format import render_window

        engine = self._get_engine()
        turns = conv.turns
        batch = []
        for i, t in enumerate(turns):
            window = render_window(conv, turns[max(0, i - self.context_turns):i], [t])
            batch.append([{"role": "system", "content": self.system_prompt},
                          {"role": "user", "content": build_user_prompt(window)}])
        # k == 1 is the greedy pass (06_vote_label.py --greedy); same rule as _label_window
        sp = SamplingParams(n=self.k, temperature=self.sample_temperature if self.k > 1 else self.temperature,
                            max_tokens=self.max_new_tokens)
        outs = engine.chat(batch, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": self.enable_thinking})
        store = self.votes.setdefault(conv.conversation_id, {})
        first: Dict[int, AddresseeLabel] = {}
        for t, out in zip(turns, outs):
            samples = []
            for o in out.outputs:
                try:
                    lab = parse_label_response(o.text, conv, [t.turn_id], self.name)[t.turn_id]
                except Exception:
                    lab = AddresseeLabel(turn_id=t.turn_id, addressees=["UNKNOWN"], confidence=0.0,
                                         rationale="unparsable sample", labeler=f"{self.name}(failed)")
                samples.append(lab.normalized().addressees)
                self.stats["output_tokens"] += len(o.token_ids or [])
            store[t.turn_id] = samples
            first[t.turn_id] = AddresseeLabel(turn_id=t.turn_id, addressees=samples[0], labeler=self.name)
        self.stats["n_calls"] += len(batch)
        return first
