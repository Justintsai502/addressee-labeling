"""Golden-truth labeler alternative: OpenAI audio-capable chat model (audio + transcript).

Same "golden" role as GeminiGoldenLabeler — an audio+transcript reference labeler
the cheap Qwen candidate is measured against. Use this if your OpenAI key gives you
an audio-capable chat model (e.g. gpt-4o-audio-preview or a newer successor)
instead of, or alongside, Gemini.

RUN ON THE SERVER (or anywhere with network — no local weights, no download).

    pip install openai
    export OPENAI_API_KEY=...

Model id: OpenAI renames/replaces audio-preview models over time, and NOT every
model accepts the `input_audio` chat-completions content type (whisper-1 and
tts-1 do audio but aren't chat/reasoning models). Find what your key can
actually use before picking one:

    python3 -c "
    from openai import OpenAI
    for m in OpenAI().models.list():
        if 'audio' in m.id:
            print(m.id)
    "

then pass the right one with --model.
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import Dict, List, Optional

from ..audio_utils import have_ffmpeg, slice_audio, window_time_span
from ..parsing import parse_label_response
from ..prompts import GOLDEN_SYSTEM, build_user_prompt
from ..schema import AddresseeLabel, Conversation
from .base import AddresseeLabeler


class OpenAIGoldenLabeler(AddresseeLabeler):
    def __init__(
        self,
        model: str = "gpt-4o-audio-preview",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_turns_per_window: int = 40,
        context_turns: int = 10,
        send_audio: bool = True,
        audio_pad: float = 0.5,
        audio_format: str = "wav",
    ) -> None:
        super().__init__(max_turns_per_window, context_turns)
        self.name = model
        self.model = model
        self.temperature = temperature
        self.send_audio = send_audio
        self.audio_pad = audio_pad
        self.audio_format = audio_format
        self._api_key = api_key
        self._client = None
        self._audio_path: Optional[str] = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy: keeps openai optional

            self._client = OpenAI(api_key=self._api_key or os.environ.get("OPENAI_API_KEY"))
        return self._client

    def _prepare(self, conv: Conversation) -> None:
        self._audio_path = conv.audio_path if self.send_audio else None
        if self._audio_path and not have_ffmpeg():
            raise RuntimeError(
                "send_audio=True but ffmpeg is not installed (needed to slice "
                "each window's audio clip)."
            )

    def _label_window(
        self, conv: Conversation, context: List, target: List, window_text: str
    ) -> Dict[int, AddresseeLabel]:
        client = self._get_client()
        target_ids = [t.turn_id for t in target]

        content: List[dict] = [{"type": "text", "text": build_user_prompt(window_text)}]
        tmp_path = None
        if self._audio_path:
            t0, t1 = window_time_span(context + target)
            fd, tmp_path = tempfile.mkstemp(suffix=f".{self.audio_format}", prefix="win_")
            os.close(fd)
            slice_audio(self._audio_path, t0, t1, tmp_path, pad=self.audio_pad)
            with open(tmp_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "input_audio",
                "input_audio": {"data": b64, "format": self.audio_format},
            })

        # 'modalities' only exists on audio-output-capable chat models (the
        # gpt-4o-audio-preview family) and is REJECTED as an unknown parameter by
        # plain text/vision models — only send it when we're actually attaching
        # audio input, to keep it safely off text-only (--no-audio) requests.
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": GOLDEN_SYSTEM},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        if self._audio_path:
            kwargs["modalities"] = ["text"]

        try:
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            return parse_label_response(raw, conv, target_ids, self.name)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
