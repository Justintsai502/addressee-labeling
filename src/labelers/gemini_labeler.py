"""Golden-truth labeler: Gemini 3 Pro with audio + transcript.

This is the *reference* labeler. It sees the most information (audio AND text) with
the strongest available model, and its output is treated as the target that the
cheap text-only candidate is measured against.

RUN THIS ON THE SERVER. It needs the `google-genai` package and a GEMINI_API_KEY,
and it uploads audio through the Files API (so multi-minute conversations are fine).
Gemini is an API model — there is no local weight download — but audio tokens are
not free, so run it only on the sampled golden subset, not the full corpus.

    pip install google-genai
    export GEMINI_API_KEY=...            # from https://aistudio.google.com/apikey
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..parsing import parse_label_response
from ..prompts import GOLDEN_SYSTEM, build_user_prompt
from ..schema import AddresseeLabel, Conversation
from .base import AddresseeLabeler


class GeminiGoldenLabeler(AddresseeLabeler):
    def __init__(
        self,
        model: str = "gemini-3-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_turns_per_window: int = 40,
        context_turns: int = 10,
        send_audio: bool = True,
    ) -> None:
        super().__init__(max_turns_per_window, context_turns)
        self.name = model
        self.model = model
        self.temperature = temperature
        self.send_audio = send_audio
        self._api_key = api_key
        self._client = None            # lazily created on first use
        self._audio_handle = None      # uploaded File handle for the current conversation

    def _get_client(self):
        if self._client is None:
            # Imported lazily so the rest of the project runs without google-genai.
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # -- upload the conversation's audio once per conversation ----------------
    def _prepare(self, conv: Conversation) -> None:
        self._audio_handle = None
        if not (self.send_audio and conv.audio_path):
            return
        client = self._get_client()
        # Files API handles large/long audio; returns a handle we pass in contents.
        self._audio_handle = client.files.upload(file=conv.audio_path)

    def _cleanup(self, conv: Conversation) -> None:
        if self._audio_handle is not None:
            try:
                self._get_client().files.delete(name=self._audio_handle.name)
            except Exception:
                pass  # best-effort; uploaded files expire on their own
            self._audio_handle = None

    def _label_window(
        self, conv: Conversation, context: List, target: List, window_text: str
    ) -> Dict[int, AddresseeLabel]:
        from google.genai import types  # type: ignore

        client = self._get_client()
        target_ids = [t.turn_id for t in target]

        parts: List[object] = []
        if self._audio_handle is not None:
            parts.append(self._audio_handle)
        parts.append(build_user_prompt(window_text))

        response = client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=GOLDEN_SYSTEM,
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )
        return parse_label_response(response.text, conv, target_ids, self.name)
