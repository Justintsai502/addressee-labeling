"""Golden-truth labeler: Gemini 3 Pro with audio + transcript.

This is the *reference* labeler. It sees the most information (audio AND text) with
the strongest available model, and its output is treated as the target the cheap
text-only candidate is measured against.

RUN THIS ON THE SERVER. It needs the `google-genai` package and a GEMINI_API_KEY.
Gemini is an API model — there is no local weight download — but audio tokens are
not free, so run it only on the sampled golden subset, not the full corpus.

    pip install google-genai
    export GEMINI_API_KEY=...            # https://aistudio.google.com/apikey

Audio handling: by default each transcript *window* is cut to its own time span
(`slice_audio=True`) so a 30-minute meeting is not re-uploaded whole on every call.
Set `slice_audio=False` to upload the entire file once (fine for short clips).

Model id note: Google renames/retires preview model ids frequently (there were 50+
"gemini-*" entries at time of writing, many dated/preview). If you get a 404 on the
configured model, list what's actually available to your key:

    python3 -c "from google import genai; [print(m.name) for m in genai.Client().models.list()]"

then pass the correct id explicitly with --model.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, List, Optional

from ..audio_utils import have_ffmpeg, slice_audio, window_time_span
from ..parsing import parse_label_response
from ..prompts import GOLDEN_SYSTEM, build_user_prompt
from ..schema import AddresseeLabel, Conversation
from .base import AddresseeLabeler


class GeminiGoldenLabeler(AddresseeLabeler):
    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_turns_per_window: int = 40,
        context_turns: int = 10,
        send_audio: bool = True,
        slice_audio: bool = True,
        audio_pad: float = 0.5,
    ) -> None:
        super().__init__(max_turns_per_window, context_turns)
        self.name = model
        self.model = model
        self.temperature = temperature
        self.send_audio = send_audio
        self.slice_audio = slice_audio
        self.audio_pad = audio_pad
        self._api_key = api_key
        self._client = None
        self._audio_path: Optional[str] = None   # current conversation's audio
        self._whole_handle = None                # used when slice_audio=False

    def _get_client(self):
        if self._client is None:
            from google import genai  # lazy: keeps google-genai optional

            self._client = genai.Client(api_key=self._api_key or os.environ.get("GEMINI_API_KEY"))
        return self._client

    # -- Files API helpers ----------------------------------------------------
    def _upload_active(self, path: str):
        """Upload a file and wait until it is ACTIVE (required for audio)."""
        client = self._get_client()
        f = client.files.upload(file=path)
        # Poll until the file is processed and usable.
        for _ in range(60):
            if getattr(f.state, "name", str(f.state)) == "ACTIVE":
                return f
            time.sleep(1.0)
            f = client.files.get(name=f.name)
        return f  # best effort; generate_content will error clearly if not ready

    def _delete_quiet(self, handle) -> None:
        try:
            self._get_client().files.delete(name=handle.name)
        except Exception:
            pass  # uploaded files expire on their own

    # -- per-conversation setup ----------------------------------------------
    def _prepare(self, conv: Conversation) -> None:
        self._audio_path = conv.audio_path if self.send_audio else None
        self._whole_handle = None
        if not self._audio_path:
            return
        if self.slice_audio and not have_ffmpeg():
            raise RuntimeError(
                "slice_audio=True but ffmpeg is not installed. Install ffmpeg or "
                "construct the labeler with slice_audio=False."
            )
        if not self.slice_audio:
            self._whole_handle = self._upload_active(self._audio_path)

    def _cleanup(self, conv: Conversation) -> None:
        if self._whole_handle is not None:
            self._delete_quiet(self._whole_handle)
            self._whole_handle = None

    # -- window labeling ------------------------------------------------------
    def _label_window(
        self, conv: Conversation, context: List, target: List, window_text: str
    ) -> Dict[int, AddresseeLabel]:
        from google.genai import types

        client = self._get_client()
        target_ids = [t.turn_id for t in target]

        parts: List[object] = []
        note = ""
        clip_handle = None
        tmp_path = None
        if self._audio_path:
            if self.slice_audio:
                t0, t1 = window_time_span(context + target)
                fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="ami_win_")
                os.close(fd)
                slice_audio(self._audio_path, t0, t1, tmp_path, pad=self.audio_pad)
                clip_handle = self._upload_active(tmp_path)
                parts.append(clip_handle)
                note = (f"The attached audio clip covers roughly {t0:.1f}s–{t1:.1f}s "
                        f"of the meeting (the turns in this window).\n\n")
            else:
                parts.append(self._whole_handle)

        parts.append(note + build_user_prompt(window_text))

        try:
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
        finally:
            if clip_handle is not None:
                self._delete_quiet(clip_handle)
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
