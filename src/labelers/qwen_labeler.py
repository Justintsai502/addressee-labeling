"""Candidate labeler: Qwen with transcript only (no audio).

This is the *cheap, scalable* labeler we are validating. If it agrees closely
enough with the Gemini golden set, we trust it to label the full 2000-hour corpus
at near-zero marginal cost (self-hosted, no per-call API fee).

It talks to any OpenAI-compatible chat endpoint, which is deliberately how the
heavy model stays OFF this laptop and ON the server:

  Option A - self-hosted (no API fee, weights live on the server):
      pip install vllm openai
      vllm serve Qwen/Qwen3-32B --port 8000          # downloads weights ON THE SERVER
      base_url = "http://localhost:8000/v1"          # api_key can be any string

  Option B - hosted API (no download at all):
      base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
      api_key  = $DASHSCOPE_API_KEY

Either way this file never downloads or loads model weights itself, so it is safe
to import and construct anywhere; it only reaches out when `label_conversation`
actually runs against a reachable endpoint.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..parsing import parse_label_response
from ..prompts import CANDIDATE_SYSTEM, build_user_prompt
from ..schema import AddresseeLabel, Conversation
from .base import AddresseeLabeler


class QwenCandidateLabeler(AddresseeLabeler):
    def __init__(
        self,
        model: str = "Qwen3-32B",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_turns_per_window: int = 40,
        context_turns: int = 10,
    ) -> None:
        super().__init__(max_turns_per_window, context_turns)
        self.name = model
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    def _label_window(
        self, conv: Conversation, context: List, target: List, window_text: str
    ) -> Dict[int, AddresseeLabel]:
        client = self._get_client()
        target_ids = [t.turn_id for t in target]

        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": CANDIDATE_SYSTEM},
                {"role": "user", "content": build_user_prompt(window_text)},
            ],
            # vLLM & DashScope both honor this; it strongly nudges valid JSON.
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return parse_label_response(raw, conv, target_ids, self.name)
