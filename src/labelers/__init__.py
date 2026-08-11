"""Addressee labelers.

Only the base class and the offline mock are imported eagerly; the Gemini and Qwen
backends are imported lazily via `get_labeler` so that importing this package never
requires google-genai / openai to be installed.
"""

from __future__ import annotations

from typing import Any

from .base import AddresseeLabeler
from .mock_labeler import MockHeuristicLabeler

__all__ = ["AddresseeLabeler", "MockHeuristicLabeler", "get_labeler"]


def get_labeler(kind: str, **kwargs: Any) -> AddresseeLabeler:
    """Factory: 'gemini' | 'openai' | 'qwen' | 'local' | 'mock'.

    - 'gemini' : golden labeler, Gemini API, audio+transcript.
    - 'openai' : golden labeler alternative, OpenAI audio-capable chat model.
    - 'qwen'   : candidate via an OpenAI-compatible endpoint (vllm serve / DashScope).
    - 'local'  : candidate via locally DOWNLOADED weights (vllm/hf) — swap models by id.
    - 'mock'   : offline heuristic, no model.
    """
    kind = kind.lower()
    if kind in ("gemini", "golden"):
        from .gemini_labeler import GeminiGoldenLabeler

        return GeminiGoldenLabeler(**kwargs)
    if kind in ("openai", "gpt"):
        from .openai_labeler import OpenAIGoldenLabeler

        return OpenAIGoldenLabeler(**kwargs)
    if kind in ("qwen", "endpoint", "candidate"):
        from .qwen_labeler import QwenCandidateLabeler

        return QwenCandidateLabeler(**kwargs)
    if kind in ("local", "hf", "vllm"):
        from .local_labeler import LocalLLMLabeler

        return LocalLLMLabeler(**kwargs)
    if kind in ("mock", "heuristic"):
        return MockHeuristicLabeler(**kwargs)
    raise ValueError(f"unknown labeler kind: {kind!r}")
