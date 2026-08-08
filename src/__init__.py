"""Addressee-labeling validation harness.

Public surface:
    from src.schema import Conversation, Turn, AddresseeLabel
    from src.labelers import get_labeler
    from src.evaluate import evaluate, acceptance_check, format_report
"""

from __future__ import annotations

__all__ = ["schema", "io_utils", "transcript_format", "prompts", "parsing",
           "evaluate", "labelers", "config"]
