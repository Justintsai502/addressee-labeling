#!/usr/bin/env python3
"""End-to-end OFFLINE demo — no LLM, no network, no heavy deps.

Proves the plumbing (schema -> windowing -> labeling -> evaluation -> report)
works on any machine. It uses the rule-based MockHeuristicLabeler to stand in for
both models: a clean pass plays the "golden" set, a noisier pass plays the
"candidate", so the evaluation prints realistic (non-100%) agreement numbers.

    python run_demo.py

On the server you replace the two mock labelers with the real ones
(get_labeler('gemini', ...) and get_labeler('qwen', ...)); everything else — the
windowing, parsing, evaluation, and report — is exactly the code exercised here.
"""
from __future__ import annotations

from pathlib import Path

from src.evaluate import acceptance_check, evaluate, format_report
from src.io_utils import load_conversations
from src.labelers import MockHeuristicLabeler

ROOT = Path(__file__).resolve().parent


def main() -> None:
    convs = load_conversations(ROOT / "data/sample/conversations.jsonl")
    print(f"loaded {len(convs)} sample conversations "
          f"({sum(len(c.turns) for c in convs)} turns)\n")

    # Stand-ins for the two real models.
    golden = MockHeuristicLabeler(noise=0.0).label_all(convs)      # ~ audio+transcript
    candidate = MockHeuristicLabeler(noise=0.35, seed=7).label_all(convs)  # ~ transcript only

    result = evaluate(golden, candidate, convs)
    acc = acceptance_check(result, accept_kappa=0.75, accept_exact=0.80)
    print(format_report(result, acc))
    print("\n(demo numbers are from a noise-injected mock, not a real model)")


if __name__ == "__main__":
    main()
