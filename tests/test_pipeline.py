#!/usr/bin/env python3
"""Self-contained tests — run with plain `python tests/test_pipeline.py` (no pytest).

Covers the offline-critical logic: JSON parsing/validation, windowing, and the
evaluation metrics (against hand-computed expected values).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import cohen_kappa, evaluate, per_class_prf, build_records  # noqa: E402
from src.parsing import parse_label_response  # noqa: E402
from src.schema import AddresseeLabel, Conversation, Turn, GROUP, UNKNOWN  # noqa: E402
from src.transcript_format import iter_windows  # noqa: E402

_fail = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global _fail
    status = "ok  " if cond else "FAIL"
    if not cond:
        _fail += 1
    print(f"[{status}] {name}{('  -> ' + extra) if extra and not cond else ''}")


def approx(a, b, tol=1e-6) -> bool:
    return a is not None and abs(a - b) <= tol


# --- fixtures ---------------------------------------------------------------
def make_conv() -> Conversation:
    return Conversation(
        conversation_id="c1",
        speakers=["Alice", "Bob", "Carol"],
        turns=[
            Turn(0, "Alice", 0.0, 1.0, "Bob, ready?"),
            Turn(1, "Bob", 1.1, 2.0, "Yep."),
            Turn(2, "Carol", 2.1, 3.0, "Nice.", overlap=True),
        ],
    )


# --- parsing ----------------------------------------------------------------
def test_parsing() -> None:
    conv = make_conv()
    raw = """```json
    {"labels": [
        {"turn_id": 0, "addressees": ["Bob"], "confidence": 0.9},
        {"turn_id": 1, "addressees": ["Alice"]},
        {"turn_id": 2, "addressees": ["Alice", "Alice"]}
    ]}
    ```"""
    got = parse_label_response(raw, conv, [0, 1, 2], "test")
    check("parse: fenced json parsed", set(got) == {0, 1, 2})
    check("parse: dedup addressees", got[2].addressees == ["Alice"])

    # invalid id (speaker addressing self) dropped -> UNKNOWN; missing turn filled
    raw2 = '{"labels": [{"turn_id": 0, "addressees": ["Alice"]}]}'
    got2 = parse_label_response(raw2, conv, [0, 1], "test")
    check("parse: self-address dropped -> UNKNOWN", got2[0].addressees == [UNKNOWN])
    check("parse: missing turn filled -> UNKNOWN", got2[1].addressees == [UNKNOWN])


# --- windowing --------------------------------------------------------------
def test_windows() -> None:
    conv = Conversation(
        conversation_id="c",
        speakers=["A", "B"],
        turns=[Turn(i, "A" if i % 2 == 0 else "B", i, i + 0.5, "x") for i in range(10)],
    )
    wins = iter_windows(conv, max_turns_per_window=4, context_turns=2)
    check("windows: count", len(wins) == 3, str(len(wins)))
    check("windows: first has no context", wins[0][0] == [])
    check("windows: second carries 2 context turns", len(wins[1][0]) == 2)
    targets = [t.turn_id for _, tg in wins for t in tg]
    check("windows: every turn labeled once", targets == list(range(10)))


# --- metrics ----------------------------------------------------------------
def test_kappa() -> None:
    # 8 agree, 2 disagree over 2 classes -> known kappa.
    pairs = [("X", "X")] * 4 + [("Y", "Y")] * 4 + [("X", "Y"), ("Y", "X")]
    k = cohen_kappa(pairs)
    # po=0.8; marginals are 0.5/0.5 each -> pe=0.5 -> kappa=0.6
    check("kappa: hand-computed 0.6", approx(k, 0.6, 1e-9), str(k))
    check("kappa: perfect -> 1.0", approx(cohen_kappa([("A", "A"), ("B", "B")]), 1.0))
    check("kappa: single-class degenerate -> 1.0", approx(cohen_kappa([("A", "A")]), 1.0))


def test_evaluate() -> None:
    conv = make_conv()
    gold = {
        "c1": {
            0: AddresseeLabel(0, ["Bob"]),
            1: AddresseeLabel(1, ["Alice"]),
            2: AddresseeLabel(2, ["Bob"]),  # backchannel to floor holder
        }
    }
    pred = {
        "c1": {
            0: AddresseeLabel(0, ["Bob"]),      # correct
            1: AddresseeLabel(1, ["Alice"]),    # correct
            2: AddresseeLabel(2, [GROUP]),      # wrong (overlap turn)
        }
    }
    res = evaluate(gold, pred, [conv])
    overall = res["strata"]["overall"]
    check("eval: 3 turns scored", res["n_scored_turns"] == 3)
    check("eval: exact match 2/3", approx(overall["exact_match"], 2 / 3))
    check("eval: overlap stratum present", res["strata"]["overlap_turns"]["n"] == 1)
    check("eval: overlap turn wrong -> exact 0",
          approx(res["strata"]["overlap_turns"]["exact_match"], 0.0))
    prf = per_class_prf(build_records(gold, pred, [conv]))
    check("eval: Bob recall 1/2 (one Bob missed)", approx(prf["Bob"]["recall"], 0.5))


def main() -> None:
    test_parsing()
    test_windows()
    test_kappa()
    test_evaluate()
    print()
    if _fail:
        print(f"{_fail} check(s) FAILED")
        sys.exit(1)
    print("all checks passed ✅")


if __name__ == "__main__":
    main()
