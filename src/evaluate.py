"""Agreement metrics between a candidate labeler and the golden set.

Design choices (why these metrics):

- Addressee classes are highly imbalanced: in two-party stretches the addressee is
  trivially "the other person". Raw accuracy therefore looks great even for a bad
  labeler, so we ALSO report Cohen's kappa (chance-corrected) and per-class F1.
- A turn can have several addressees, so we report set-level metrics (exact-set
  match, mean Jaccard) alongside the single-addressee analysis.
- The whole point of the golden set is to expose where AUDIO helps, so we stratify
  by the hard cases: multi-party (>=3 speakers) turns and time-overlapping turns.

No third-party dependency is required (pure Python). numpy is used only if present,
and only for convenience — results are identical without it.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .schema import AddresseeLabel, Conversation, GROUP, UNKNOWN

_BACKCHANNELS = {
    "yeah", "yep", "yes", "mm", "mm-hm", "mmhm", "uh-huh", "right", "ok", "okay",
    "sure", "exactly", "true", "haha", "hmm", "oh", "wow",
}


def _is_backchannel(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and len(words) <= 3 and any(w in _BACKCHANNELS for w in words)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class TurnRecord:
    """One aligned (gold, pred) decision plus the metadata we stratify on."""

    __slots__ = ("conv_id", "turn_id", "gold", "pred", "n_speakers",
                 "overlap", "backchannel")

    def __init__(self, conv_id, turn_id, gold, pred, n_speakers, overlap, backchannel):
        self.conv_id = conv_id
        self.turn_id = turn_id
        self.gold: Set[str] = gold
        self.pred: Set[str] = pred
        self.n_speakers = n_speakers
        self.overlap = overlap
        self.backchannel = backchannel

    @property
    def exact(self) -> bool:
        return self.gold == self.pred

    @property
    def jaccard(self) -> float:
        return _jaccard(self.gold, self.pred)

    @property
    def both_single(self) -> bool:
        return len(self.gold) == 1 and len(self.pred) == 1


def build_records(
    gold_by_conv: Dict[str, Dict[int, AddresseeLabel]],
    pred_by_conv: Dict[str, Dict[int, AddresseeLabel]],
    convs: Sequence[Conversation],
) -> List[TurnRecord]:
    conv_index = {c.conversation_id: c for c in convs}
    records: List[TurnRecord] = []
    for conv_id, gold in gold_by_conv.items():
        pred = pred_by_conv.get(conv_id, {})
        conv = conv_index.get(conv_id)
        n_speakers = conv.n_speakers if conv else 0
        for turn_id, gold_label in gold.items():
            if turn_id not in pred:
                continue  # only score turns both labeled
            turn = conv.turn_by_id(turn_id) if conv else None
            records.append(
                TurnRecord(
                    conv_id=conv_id,
                    turn_id=turn_id,
                    gold=set(gold_label.normalized().addressees),
                    pred=set(pred[turn_id].normalized().addressees),
                    n_speakers=n_speakers,
                    overlap=bool(turn.overlap) if turn else False,
                    backchannel=_is_backchannel(turn.text) if turn else False,
                )
            )
    return records


# --- metric primitives -------------------------------------------------------

def cohen_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    """Cohen's kappa for single-label (gold, pred) categorical pairs."""
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for g, p in pairs:
        conf[idx[g]][idx[p]] += 1
    n = len(pairs)
    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) / n for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    if abs(1 - pe) < 1e-12:
        return 1.0  # perfect + degenerate (only one class) -> treat as agreement
    return (po - pe) / (1 - pe)


def per_class_prf(records: List[TurnRecord]) -> Dict[str, Dict[str, float]]:
    """Binary one-vs-rest precision/recall/F1 for each addressee class."""
    classes = sorted({c for r in records for c in (r.gold | r.pred)})
    out: Dict[str, Dict[str, float]] = {}
    for c in classes:
        tp = sum(1 for r in records if c in r.gold and c in r.pred)
        fp = sum(1 for r in records if c not in r.gold and c in r.pred)
        fn = sum(1 for r in records if c in r.gold and c not in r.pred)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    return out


def _summary(records: List[TurnRecord]) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    exact = sum(1 for r in records if r.exact) / n
    mean_jac = sum(r.jaccard for r in records) / n
    single_pairs = [
        (next(iter(r.gold)), next(iter(r.pred))) for r in records if r.both_single
    ]
    kappa = cohen_kappa(single_pairs)
    single_acc = (
        sum(1 for g, p in single_pairs if g == p) / len(single_pairs)
        if single_pairs else None
    )
    prf = per_class_prf(records)
    macro_f1 = (sum(v["f1"] for v in prf.values()) / len(prf)) if prf else 0.0
    tp = sum(v["tp"] for v in prf.values())
    fp = sum(v["fp"] for v in prf.values())
    fn = sum(v["fn"] for v in prf.values())
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {
        "n": n,
        "exact_match": exact,
        "mean_jaccard": mean_jac,
        "single_addressee_accuracy": single_acc,
        "cohen_kappa": kappa,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "n_single": len(single_pairs),
    }


def confusion_matrix(records: List[TurnRecord]) -> Tuple[List[str], List[List[int]]]:
    pairs = [
        (next(iter(r.gold)), next(iter(r.pred))) for r in records if r.both_single
    ]
    labels = sorted({x for p in pairs for x in p})
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in pairs:
        m[idx[g]][idx[p]] += 1
    return labels, m


def evaluate(
    gold_by_conv: Dict[str, Dict[int, AddresseeLabel]],
    pred_by_conv: Dict[str, Dict[int, AddresseeLabel]],
    convs: Sequence[Conversation],
) -> Dict:
    """Full evaluation report as a plain dict (JSON-serializable)."""
    records = build_records(gold_by_conv, pred_by_conv, convs)

    strata = {
        "overall": records,
        "two_party (n_speakers<=2)": [r for r in records if r.n_speakers <= 2],
        "multi_party (n_speakers>=3)": [r for r in records if r.n_speakers >= 3],
        "overlap_turns": [r for r in records if r.overlap],
        "non_overlap_turns": [r for r in records if not r.overlap],
        "backchannels": [r for r in records if r.backchannel],
    }
    result = {
        "n_conversations": len(gold_by_conv),
        "n_scored_turns": len(records),
        "strata": {name: _summary(rs) for name, rs in strata.items()},
        "per_class": per_class_prf(records),
    }
    labels, matrix = confusion_matrix(records)
    result["confusion_single_addressee"] = {"labels": labels, "matrix": matrix}
    return result


def acceptance_check(result: Dict, accept_kappa: float, accept_exact: float) -> Dict:
    """Decide PASS/FAIL against pre-registered thresholds."""
    overall = result["strata"]["overall"]
    kappa = overall.get("cohen_kappa")
    exact = overall.get("exact_match", 0.0)
    passed = (kappa is not None and kappa >= accept_kappa) and (exact >= accept_exact)
    return {
        "passed": bool(passed),
        "cohen_kappa": kappa,
        "accept_kappa": accept_kappa,
        "exact_match": exact,
        "accept_exact_match": accept_exact,
    }


# --- pretty printing ---------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        return f"{v:6.3f}"
    return f"{v:>6}"


def format_report(result: Dict, acceptance: Optional[Dict] = None) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("ADDRESSEE LABELING — candidate vs golden agreement")
    lines.append("=" * 72)
    lines.append(f"conversations: {result['n_conversations']}   "
                 f"scored turns: {result['n_scored_turns']}")
    lines.append("")
    header = f"{'stratum':<32}{'n':>5}{'exact':>8}{'jacc':>8}{'kappa':>8}{'macF1':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, s in result["strata"].items():
        if not s.get("n"):
            lines.append(f"{name:<32}{0:>5}   (no turns)")
            continue
        lines.append(
            f"{name:<32}{s['n']:>5}{_fmt(s['exact_match'])}"
            f"{_fmt(s['mean_jaccard'])}{_fmt(s['cohen_kappa'])}{_fmt(s['macro_f1'])}"
        )
    lines.append("")
    lines.append("Per-class F1 (one-vs-rest):")
    lines.append(f"  {'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for c, v in sorted(result["per_class"].items(), key=lambda kv: -kv[1]["support"]):
        lines.append(f"  {c:<12}{v['precision']:8.3f}{v['recall']:8.3f}"
                     f"{v['f1']:8.3f}{v['support']:>9}")
    lines.append("")
    cm = result["confusion_single_addressee"]
    if cm["labels"]:
        lines.append("Confusion (single-addressee turns; rows=gold, cols=pred):")
        head = "  " + " " * 10 + "".join(f"{l[:8]:>9}" for l in cm["labels"])
        lines.append(head)
        for i, l in enumerate(cm["labels"]):
            row = "".join(f"{cm['matrix'][i][j]:>9}" for j in range(len(cm["labels"])))
            lines.append(f"  {l[:10]:<10}{row}")
    if acceptance is not None:
        lines.append("")
        verdict = "PASS ✅" if acceptance["passed"] else "FAIL ❌"
        lines.append(f"Acceptance: {verdict}  "
                     f"(kappa {_fmt(acceptance['cohen_kappa'])} vs "
                     f">={acceptance['accept_kappa']}, "
                     f"exact {_fmt(acceptance['exact_match'])} vs "
                     f">={acceptance['accept_exact_match']})")
    lines.append("=" * 72)
    return "\n".join(lines)
