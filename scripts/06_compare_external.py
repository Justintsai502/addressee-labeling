#!/usr/bin/env python3
"""Compare our whole-meeting labels against an external clip-based label bundle.

Use case: a collaborator labelled 3-minute AMI clips (their conversation_id is
`{MEETING}_t{t0}-{t1}`, turn ids restart at 0 per clip) while we labelled whole
meetings (turn ids run across the meeting). Turn ids are therefore NOT
comparable, and neither are raw timestamps — in the external bundle a clip's
turn times are relative to the clip start, so a turn at 12.3s in clip
`ES2002a_t0360-0540` is really at 372.3s on the meeting timeline.

This aligns the two by (speaker, absolute time, text) and scores only the turns
present on both sides.

    python3 scripts/06_compare_external.py \
        --external-dir  ~/Downloads/addressee_labels_ami_golden30 \
        --external-labels labels/gemini_3.1_pro.jsonl \
        --my-conversations data/ami/conversations.jsonl \
        --my-labels outputs/golden_openai_full.jsonl \
        --report outputs/compare_gpt5_vs_gemini.json

Both --my-labels and --external-labels are treated symmetrically; "gold" in the
printed report is the external set purely by convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
from src.evaluate import acceptance_check, evaluate, format_report
from src.io_utils import load_conversations, load_labels
from src.schema import AddresseeLabel, Conversation, Turn


def load_external(ext_dir: Path, labels_rel: str):
    """Return (clip_meta, clip_convs, clip_labels) from an external bundle."""
    meta = {}
    meta_path = ext_dir / "instances_metadata.jsonl"
    if meta_path.exists():
        for line in open(meta_path, encoding="utf-8"):
            r = json.loads(line)
            meta[r["conversation_id"]] = r

    convs = {}
    for line in open(ext_dir / "conversations.jsonl", encoding="utf-8"):
        r = json.loads(line)
        convs[r["conversation_id"]] = r

    labels = {}
    model_ids = set()
    for line in open(ext_dir / labels_rel, encoding="utf-8"):
        r = json.loads(line)
        labels[r["conversation_id"]] = {
            d["turn_id"]: AddresseeLabel.from_dict(d) for d in r["labels"]
        }
        if r.get("model_id"):
            model_ids.add(r["model_id"])
    return meta, convs, labels, model_ids


def clip_offset(cid: str, meta: dict, conv: dict) -> float:
    """Seconds to add to a clip-relative time to get the meeting timeline."""
    if cid in meta and meta[cid].get("clip_t0_sec") is not None:
        return float(meta[cid]["clip_t0_sec"])
    m = (conv.get("meta") or {})
    if m.get("t0") is not None:
        return float(m["t0"])
    # fall back to parsing `{MEETING}_t{t0}-{t1}`
    if "_t" in cid:
        try:
            return float(cid.rsplit("_t", 1)[1].split("-")[0])
        except (ValueError, IndexError):
            pass
    return 0.0


def meeting_of(cid: str, meta: dict, conv: dict) -> str:
    if cid in meta and meta[cid].get("ami_meeting"):
        return meta[cid]["ami_meeting"]
    m = (conv.get("meta") or {})
    return m.get("meeting") or cid.rsplit("_t", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-dir", required=True)
    ap.add_argument("--external-labels", required=True,
                    help="path inside the bundle, e.g. labels/gemini_3.1_pro.jsonl")
    ap.add_argument("--my-conversations", required=True)
    ap.add_argument("--my-labels", required=True)
    ap.add_argument("--tolerance", type=float, default=0.75,
                    help="max |start time| difference (s) when matching turns")
    ap.add_argument("--require-text-match", action="store_true", default=True,
                    help="only align turns whose transcript text is identical")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    ext_dir = Path(args.external_dir).expanduser()
    ext_meta, ext_convs, ext_labels, model_ids = load_external(
        ext_dir, args.external_labels)
    my_convs = {c.conversation_id: c for c in load_conversations(args.my_conversations)}
    my_labels = load_labels(args.my_labels)

    print(f"external : {args.external_labels}  model(s)={sorted(model_ids) or '?'}  "
          f"{len(ext_labels)} clips")
    print(f"mine     : {args.my_labels}  "
          f"{sum(len(v) for v in my_labels.values())} labelled turns "
          f"across {len(my_labels)} conversation(s)\n")

    # --- align ------------------------------------------------------------
    aligned_turns: List[Turn] = []
    gold: Dict[int, AddresseeLabel] = {}
    pred: Dict[int, AddresseeLabel] = {}
    next_id = 0
    stats = {"clips_matched": 0, "clips_skipped": 0,
             "turns_total": 0, "turns_aligned": 0, "turns_unlabelled": 0}
    skipped_meetings = set()

    for cid, econv in sorted(ext_convs.items()):
        meeting = meeting_of(cid, ext_meta, econv)
        if meeting not in my_convs:
            stats["clips_skipped"] += 1
            skipped_meetings.add(meeting)
            continue
        stats["clips_matched"] += 1
        off = clip_offset(cid, ext_meta, econv)
        mine_c = my_convs[meeting]
        mine_labels = my_labels.get(meeting, {})

        for et in econv["turns"]:
            stats["turns_total"] += 1
            abs_start = float(et["start"]) + off
            cands = [
                mt for mt in mine_c.turns
                if mt.speaker == et["speaker"]
                and abs(mt.start - abs_start) <= args.tolerance
                and (not args.require_text_match or mt.text == et["text"])
            ]
            if not cands:
                continue
            mt = min(cands, key=lambda t: abs(t.start - abs_start))
            el = ext_labels.get(cid, {}).get(et["turn_id"])
            ml = mine_labels.get(mt.turn_id)
            if el is None or ml is None:
                stats["turns_unlabelled"] += 1
                continue
            aligned_turns.append(Turn(
                turn_id=next_id, speaker=mt.speaker, start=mt.start, end=mt.end,
                text=mt.text, overlap=mt.overlap,
            ))
            gold[next_id] = AddresseeLabel(next_id, el.normalized().addressees,
                                           el.confidence, el.rationale, el.labeler)
            pred[next_id] = AddresseeLabel(next_id, ml.normalized().addressees,
                                           ml.confidence, ml.rationale, ml.labeler)
            next_id += 1
            stats["turns_aligned"] += 1

    print("alignment:")
    print(f"  clips matched to a meeting we have : {stats['clips_matched']}")
    print(f"  clips skipped (meeting not in ours): {stats['clips_skipped']}"
          + (f"  -> {sorted(skipped_meetings)}" if skipped_meetings else ""))
    print(f"  turns aligned                      : "
          f"{stats['turns_aligned']}/{stats['turns_total']}")
    if stats["turns_unlabelled"]:
        print(f"  matched but unlabelled on one side : {stats['turns_unlabelled']}")
    print()

    if not aligned_turns:
        print("No turns aligned — nothing to compare. Check that --my-conversations "
              "covers the same meetings, and that both sides were labelled.")
        return

    conv = Conversation(conversation_id="aligned", turns=aligned_turns)
    result = evaluate({"aligned": gold}, {"aligned": pred}, [conv])
    acc = acceptance_check(result, 0.75, 0.80)
    print(format_report(result, acc))
    print("\nNOTE: this is agreement between two labelers, not accuracy — "
          "neither side is human-verified ground truth.")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"alignment": stats, "external_models": sorted(model_ids),
                       "result": result, "acceptance": acc},
                      f, ensure_ascii=False, indent=2)
        print(f"\nwrote report -> {args.report}")


if __name__ == "__main__":
    main()
