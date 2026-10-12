#!/usr/bin/env python3
"""Score K-sample votes (and an optional greedy pass) against AMI HUMAN addressee labels.

1. Set level (existing metrics, src.evaluate): human vs greedy, human vs majority-vote set.
   Whole-group address is normalised on both sides: GROUP == {all other participants}.
2. Agent view (src.agent_view): every seat that is not the speaker plays the agent.
   Human class in {to_agent, not_to_agent}; LLM vote distribution over
   {to_agent, not_to_agent, uncertain}. Reports BA (to/not), the uncertain rate and what the
   human said on uncertain turns, calibration of P(to_agent) (ECE, 10 bins), strata
   (human group / individual / subset, elicit dialogue acts, overlap).
3. Review export for the human-check page: disagreements first, then a random sample of
   agreements; each item has context, human set, greedy set, the K samples.

    python3 scripts/07_eval_agent_view.py --convs data/ami/dacts_pilot.jsonl \
        --human outputs/agent_view_v3/human_pilot.jsonl --run-dir outputs/agent_view_v3/qwen3_32b
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from src.agent_view import NOT, TO, UNC, VOTE_CLASSES, agent_view, argmax_class, is_group, vote_distribution
from src.evaluate import evaluate
from src.io_utils import load_conversations, load_labels
from src.schema import GROUP, UNKNOWN, AddresseeLabel


def norm_set(addr, speaker, speakers):
    s = sorted({a for a in addr if a})
    if is_group(s, speaker, speakers):
        return [GROUP]
    if UNKNOWN in s:
        return [UNKNOWN]
    return s


def ba(pairs):
    """pairs: (human_class, pred_class) with human in {to, not}; uncertain counts as wrong."""
    rec = {}
    for c in (TO, NOT):
        sub = [p for h, p in pairs if h == c]
        rec[c] = sum(p == c for p in sub) / len(sub) if sub else float("nan")
    acc = sum(h == p for h, p in pairs) / len(pairs) if pairs else float("nan")
    return {"ba": (rec[TO] + rec[NOT]) / 2, "accuracy": acc, "recall_to_agent": rec[TO], "recall_not_to_agent": rec[NOT],
            "n_to": sum(h == TO for h, _ in pairs), "n_not": sum(h == NOT for h, _ in pairs)}


def ece(probs, labels, bins=10):
    tot, err, table = len(probs), 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        err += len(idx) / tot * abs(conf - acc)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx), "mean_p_to": round(conf, 3), "frac_human_to": round(acc, 3)})
    return err, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--review-n-agree", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    run = Path(args.run_dir)
    convs = {c.conversation_id: c for c in load_conversations(args.convs)}
    raw_turns = {json.loads(l)["conversation_id"]: json.loads(l)["turns"] for l in open(args.convs) if l.strip()}
    human = load_labels(args.human)
    votes = {json.loads(l)["conversation_id"]: {t["turn_id"]: t["samples"] for t in json.loads(l)["turns"]}
             for l in open(run / "votes.jsonl") if l.strip()}
    greedy = load_labels(run / "greedy.jsonl") if (run / "greedy.jsonl").exists() else {}

    def normalised(labels_by_conv):
        out = {}
        for cid, labs in labels_by_conv.items():
            conv = convs[cid]
            out[cid] = {tid: AddresseeLabel(turn_id=tid, addressees=norm_set(l.addressees, conv.turn_by_id(tid).speaker, conv.speakers),
                                            labeler=l.labeler) for tid, l in labs.items()}
        return out

    majority = {cid: {tid: AddresseeLabel(turn_id=tid, addressees=list(Counter(tuple(norm_set(s, convs[cid].turn_by_id(tid).speaker, convs[cid].speakers))
                                                                              for s in samples).most_common(1)[0][0]), labeler="vote-majority")
                      for tid, samples in v.items()} for cid, v in votes.items()}
    H = normalised(human)
    report = {"set_level": {}}
    if greedy:
        report["set_level"]["greedy_vs_human"] = evaluate(H, normalised(greedy), list(convs.values()))["strata"]["overall"]
    report["set_level"]["vote_majority_vs_human"] = evaluate(H, majority, list(convs.values()))["strata"]["overall"]

    # agent view
    rows = []
    for cid, labs in human.items():
        conv = convs[cid]
        meta = {t["turn_id"]: t for t in raw_turns[cid]}
        for tid, hl in labs.items():
            turn = conv.turn_by_id(tid)
            samples = votes.get(cid, {}).get(tid)
            if samples is None:
                continue
            grp = is_group(hl.addressees, turn.speaker, conv.speakers)
            kind = "group" if grp else ("individual" if len(hl.addressees) == 1 else "subset")
            for agent in conv.speakers:
                if agent == turn.speaker:
                    continue
                hc = agent_view(hl.addressees, turn.speaker, agent)
                dist = vote_distribution(samples, turn.speaker, agent)
                gc = agent_view(greedy[cid][tid].addressees, turn.speaker, agent) if greedy and tid in greedy.get(cid, {}) else None
                rows.append({"conv": cid, "turn": tid, "agent": agent, "human": hc, "dist": dist, "vote": argmax_class(dist),
                             "greedy": gc, "kind": kind, "elicit": str(meta[tid].get("da_type", "")).startswith("el."),
                             "overlap": bool(turn.overlap)})
    ag = {}

    def summarise(sub):
        if not sub:
            return {"n": 0}
        out = {"n": len(sub), "vote": ba([(r["human"], r["vote"]) for r in sub]),
               "uncertain_rate_vote": sum(r["vote"] == UNC for r in sub) / len(sub)}
        if sub[0]["greedy"] is not None:
            out["greedy"] = ba([(r["human"], r["greedy"]) for r in sub])
            out["uncertain_rate_greedy"] = sum(r["greedy"] == UNC for r in sub) / len(sub)
        return out

    ag["overall"] = summarise(rows)
    for k in ("group", "individual", "subset"):
        ag[f"human_{k}"] = summarise([r for r in rows if r["kind"] == k])
    ag["elicit_dialogue_acts"] = summarise([r for r in rows if r["elicit"]])
    ag["overlap"] = summarise([r for r in rows if r["overlap"]])
    unc = [r for r in rows if r["vote"] == UNC]
    ag["vote_uncertain_breakdown_by_human"] = dict(Counter(f"{r['human']}|{r['kind']}" for r in unc))
    e, table = ece([r["dist"][TO] for r in rows], [1 if r["human"] == TO else 0 for r in rows])
    ag["calibration_p_to_agent"] = {"ece": e, "reliability": table}
    ag["vote_agreement_hist"] = dict(Counter(round(max(r["dist"].values()), 1) for r in rows))
    report["agent_view"] = ag
    report["notes"] = {"human_labels": "AMI manual dialogue-act addressee (unlabeled acts excluded)",
                       "agent_view": "each non-speaker seat as the agent; BA counts uncertain as wrong",
                       "human_audited": False}
    (run / "agent_view_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # review export: per dialogue act (seat-independent), disagreements first
    rng = random.Random(args.seed)
    items = []
    for cid, labs in human.items():
        conv = convs[cid]
        order = [t.turn_id for t in conv.turns]
        pos = {tid: i for i, tid in enumerate(order)}
        meta = {t["turn_id"]: t for t in raw_turns[cid]}
        for tid, hl in labs.items():
            samples = votes.get(cid, {}).get(tid)
            if samples is None:
                continue
            turn = conv.turn_by_id(tid)
            hs = norm_set(hl.addressees, turn.speaker, conv.speakers)
            sample_sets = [norm_set(s, turn.speaker, conv.speakers) for s in samples]
            maj = list(Counter(tuple(s) for s in sample_sets).most_common(1)[0][0])
            gs = norm_set(greedy[cid][tid].addressees, turn.speaker, conv.speakers) if greedy and tid in greedy.get(cid, {}) else None
            ctx = [{"speaker": conv.turns[j].speaker, "text": conv.turns[j].text}
                   for j in range(max(0, pos[tid] - 4), min(len(order), pos[tid] + 3)) if j != pos[tid]]
            before = [c for j, c in zip(range(max(0, pos[tid] - 4), pos[tid]), ctx)]
            items.append({"id": f"{cid}:{tid}", "meeting": cid, "turn_id": tid, "speaker": turn.speaker,
                          "speakers": conv.speakers, "text": turn.text, "da_type": meta[tid].get("da_type"),
                          "start": turn.start, "context_before": [{"speaker": conv.turns[j].speaker, "text": conv.turns[j].text}
                                                                  for j in range(max(0, pos[tid] - 4), pos[tid])],
                          "context_after": [{"speaker": conv.turns[j].speaker, "text": conv.turns[j].text}
                                            for j in range(pos[tid] + 1, min(len(order), pos[tid] + 3))],
                          "human": hs, "llm_majority": maj, "llm_greedy": gs, "llm_samples": sample_sets,
                          "agree": hs == maj})
    dis = [i for i in items if not i["agree"]]
    agr = [i for i in items if i["agree"]]
    rng.shuffle(dis); rng.shuffle(agr)
    review = dis + agr[: args.review_n_agree]
    (run / "review_items.json").write_text(json.dumps({"labeler": next(iter(json.loads(l)["labeler"] for l in open(run / "votes.jsonl"))),
                                                       "n_items_total": len(items), "n_disagree": len(dis), "items": review},
                                                      ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"set_level": report["set_level"], "agent_view_overall": ag["overall"],
                      "group": ag["human_group"], "individual": ag["human_individual"], "ece": e,
                      "review_items": len(review), "disagreements": len(dis)}, indent=2))


if __name__ == "__main__":
    main()
