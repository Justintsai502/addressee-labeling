#!/usr/bin/env python3
"""Report for the ADS v3 / prompt v5 dev run (consistency and rule compliance only; v4 / generator are not gold).

    python3 scripts/14_report_v5.py --causal outputs/ads_v3_v5/dev20_causal --full outputs/ads_v3_v5/dev20_full \
        --frames outputs/ads_v3_v5/dev20_frames --v4 outputs/agent_view_v3/ttv2_qwen3_32b_v4_causal/votes.jsonl \
        --ref outputs/agent_view_v3/ttv2_ref_pilot.jsonl --fixed data/ttv2/fixed_cases_v5.json \
        --lock /work/b12901015/mpmoshi/artifacts/ads_v3_labeling_1011/LOCK_dev20.json --out outputs/ads_v3_v5/dev20_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from src.clause_v5 import ALL, majority_norm


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm_v4(sample, speaker, speakers):
    s = {a for a in sample if a}
    if not s or "UNKNOWN" in s:
        return None
    others = {x for x in speakers if x != speaker}
    if "GROUP" in s or s == others:
        return ALL
    return "|".join(sorted(s))


def validity(rows):
    slots = [v for r in rows for v in r["slots"]]
    r1 = Counter(v["fail"] or "valid" for v in slots)
    fin = [f for r in rows for f in r["final"]]
    return {"n_clauses": len(rows), "n_slots": len(slots), "round1": dict(r1),
            "round1_invalid_rate": round(1 - r1["valid"] / max(1, len(slots)), 4),
            "retry_fixed": sum(1 for v in slots if v["norm"] is None and v.get("retry", {}).get("norm")),
            "invalid_after_retry_rate": round(sum(f["norm"] is None for f in fin) / max(1, len(fin)), 4),
            "masked_clause_rate": round(sum(r["valid_vote_count"] < 3 for r in rows) / max(1, len(rows)), 4),
            "majority_count_hist": dict(sorted(Counter(r["majority_count"] for r in rows).items()))}


def fixed_eval(rows, fixed):
    by = defaultdict(list)
    for r in rows:
        if r["conversation_id"].startswith("fixed:"):
            by[r["conversation_id"][6:]].append(r)
    out, npass, ncount = [], 0, 0
    for case in fixed["cases"]:
        cl = sorted(by[case["case_id"]], key=lambda r: r["clause_idx"])
        res = []
        ok_all = len(cl) == len(case["expect"])
        for r, e in zip(cl, case["expect"]):
            maj = r["majority"]
            ok = None if e.get("report_only") else (maj in e["any_of"] and not any(
                f in (maj or "").split("|") for f in e.get("forbid", [])))
            if ok is False:
                ok_all = False
            res.append({"clause": r["clause"], "majority": maj, "count": r["majority_count"],
                        "valid": r["valid_vote_count"], "ok": ok,
                        "fails": Counter(v["fail"] for v in r["slots"] if v["fail"])})
        report_only = all(e.get("report_only") for e in case["expect"])
        if not report_only:
            ncount += 1
            npass += int(ok_all)
        out.append({"case_id": case["case_id"], "pass": None if report_only else ok_all, "clauses": res})
    return {"pass": npass, "n": ncount, "cases": out}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--causal", "--frames", "--out"):
        ap.add_argument(a, required=True)
    for a in ("--full", "--v4", "--ref", "--fixed", "--lock"):  # optional comparisons / gates
        ap.add_argument(a, default=None)
    args = ap.parse_args()
    C = load_jsonl(Path(args.causal) / "clause_votes.jsonl")
    F = load_jsonl(Path(args.full) / "clause_votes.jsonl") if args.full else []
    fixed = json.loads(Path(args.fixed).read_text()) if args.fixed else None
    Cd = [r for r in C if not r["conversation_id"].startswith("fixed:")]
    Fd = [r for r in F if not r["conversation_id"].startswith("fixed:")]
    rep = {"human_audited": False, "note": "dev20 = pilot dialogues used for prompt v4 development; not held-out. "
           "Agreement numbers are consistency checks, not accuracy."}
    rep["runs"] = {m: json.loads((Path(d) / "run.json").read_text()) for m, d in (("causal", args.causal), ("full", args.full))
                   if d and (Path(d) / "run.json").exists()}
    rep["validity"] = {"causal": validity(Cd), **({"full": validity(Fd)} if Fd else {})}
    if fixed:
        rep["fixed_cases"] = {"causal": fixed_eval(C, fixed), **({"full": fixed_eval(F, fixed)} if F else {})}
    # clauses per turn, multi-addressee turns
    turns = defaultdict(list)
    for r in Cd:
        turns[(r["conversation_id"], r["turn_id"])].append(r)
    multi = [(k, sorted(v, key=lambda r: r["clause_idx"])) for k, v in turns.items()
             if len({r["majority"] for r in v if r["valid_vote_count"] >= 3}) > 1]
    rep["turns"] = {"n_turns": len(turns), "n_clauses": len(Cd), "clauses_per_turn": round(len(Cd) / len(turns), 3),
                    "multi_clause_turns": sum(len(v) > 1 for v in turns.values()),
                    "turns_with_differing_clause_addressees": len(multi),
                    "examples": [{"id": f"{k[0]}/{k[1]}", "clauses": [[r["clause"], r["majority"], r["majority_count"]] for r in v]}
                                 for k, v in multi[:15]]}
    # v4 causal regression on single-clause turns
    v4 = {}
    spk = {}
    for r in Cd:
        spk[(r["conversation_id"], r["turn_id"])] = (r["speaker"], r["speakers"])
    for row in (load_jsonl(args.v4) if args.v4 else []):
        for t in row["turns"]:
            key = (row["conversation_id"], t["turn_id"])
            if key in spk:
                s, ss = spk[key]
                v4[key] = majority_norm([norm_v4(x, s, ss) for x in t["samples"]])
    agree = n = 0
    diffs = []
    for k, v in turns.items():
        if len(v) != 1 or k not in v4 or v[0]["majority"] is None or v4[k][0] is None:
            continue
        n += 1
        if v[0]["majority"] == v4[k][0]:
            agree += 1
        elif len(diffs) < 15:
            diffs.append({"id": f"{k[0]}/{k[1]}", "text": v[0]["clause"], "v5": v[0]["majority"], "v4": v4[k][0]})
    rep["v4_causal_single_clause_agreement"] = {"n": n, "agree": agree, "rate": round(agree / max(1, n), 4), "diffs": diffs}
    # causal vs full
    fk = {(r["conversation_id"], r["turn_id"], r["clause_idx"]): r for r in Fd}
    flips = conf = both = 0
    flip_ex = []
    for r in Cd:
        f = fk.get((r["conversation_id"], r["turn_id"], r["clause_idx"]))
        if not f or r["majority"] is None or f["majority"] is None:
            continue
        both += 1
        if r["majority"] != f["majority"]:
            flips += 1
            if r["majority_count"] >= 4 and f["majority_count"] >= 4:
                conf += 1
                if len(flip_ex) < 15:
                    flip_ex.append({"id": f"{r['conversation_id']}/{r['turn_id']}/{r['clause_idx']}", "clause": r["clause"],
                                    "causal": r["majority"], "full": f["majority"]})
    rep["causal_vs_full"] = {"n": both, "flips": flips, "flip_rate": round(flips / max(1, both), 4),
                             "confident_flips": conf, "confident_flip_rate": round(conf / max(1, both), 4), "examples": flip_ex}
    # generator reference (turn level) vs v5 causal clause majority
    ref = {}
    for row in (load_jsonl(args.ref) if args.ref else []):
        for l in row["labels"]:
            ref[(row["conversation_id"], l["turn_id"])] = l["addressees"]
    g = Counter()
    for r in Cd:
        a = ref.get((r["conversation_id"], r["turn_id"]))
        if a is None or r["majority"] is None:
            continue
        gn = norm_v4(a, r["speaker"], r["speakers"])
        typ = "EVERYONE" if gn == ALL else "named"
        g[(typ, "n")] += 1
        g[(typ, "agree")] += int(gn == r["majority"])
    rep["generator_agreement"] = {t: {"n": g[(t, "n")], "rate": round(g[(t, "agree")] / max(1, g[(t, "n")]), 4)}
                                  for t in ("EVERYONE", "named")}
    # greedy vs vote majority
    gr = [r for r in Cd if r.get("greedy")]
    rep["greedy_vs_majority"] = {"n": len(gr), "agree_rate": round(
        sum(r["greedy"]["norm"] == r["majority"] for r in gr) / max(1, len(gr)), 4),
        "greedy_invalid": sum(r["greedy"]["norm"] is None for r in gr)}
    # cue diagnostics
    cl = load_jsonl(Path(args.frames) / "clauses.jsonl")
    kinds = Counter(c["kind"] for r in cl for c in r["cues"] if c)
    pos = Counter()
    for r in cl:
        for c in r["cues"]:
            if c and c["kind"] == "in_clause":
                x = c["rel_char_end"]
                pos["first_third" if x < 1 / 3 else "middle_third" if x < 2 / 3 else "last_third"] += 1
    same = sum(1 for r in Cd if len({(f["cue"] or "").strip().lower() for f in r["final"] if f["norm"]}) == 1)
    rep["cue_diagnostic"] = {"_doc": "estimated_cue_end is a post-hoc LLM explanation, not a decidability label",
                             "kinds": dict(kinds), "position_in_clause": dict(pos),
                             "clauses_all_votes_same_cue": round(same / max(1, len(Cd)), 4)}
    # alignment + frames
    summ = json.loads((Path(args.frames) / "summary.json").read_text())
    lq = sum(r["low_quality_alignment"] for r in cl)
    rep["alignment"] = {"clauses": len(cl), "low_quality": lq, "low_quality_rate": round(lq / max(1, len(cl)), 4),
                        "matched_frac_hist": dict(sorted(Counter(round(r["matched_frac"], 1) for r in cl).items()))}
    fr = Counter()
    for k, v in summ["frames"].items():
        fr[k.split(":")[1]] += v
    tot = sum(fr.values())
    rep["frames"] = {"reasons_all_views": dict(fr), "reason_rates": {k: round(v / tot, 4) for k, v in fr.items()},
                     "hard_class_counts_by_agent": summ["frames_by_class"], "class_index": summ["class_index"]}
    cls = Counter()
    for r in cl:
        for a, v in r["agent_view"].items():
            cls[v["hard"] if v["mask"] else "masked"] += 1
    rep["clause_class_distribution_all_views"] = dict(cls)
    # gates (a LOCK file with "gates_for_next_step", e.g. LOCK_dev20.json)
    if not args.lock:
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {args.out} (no --lock: gates not evaluated)")
        return
    lock = json.loads(Path(args.lock).read_text())["gates_for_next_step"]
    vc = rep["validity"]["causal"]
    fx = rep.get("fixed_cases", {}).get("causal", {"pass": 0, "n": 0})
    rep["gates"] = {
        "invalid_vote_rate_after_retry": [vc["invalid_after_retry_rate"], vc["invalid_after_retry_rate"] <= lock["invalid_vote_rate_after_retry_max"]],
        "masked_clause_rate": [vc["masked_clause_rate"], vc["masked_clause_rate"] <= lock["masked_clause_rate_max"]],
        "low_quality_alignment_rate": [rep["alignment"]["low_quality_rate"], rep["alignment"]["low_quality_rate"] <= lock["low_quality_alignment_clause_rate_max"]],
        "fixed_cases_causal": [f"{fx['pass']}/{fx['n']}", fx["pass"] >= 10]}
    Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(rep["gates"], indent=1))


if __name__ == "__main__":
    main()
