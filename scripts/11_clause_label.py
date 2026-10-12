#!/usr/bin/env python3
"""Step 2 of the ADS v3 pipeline: clause-level addressee votes with prompt v5 (see docs/ADS_V3_PIPELINE.md).

One request per (clause, vote slot). Each request has its own seed, derived from a hash of
(base seed, conversation, turn, clause, round, slot), so results do not depend on batch order and a resumed run
gives the same seeds. Invalid votes (unparsable / illegal_name / self_address / missing_target / unknown) are
re-sampled once; the original and the replacement are both kept and a slot counts at most one vote.
Optional greedy pass (k = 1, T = 0) for reporting only.

Conversations are processed in chunks; rows are appended to clause_votes.jsonl after each chunk, and --resume
skips conversations already in that file.

    # CPU dry run (no model)
    python3 scripts/11_clause_label.py --convs data/ttv2/convs_pilot.jsonl --mode causal --backend mock \
        --out-dir outputs/ads_v3_v5/mock_causal
    # GPU (vLLM env)
    python scripts/11_clause_label.py --convs data/ttv2/convs_pilot.jsonl --fixed data/ttv2/fixed_cases_v5.json \
        --mode causal --greedy --out-dir outputs/ads_v3_v5/dev20_causal

Writes <out>/clause_votes.jsonl (one row per clause), <out>/run.json, <out>/system_prompt.txt.
Seed rule history: the dev20 run (EXP-062, 2026-10-11) used `seed + 10*index + slot` (retry + 100000);
hashed seeds since 2026-10-12 (the old rule collides above 10,000 clauses).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from src.clause_v5 import (PROMPTS_V5, majority_norm, parse_vote, render_causal, render_full, split_clauses)
from src.io_utils import load_conversations

ANSWER_KEEP = 2000


def request_seed(base: int, it: dict, rnd: str, slot: int) -> int:
    key = f"{base}|{it['conversation_id']}|{it['turn_id']}|{it['clause_idx']}|{rnd}|{slot}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def _items_for(cid, speakers, turns, context_turns, turn_ids=None):
    items = []
    for ti, (spk, text) in enumerate(turns):
        cls = split_clauses(text, speakers)
        for c in cls:
            items.append({"conversation_id": cid, "turn_id": turn_ids[ti] if turn_ids else ti, "turn_index": ti,
                          "speaker": spk, "speakers": list(speakers), "clause_idx": c.idx, "n_clauses": len(cls),
                          "char_start": c.start, "char_end": c.end, "clause": c.text, "turn_text": text,
                          "user": {"causal": render_causal(speakers, turns, ti, cls, c.idx, context_turns),
                                   "full": render_full(speakers, turns, ti, cls, c.idx)}})
    return items


def conv_items(conv, context_turns):
    return _items_for(conv.conversation_id, conv.speakers, [(t.speaker, t.text) for t in conv.turns],
                      context_turns, [t.turn_id for t in conv.turns])


def fixed_items(path, context_turns):
    """Fixed boundary cases: only the LAST turn of each case is labelled."""
    fc = json.loads(Path(path).read_text())
    items = []
    for case in fc["cases"]:
        turns = [tuple(x) for x in case["turns"]]
        last = len(turns) - 1
        items += [it for it in _items_for(f"fixed:{case['case_id']}", fc["speakers"], turns, context_turns)
                  if it["turn_index"] == last]
    return items


class MockEngine:
    """CPU stand-in: answers with a random participant / GROUP, sometimes garbage, to exercise the pipeline."""

    def generate(self, reqs):
        outs = []
        for it, seed, temp in reqs:
            rng = random.Random(seed)
            r = rng.random()
            others = [s for s in it["speakers"] if s != it["speaker"]]
            if r < 0.08:
                txt = "<think>hmm</think> not json"
            elif r < 0.12:
                txt = '{"addressees": ["UNKNOWN"], "cue": "none_needed", "confidence": 0.2}'
            else:
                a = ["GROUP"] if rng.random() < 0.4 else [rng.choice(others)]
                txt = json.dumps({"addressees": a, "cue": it["clause"].split()[0], "confidence": 0.7})
            outs.append((txt, 0, 0))
        return outs


class VllmEngine:
    def __init__(self, model, max_model_len, max_new_tokens, enable_thinking, system, gpu_mem, tp):
        from vllm import LLM

        self.llm = LLM(model=model, max_model_len=max_model_len, enable_prefix_caching=True, trust_remote_code=True,
                       gpu_memory_utilization=gpu_mem, tensor_parallel_size=tp)
        self.max_new_tokens, self.enable_thinking, self.system = max_new_tokens, enable_thinking, system

    def generate(self, reqs):
        from vllm import SamplingParams

        msgs = [[{"role": "system", "content": self.system}, {"role": "user", "content": u}] for u, _, _ in reqs]
        sps = [SamplingParams(n=1, temperature=temp, seed=seed, max_tokens=self.max_new_tokens) for _, seed, temp in reqs]
        outs = self.llm.chat(msgs, sps, use_tqdm=True, chat_template_kwargs={"enable_thinking": self.enable_thinking})
        return [(o.outputs[0].text, len(o.prompt_token_ids or []), len(o.outputs[0].token_ids or [])) for o in outs]


def split_answer(text):
    i = text.rfind("</think>")
    if i < 0:
        return text.strip(), 0
    return text[i + len("</think>"):].strip(), i


def run_round(engine, mode, reqs, temp, backend):
    """reqs = [(item, seed)] -> one parsed vote per request."""
    outs = engine.generate([((it if backend == "mock" else it["user"][mode]), seed, temp) for it, seed in reqs])
    res = []
    for (it, seed), (txt, n_in, n_out) in zip(reqs, outs):
        ans, think_chars = split_answer(txt)
        v = parse_vote(txt, it["speakers"], it["speaker"])
        res.append({"seed": seed, "answer": ans[:ANSWER_KEEP], "think_chars": think_chars, "in_tok": n_in,
                    "out_tok": n_out, **v})
    return res


def label_chunk(engine, items, args):
    """K slots per clause, one retry round for invalid slots, optional greedy. -> (rows, stats)."""
    k = args.k
    r1 = run_round(engine, args.mode, [(it, request_seed(args.seed, it, "r1", s)) for it in items for s in range(k)],
                   args.temperature, args.backend)
    slots = [r1[i * k:(i + 1) * k] for i in range(len(items))]
    retry = [(i, s) for i in range(len(items)) for s in range(k) if slots[i][s]["norm"] is None]
    if retry:
        r2 = run_round(engine, args.mode, [(items[i], request_seed(args.seed, items[i], "retry", s)) for i, s in retry],
                       args.temperature, args.backend)
        for (i, s), v in zip(retry, r2):
            slots[i][s] = {**slots[i][s], "retry": v}
    greedy = run_round(engine, args.mode, [(it, args.seed) for it in items], 0.0, args.backend) if args.greedy else None
    rows, tok_in, tok_out = [], 0, 0
    for i, it in enumerate(items):
        finals = []
        for v in slots[i]:
            fin = v.get("retry", v) if v["norm"] is None else v
            finals.append({"norm": fin["norm"], "cue": fin["cue"], "confidence": fin["confidence"],
                           "from_retry": v["norm"] is None})
            for x in (v, v.get("retry")):
                if x:
                    tok_in += x["in_tok"]
                    tok_out += x["out_tok"]
        maj, maj_n = majority_norm([x["norm"] for x in finals])
        row = {key: it[key] for key in it if key != "user"}
        row.update({"mode": args.mode, "slots": slots[i], "final": finals,
                    "valid_vote_count": sum(x["norm"] is not None for x in finals),
                    "majority": maj, "majority_count": maj_n})
        if greedy:
            g = greedy[i]
            row["greedy"] = {key: g[key] for key in ("norm", "fail", "cue", "confidence", "answer")}
            tok_in += g["in_tok"]
            tok_out += g["out_tok"]
        rows.append(row)
    return rows, {"requests_round1": len(r1), "retry": len(retry), "greedy": len(items) if greedy else 0,
                  "input_tokens": tok_in, "output_tokens": tok_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", required=True, help="conversations JSONL (scripts/00c_prepare_ttv2.py)")
    ap.add_argument("--fixed", default=None, help="optional fixed boundary cases JSON (labelled first)")
    ap.add_argument("--mode", choices=("causal", "full"), default="causal",
                    help="causal = training labels; full = sees later turns (comparison only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backend", choices=("vllm", "mock"), default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--greedy", action="store_true", help="also one T=0 answer per clause (reporting only)")
    ap.add_argument("--context-turns", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--chunk-convs", type=int, default=50, help="conversations per vLLM batch / per append")
    ap.add_argument("--resume", action="store_true", help="skip conversations already in clause_votes.jsonl")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    votes_path = out / "clause_votes.jsonl"
    done = set()
    if args.resume and votes_path.exists():
        done = {json.loads(l)["conversation_id"] for l in open(votes_path, encoding="utf-8") if l.strip()}
    elif votes_path.exists():
        sys.exit(f"{votes_path} exists; use --resume or a new --out-dir")
    system = PROMPTS_V5[args.mode]
    convs = [c for c in load_conversations(args.convs) if c.conversation_id not in done]
    chunks = []
    if args.fixed and not any(d.startswith("fixed:") for d in done):
        chunks.append(fixed_items(args.fixed, args.context_turns))
    for i in range(0, len(convs), args.chunk_convs):
        chunks.append([it for c in convs[i:i + args.chunk_convs] for it in conv_items(c, args.context_turns)])
    n_items = sum(len(c) for c in chunks)
    print(f"{len(convs)} conversations to do ({len(done)} already done), {n_items} clauses, mode={args.mode}, "
          f"k={args.k}, {len(chunks)} chunk(s)", file=sys.stderr, flush=True)
    if n_items == 0:
        print("nothing to do", file=sys.stderr)
        return
    t0 = time.time()
    engine = MockEngine() if args.backend == "mock" else VllmEngine(
        args.model, args.max_model_len, args.max_new_tokens, not args.no_think, system,
        args.gpu_memory_utilization, args.tensor_parallel_size)
    t_load = time.time() - t0
    stats = {"requests_round1": 0, "retry": 0, "greedy": 0, "input_tokens": 0, "output_tokens": 0}
    for ci, items in enumerate(chunks, 1):
        if not items:
            continue
        rows, st = label_chunk(engine, items, args)
        with open(votes_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for key in stats:
            stats[key] += st[key]
        print(f"[chunk {ci}/{len(chunks)}] {len(items)} clauses, {st['retry']} retried "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    run = {"args": vars(args), "n_clauses_this_run": n_items, **stats, "load_seconds": t_load,
           "generate_seconds": time.time() - t0 - t_load, "resumed_from": len(done),
           "seed_rule": "sha256(base|conversation|turn|clause|round|slot)[:8]",
           "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
           "convs_sha256": hashlib.sha256(Path(args.convs).read_bytes()).hexdigest(),
           "fixed_sha256": hashlib.sha256(Path(args.fixed).read_bytes()).hexdigest() if args.fixed else None}
    try:
        import vllm
        run["vllm_version"] = vllm.__version__
    except Exception:
        run["vllm_version"] = None
    name = "run.json" if not done else f"run_resume_{int(time.time())}.json"
    (out / name).write_text(json.dumps(run, indent=2) + "\n")
    (out / "system_prompt.txt").write_text(system)
    print(f"wrote {votes_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
