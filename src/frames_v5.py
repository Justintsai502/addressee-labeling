"""ADS v3 frame builder logic (12.5 Hz) — rev 3 §1.5. CPU only, numpy.

Two layers, combined per agent seat:
  activity[seat, T]  energy VAD on the whole per-seat track (independent of ASR / alignment)
  spans              clause time spans from word alignment (alignment never decides activity)
Precedence per frame (agent seat a):
  1 only a active -> self   2 nobody active -> none   3 one human h active inside one of h's clause spans
  -> that clause's soft label (or masked if the clause is masked)   4 human active without a reliable span
  -> masked ("no_span")   5 >= 2 seats active -> follow the INCOMING speaker (user, 2026-10-12): the active
  seat whose current activity run started latest; agent -> self, human -> rules 3/4 for that seat
  (reason "overlap_new" / "overlap_self"); several seats started in the same frame -> masked ("overlap_tie").
Frame -> span: the span covering the largest part of the 80 ms frame; exact tie -> earlier span;
covered < 50% -> rule 4.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FRAME = 0.08
SUB = 0.01                 # VAD sub-frame
VAD_RMS = 1e-3             # per-seat TTS tracks are digital silence outside speech (CPU check 2026-10-11)
VAD_FILL_GAP = 0.20        # fill silences shorter than this inside speech
FRAME_ACTIVE_FRAC = 0.5    # frame active if >= 50% of its sub-frames are active
CLAUSE_MATCH_FRAC = 0.6    # clause aligned if >= 60% of its own words matched
CLASS_IDX = {"none": 0, "self": 1, "to_agent_itself": 2, "to_everyone": 3, "not_to_agent": 4}

_TOK = re.compile(r"[A-Za-z0-9']+")


def norm_tokens(text: str) -> List[Tuple[str, int, int]]:
    """Lowercased word tokens with char offsets (brackets/punctuation dropped)."""
    return [(m.group(0).lower().replace("'", ""), m.start(), m.end()) for m in _TOK.finditer(text)]


def vad_subframes(x: np.ndarray, sr: int) -> np.ndarray:
    hop = int(round(sr * SUB))
    n = len(x) // hop
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1))
    act = e > VAD_RMS
    gap = int(round(VAD_FILL_GAP / SUB))
    idx = np.nonzero(act)[0]
    for a, b in zip(idx, idx[1:]):
        if 1 < b - a <= gap:
            act[a:b] = True
    return act


def frames_from_sub(act: np.ndarray, n_frames: int) -> np.ndarray:
    per = int(round(FRAME / SUB))
    pad = np.zeros(n_frames * per, bool)
    m = min(len(act), len(pad))
    pad[:m] = act[:m]
    return pad.reshape(n_frames, per).mean(1) >= FRAME_ACTIVE_FRAC


def match_words(script_toks: Sequence[str], asr_toks: Sequence[str]) -> Dict[int, int]:
    """script token index -> asr token index (difflib matching blocks, no autojunk)."""
    sm = difflib.SequenceMatcher(None, list(script_toks), list(asr_toks), autojunk=False)
    m = {}
    for b in sm.get_matching_blocks():
        for k in range(b.size):
            m[b.a + k] = b.b + k
    return m


def clause_spans(utts: List[Dict], asr_words: List[Tuple[float, float, str]]) -> List[Dict]:
    """utts = this seat's turns in order: {turn_id, text, start, end, clauses:[(char_start, char_end)]}.
    Returns one dict per clause: {turn_id, clause_idx, t0, t1, low_quality, matched_frac, word_times}."""
    stoks, owner = [], []  # owner = (utt index, char start, char end)
    for ui, u in enumerate(utts):
        for tok, a, b in norm_tokens(u["text"]):
            stoks.append(tok)
            owner.append((ui, a, b))
    atoks, a_owner = [], []  # an ASR word can hold several tokens ("check-in")
    for wi, w in enumerate(asr_words):
        for tok, _, _ in norm_tokens(w[2]):
            atoks.append(tok)
            a_owner.append(wi)
    m = {i: a_owner[j] for i, j in match_words(stoks, atoks).items()}  # script token -> ASR word index
    out = []
    for ui, u in enumerate(utts):
        idx = [i for i, o in enumerate(owner) if o[0] == ui]
        per_clause = []
        for ci, (ca, cb) in enumerate(u["clauses"]):
            ws = [i for i in idx if ca <= owner[i][1] < cb]
            hit = [i for i in ws if i in m]
            frac = len(hit) / len(ws) if ws else 0.0
            times = {owner[i][1]: (asr_words[m[i]][0], asr_words[m[i]][1]) for i in hit}  # char start -> (t0, t1)
            ok = bool(hit) and frac >= CLAUSE_MATCH_FRAC
            per_clause.append({"turn_id": u["turn_id"], "clause_idx": ci, "matched_frac": round(frac, 3),
                               "t0": asr_words[m[hit[0]]][0] if ok else None,
                               "t1": asr_words[m[hit[-1]]][1] if ok else None,
                               "low_quality": not ok, "word_times": times})
        if all(not c["low_quality"] for c in per_clause):
            # contiguous inside the utterance: boundary at the midpoint of the gap between clauses
            for c, d in zip(per_clause, per_clause[1:]):
                if d["t0"] > c["t1"]:
                    mid = (c["t1"] + d["t0"]) / 2
                    c["t1"], d["t0"] = mid, mid
        else:
            # fallback: character-proportional split of the utterance window; all clauses flagged
            hit_all = [i for i in idx if i in m]
            if hit_all:
                w0, w1 = asr_words[m[hit_all[0]]][0], asr_words[m[hit_all[-1]]][1]
            else:
                w0, w1 = u["start"], u["end"]
            n = max(1, len(u["text"]))
            for c, (ca, cb) in zip(per_clause, u["clauses"]):
                c["t0"], c["t1"] = w0 + (w1 - w0) * ca / n, w0 + (w1 - w0) * cb / n
                c["low_quality"] = True
        out.extend(per_clause)
    return out


def assign_span(frame_i: int, spans: List[Tuple[float, float, int]]) -> Optional[int]:
    """spans = [(t0, t1, key)] in time order; -> key of the span covering most of the frame (>= 50%), ties -> earlier."""
    f0, f1 = frame_i * FRAME, (frame_i + 1) * FRAME
    best, best_cov = None, 0.0
    for t0, t1, key in spans:
        cov = max(0.0, min(f1, t1) - max(f0, t0))
        if cov > best_cov + 1e-9:
            best, best_cov = key, cov
    return best if best_cov >= FRAME / 2 - 1e-9 else None


def build_view(active: np.ndarray, seats: Sequence[str], agent: str, spans_by_seat: Dict[str, List],
               clause_label: Dict, n_frames: int):
    """active[s_index, T]; clause_label[key] = {"mask": 0/1, "soft": {cls: p} or None} for agent `agent`.
    -> soft[T, 5] float32, mask[T] uint8, reason[T] (str codes)."""
    soft = np.zeros((n_frames, 5), np.float32)
    mask = np.ones(n_frames, np.uint8)
    reason = np.array([""] * n_frames, dtype=object)
    ai = seats.index(agent)
    onset = run_onsets(active)
    for i in range(n_frames):
        on = [k for k in range(len(seats)) if active[k, i]]
        if not on:
            soft[i, CLASS_IDX["none"]] = 1
            reason[i] = "none"
            continue
        tag = ""
        if len(on) >= 2:
            latest = max(onset[k, i] for k in on)
            new = [k for k in on if onset[k, i] == latest]
            if len(new) > 1:
                mask[i], reason[i] = 0, "overlap_tie"
                continue
            on, tag = new, "overlap_"
        if on[0] == ai:
            soft[i, CLASS_IDX["self"]] = 1
            reason[i] = tag + "self" if tag else "self"
            continue
        key = assign_span(i, spans_by_seat[seats[on[0]]])
        if key is None:
            mask[i], reason[i] = 0, tag + "no_span"
            continue
        lab = clause_label[key]
        if not lab["mask"]:
            mask[i], reason[i] = 0, tag + "masked_clause"
            continue
        for c, p in lab["soft"].items():
            soft[i, CLASS_IDX[c]] = p
        reason[i] = (tag + "new") if tag else ("speech_lq" if lab.get("low_quality") else "speech")
    return soft, mask, reason


def run_onsets(active: np.ndarray) -> np.ndarray:
    """onset[k, i] = first frame of seat k's current activity run (-1 when inactive)."""
    onset = np.full(active.shape, -1, np.int64)
    for k in range(active.shape[0]):
        start = -1
        for i in range(active.shape[1]):
            if active[k, i]:
                if i == 0 or not active[k, i - 1]:
                    start = i
                onset[k, i] = start
    return onset
