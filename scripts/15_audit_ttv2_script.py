#!/usr/bin/env python3
"""List generation errors in ttv2 scripts (text/label level; no audio needed).

    python3 scripts/15_audit_ttv2_script.py --root /path/to/turn_taking_2000_v2 \
        --out-csv data/ttv2/script_issues.csv --out-json data/ttv2/script_issues_summary.json

Issue types (one CSV row per utterance and type):
  letter_placeholder   text has a seat letter in brackets ("[C], do you ...") instead of a name
  self_vocative        speaker opens by calling their own name ("Emil: Emil, did you ...")
  stage_direction      "[to C]" / "[to me]" written into the spoken text (TTS reads it out)
  other_bracket        other bracketed tokens that are not a participant name ("[sighs]", "[EVERYONE]", ...)
  bare_seat_letter     seat letters used as names without brackets ("A and B tackle ...")
  named_label_on_group_line  addressing = one person, but the text addresses the group and never names them
  canonical_name_leftover  a canonical template name (names.json "canonical", e.g. Alex/Brian/Casey/Dana) that is not a
                       participant survives in the text, or a garbled one in brackets ("[BRAN]")
  name_placeholder     participant name kept in brackets ("[Jerome]", "[MABEL]") - text-only, TTS reads the name
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

GROUP_WORDS = re.compile(r"\b(guys|everyone|everybody|you all|y'all|all of you|team|folks)\b", re.I)


def load(d: Path):
    rows = [json.loads(l) for l in open(d / "script.jsonl", encoding="utf-8") if l.strip()]
    names = rows[0].get("display_names") or json.loads((d / "names.json").read_text()).get("display_names")
    canon = []
    if (d / "script_canonical.jsonl").exists():
        canon = [json.loads(l) for l in open(d / "script_canonical.jsonl", encoding="utf-8") if l.strip()]
    canon_names = json.loads((d / "names.json").read_text()).get("canonical", {}) if (d / "names.json").exists() else {}
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return rows, names, canon, meta.get("setup", {}).get("source_folder", ""), canon_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "script.jsonl").exists())
    out = []
    for d in dirs:
        rows, names, canon, src, canon_names = load(d)
        seat_of = {v.lower(): k for k, v in names.items()}
        leftovers = [n for n in canon_names.values() if n not in names.values()]
        for i, u in enumerate(rows):
            if u.get("function", "speech") not in ("speech", "continuer", None) and not u.get("text"):
                continue
            txt, spk = u.get("text") or "", u["speaker"]
            nxt = rows[i + 1]["speaker"] if i + 1 < len(rows) else ""
            base = {"dialogue_id": d.name, "source_folder": src, "utt_id": u.get("utt_id", i), "index": i,
                    "speaker_seat": spk, "speaker_name": names.get(spk, spk), "text": txt,
                    "addressing": "|".join(u.get("addressing") or []), "next_speaker_seat": nxt,
                    "canonical_text": canon[i]["text"] if i < len(canon) else ""}

            def add(kind, detail, suggestion):
                out.append({**base, "issue": kind, "detail": detail, "suggested_fix": suggestion})

            for m in re.finditer(r"\[([A-D])\]", txt):
                s = m.group(1)
                add("letter_placeholder", f"[{s}] = {names.get(s, '?')}" + (" (speaks next)" if nxt == s else ""),
                    f"replace [{s}] with {names.get(s, '?')}")
            for m in re.finditer(r"\[(to [^\]]+)\]", txt, re.I):
                add("stage_direction", m.group(0), "remove the bracket from the spoken text; keep it as metadata")
            for m in re.finditer(r"\[([^\]]+)\]", txt):
                tok = m.group(1)
                if re.fullmatch(r"[A-D]", tok) or tok.lower().startswith("to "):
                    continue
                if tok.lower() in seat_of:
                    if tok.lower() != names.get(spk, "").lower():
                        add("name_placeholder", f"[{tok}]", "drop the brackets (TTS already reads the name)")
                elif any(tok.lower()[:3] == n.lower()[:3] for n in leftovers):
                    add("canonical_name_leftover", f"[{tok}] looks like a canonical name "
                        f"({', '.join(leftovers)}) that was not remapped", "replace with the participant's display name")
                else:
                    add("other_bracket", f"[{tok}]", "remove, or replace with the intended word/name")
            for n in leftovers:
                if re.search(rf"\b{n}\b", txt):
                    add("canonical_name_leftover", f"{n} (canonical name, not a participant)",
                        "replace with the participant's display name")
            own = re.escape(names.get(spk, "\0"))
            if re.match(rf"^\W*(?:\w+\W+){{0,4}}?\[?{own}\]?\s*[,!?]", txt) or re.search(rf"[,.]\s*\[?{own}\]?\s*[,?]", txt):
                who = "speaker answers own question next" if nxt == spk else ""
                add("self_vocative", who, "the vocative names the speaker; fix the speaker or the name "
                    "(and set addressing to the person called)")
            if re.search(r"\b[A-D] and [A-D]\b|\b(?:and|with|while) [B-D]\b|\b[B-D] and I\b", txt):
                add("bare_seat_letter", "", "replace seat letters with names")
            adr = u.get("addressing") or []
            if len(adr) == 1 and adr[0] in names and GROUP_WORDS.search(txt) and names[adr[0]] not in txt:
                add("named_label_on_group_line", f"label {adr[0]}={names[adr[0]]}, text says "
                    f"'{GROUP_WORDS.search(txt).group(0)}'", "label EVERYONE (or name the person in the text)")
    out.sort(key=lambda r: (r["issue"], r["dialogue_id"], r["index"]))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = ["issue", "dialogue_id", "utt_id", "index", "speaker_seat", "speaker_name", "text", "addressing",
            "next_speaker_seat", "detail", "suggested_fix", "canonical_text", "source_folder"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    by = Counter(r["issue"] for r in out)
    summ = {"n_dialogues": len(dirs), "n_utterances": sum(1 for d in dirs for _ in open(d / "script.jsonl")),
            "issues": {k: {"rows": v, "dialogues": len({r["dialogue_id"] for r in out if r["issue"] == k})}
                       for k, v in sorted(by.items())},
            "dialogues_with_any_issue_excluding_name_placeholder": len(
                {r["dialogue_id"] for r in out if r["issue"] != "name_placeholder"})}
    Path(args.out_json).write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
