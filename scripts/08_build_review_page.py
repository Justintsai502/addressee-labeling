#!/usr/bin/env python3
"""Fill scripts/review_page_template.html with <run-dir>/review_items.json -> review page HTML.

The page shows each dialogue act with context and two anonymised candidate labels (AMI human
vs LLM majority, order fixed per item, source revealed after answering) and stores each
reviewer's verdict in the artifact's db under reviews/<uid>/items/<item id>.

    python3 scripts/08_build_review_page.py --run-dir outputs/agent_view_v3/qwen3_32b --out review.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads((Path(args.run_dir) / "review_items.json").read_text(encoding="utf-8"))
    tpl = (Path(__file__).with_name("review_page_template.html")).read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    Path(args.out).write_text(tpl.replace("__REVIEW_DATA__", payload), encoding="utf-8")
    print(f"wrote {args.out}: {len(data['items'])} items ({data['n_disagree']} disagreements)")


if __name__ == "__main__":
    main()
