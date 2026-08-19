#!/usr/bin/env python3
"""Patch flashinfer's import-time TypeError on Python < 3.13.

Symptom (kills vLLM before the engine starts):

    File ".../flashinfer/comm/fd_exchange.py", line 55, in <module>
        def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
    TypeError: type 'array.array' is not subscriptable

Cause: `array.array[int]` in a type annotation is only valid from Python 3.13,
where array.array gained __class_getitem__. On 3.11/3.12 the annotation is
evaluated at import time and raises.

Fix: insert `from __future__ import annotations` at the top of the offending
module, which makes ALL annotations in that file lazy strings (PEP 563) so the
subscript is never evaluated. Annotation-only change — no runtime behaviour is
altered.

    python3 scripts/fix_vllm_flashinfer.py            # patch
    python3 scripts/fix_vllm_flashinfer.py --check    # report only
    python3 scripts/fix_vllm_flashinfer.py --revert   # undo (restores .bak)

Prefer upgrading flashinfer first (`pip install -U flashinfer-python`); use this
when no fixed release is available for your CUDA/torch combination.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FUTURE = "from __future__ import annotations"


def find_targets() -> list[Path]:
    try:
        import flashinfer
    except Exception as e:
        print(f"cannot import flashinfer ({type(e).__name__}); is it installed?")
        # Even if importing the package fails, we may still locate the files.
        import importlib.util
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or not spec.submodule_search_locations:
            return []
        root = Path(list(spec.submodule_search_locations)[0])
    else:
        root = Path(flashinfer.__file__).parent
    return sorted(root.rglob("*.py"))


def needs_patch(text: str) -> bool:
    return "array.array[" in text and FUTURE not in text


def insert_future(text: str) -> str:
    """Insert the future import after any shebang/encoding/docstring."""
    lines = text.splitlines(keepends=True)
    i = 0
    # skip shebang and encoding lines
    while i < len(lines) and (lines[i].startswith("#!") or "coding" in lines[i][:30]):
        i += 1
    # skip a module docstring if present
    stripped = "".join(lines[i:]).lstrip()
    if stripped.startswith(('"""', "'''")):
        quote = stripped[:3]
        # find the closing quote past the opening one
        rest = "".join(lines[i:])
        start = rest.index(quote)
        end = rest.index(quote, start + 3) + 3
        consumed = rest[:end].count("\n") + 1
        i += consumed
    lines.insert(i, FUTURE + "\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report, don't modify")
    ap.add_argument("--revert", action="store_true", help="restore .bak files")
    args = ap.parse_args()

    files = find_targets()
    if not files:
        print("flashinfer not found — nothing to do.")
        sys.exit(1)

    if args.revert:
        n = 0
        for f in files:
            bak = f.with_suffix(f.suffix + ".bak")
            if bak.exists():
                shutil.move(str(bak), str(f))
                print(f"reverted {f}")
                n += 1
        print(f"{n} file(s) reverted.")
        return

    hits = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if needs_patch(text):
            hits.append((f, text))

    if not hits:
        print("no files need patching (either already patched, or your "
              "flashinfer version doesn't have this bug).")
        return

    print(f"{len(hits)} file(s) affected:")
    for f, _ in hits:
        print(f"  {f}")
    if args.check:
        print("\n--check given; nothing modified.")
        return

    for f, text in hits:
        shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))
        f.write_text(insert_future(text), encoding="utf-8")
        print(f"patched {f}  (backup: {f.name}.bak)")
    print("\nDone. Verify with:  python3 -c \"import flashinfer; print('ok')\"")


if __name__ == "__main__":
    main()
