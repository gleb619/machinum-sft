#!/usr/bin/env python3
"""Audit a translated chapter set against its source, without trusting the pipeline.

    python3 scripts/audit_translation.py --src chapters/en --dst chapters/ru
    python3 scripts/audit_translation.py --src chapters/en --dst chapters/ru --json out.json

Checks, per chapter and in total:
  * line parity          — the one invariant the pipeline promises
  * metadata leaks       — `<terms>` / `<term>` / ` thinking` blocks that reached the prose
  * foreign characters   — anything outside Cyrillic, Latin, digits and punctuation; this is how
                           a Q4 tail emits «Корمאك» and how a glossary block slips in
  * empty lines in place — a line that came back blank where the source had text
  * length ratio         — Russian runs ~10-15 % longer than English; a chapter far outside that
                           is a truncation or a repeated paragraph

Exit code: 0 clean, 1 defects found, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

CYRILLIC = re.compile(r"[\u0400-\u04FF]")
LATIN = re.compile(r"[A-Za-z]")
LEAK = re.compile(r"<terms>|<term |</term>|</think>")
ALLOWED_EXTRA = set(" \t\u00a0«»„“”\"'’‘—–-….,:;!?()[]{}/*&%№#@+=<>|~^$\\/\n")


def suspicious(text: str) -> list[str]:
    """Characters that have no business in a Russian literary chapter."""
    out: list[str] = []
    for ch in text:
        if ch in ALLOWED_EXTRA or CYRILLIC.match(ch) or LATIN.match(ch) or ch.isdigit():
            continue
        cat = unicodedata.category(ch)
        # spaces and combining marks: «копа́й» carries a stress accent, which is prose-level
        # style, not a decoding fault
        if cat in {"Zs", "Zl", "Zp"} or cat.startswith("M"):
            continue
        out.append(ch)
    return sorted(set(out))


def audit_chapter(src: Path, dst: Path) -> dict:
    en = src.read_text(encoding="utf-8")
    ru = dst.read_text(encoding="utf-8")
    en_lines = en.split("\n")
    ru_lines = ru.split("\n")
    body = [i for i, line in enumerate(en_lines) if line.strip() and not line.startswith("#")]
    comparable = len(ru_lines) >= len(en_lines)
    blanks = [i for i in body if not ru_lines[i].strip()] if comparable else body
    return {
        "chapter": src.stem,
        "lines_src": sum(1 for line in en_lines if line.strip()),
        "lines_dst": sum(1 for line in ru_lines if line.strip()),
        "parity": sum(1 for line in en_lines if line.strip())
        == sum(1 for line in ru_lines if line.strip()),
        "leaks": len(LEAK.findall(ru)),
        "foreign": "".join(suspicious(ru)),
        "empty_in_place": len(blanks),
        "ratio": round(len(ru) / max(1, len(en)), 3),
        "cyrillic_share": round(len(CYRILLIC.findall(ru)) / max(1, len(ru)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if not args.src.is_dir() or not args.dst.is_dir():
        print("both --src and --dst must be directories", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for src in sorted(args.src.glob("*.md")):
        dst = args.dst / src.name
        if not dst.exists():
            rows.append({"chapter": src.stem, "missing": True})
            continue
        rows.append(audit_chapter(src, dst))

    done = [r for r in rows if not r.get("missing")]
    bad_parity = [r["chapter"] for r in done if not r["parity"]]
    leaks = [r["chapter"] for r in done if r["leaks"]]
    foreign = {r["chapter"]: r["foreign"] for r in done if r["foreign"]}
    empties = [r["chapter"] for r in done if r["empty_in_place"]]
    missing = [r["chapter"] for r in rows if r.get("missing")]
    ratios = sorted(r["ratio"] for r in done)
    median = ratios[len(ratios) // 2] if ratios else 0

    print(f"chapters: {len(done)}/{len(rows)} translated, median length ratio {median}")
    print(f"  parity failures : {len(bad_parity)} {bad_parity if bad_parity else ''}")
    print(f"  metadata leaks  : {len(leaks)} {leaks if leaks else ''}")
    print(f"  empty lines     : {len(empties)} {empties if empties else ''}")
    print(f"  foreign chars   : {len(foreign)} {foreign if foreign else ''}")
    if missing:
        print(f"  missing         : {missing}")
    off = [r for r in done if not (0.85 <= r["ratio"] <= 1.45)]
    if off:
        print(f"  ratio outliers  : {[(r['chapter'], r['ratio']) for r in off]}")

    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")

    defects = bool(bad_parity or leaks or foreign or empties or missing)
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
