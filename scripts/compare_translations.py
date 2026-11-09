#!/usr/bin/env python3
"""Compare several translations of the same chapter by measurement, then show them side by side.

    python3 scripts/compare_translations.py --src chapters/en/chapter1.md --dir /tmp/cmp3

Per file: line parity, Cyrillic share, foreign-script characters, name-variant count, mean
sentence length, and the share of sentences that repeat a 6-gram already used earlier in the
chapter (a cheap repetition proxy — quantised models loop).
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

CYR = re.compile(r"[\u0400-\u04FF]")
LAT = re.compile(r"[A-Za-z]")
FOREIGN = re.compile(
    "["
    "\u3040-\u30ff"  # kana
    "\u3400-\u4dbf\u4e00-\u9fff"  # Han
    "\uac00-\ud7af"  # Hangul
    "\u0600-\u06ff\u0590-\u05ff\u0e00-\u0e7f"  # Arabic, Hebrew, Thai
    "]"
)
NAMES = ["Файона", "Фиона", "Доул", "Доули", "Дойл", "Надя", "Надия", "Кормак"]
SENT = re.compile(r"[.!?…]+")


def metrics(text: str) -> dict:
    words = re.findall(r"[\w\u0400-\u04FF'-]+", text)
    sents = [s.strip() for s in SENT.split(text) if len(s.strip()) > 2]
    toks = [w.lower() for w in words]
    grams = [tuple(toks[i : i + 6]) for i in range(max(0, len(toks) - 5))]
    repeats = len(grams) - len(set(grams))
    variants = {}
    for n in NAMES:
        c = text.count(n)
        if c:
            variants[n] = c
    return {
        "chars": len(text),
        "words": len(words),
        "sentences": len(sents),
        "mean_sentence_words": round(len(words) / max(1, len(sents)), 1),
        "cyrillic_share": round(len(CYR.findall(text)) / max(1, len(text)), 3),
        "foreign": "".join(sorted(set(FOREIGN.findall(text)))),
        "latin": "".join(sorted(set(LAT.findall(text)))),
        "names": variants,
        "repeated_6grams": round(100 * repeats / max(1, len(grams)), 1),
        "odd_chars": "".join(
            sorted({c for c in text if unicodedata.category(c).startswith("S") or c in "|^~\\"})
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--show", type=int, default=0, help="print the first N paragraphs of each")
    args = ap.parse_args()

    src_text = args.src.read_text(encoding="utf-8")
    src_lines = [line for line in src_text.split("\n") if line.strip()]
    print(f"source: {len(src_lines)} non-empty lines, {len(src_text)} chars\n")

    files = sorted(args.dir.glob("*.md"))
    rows = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        m = metrics(text)
        m["file"] = f.stem
        m["lines"] = sum(1 for line in text.split("\n") if line.strip())
        m["parity"] = m["lines"] == len(src_lines)
        rows.append(m)

    head = (
        f"{'model':34} {'lines':>5} {'par':>4} {'words':>6} {'sent':>5} "
        f"{'w/s':>5} {'cyr':>5} {'rep%':>5} foreign"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['file'][:34]:34} {r['lines']:>5} {'ok' if r['parity'] else 'NO':>4} "
            f"{r['words']:>6} {r['sentences']:>5} {r['mean_sentence_words']:>5} "
            f"{r['cyrillic_share']:>5} {r['repeated_6grams']:>5} {r['foreign'] or '-'}"
        )
    print()
    for r in rows:
        print(f"{r['file'][:34]:34} names: {r['names']}")
    print()
    for r in rows:
        if r["latin"]:
            print(f"{r['file'][:34]:34} latin kept: {r['latin']}")

    if args.show:
        for f in files:
            body = [line for line in f.read_text(encoding="utf-8").split("\n") if line.strip()]
            print(f"\n=== {f.stem} ===")
            for line in body:
                print(f"  {line[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
