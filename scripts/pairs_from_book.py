#!/usr/bin/env python3
"""Build paragraph-level EN→RU triples from the pipeline's human-parallel chapters.

Deliberately dumb: split by non-empty line, drop headings, pair by index where both sides
exist and are long enough. The machinum-pipeline TL255 tool does the same with alignment
scoring; this exists so a first dataset can be produced without a pipeline run.

    python scripts/pairs_from_book.py \
        --en ../machinum-pipeline/books/book1 \
        --out data/raw/book1-parallel.jsonl

The output is JSONL of {system, user, reply, source}, which `msft data build` consumes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SYSTEM = (
    "You translate English literary prose into Russian. "
    "Reply with the translation only, one line per input line. "
    "Keep names, dialogue dashes and paragraph breaks exactly as in the source."
)
MIN_CHARS = 40


def paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^---.*?---", "", text, flags=re.S)
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def build(en_dir: Path, min_chars: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for en in sorted(en_dir.glob("*.en.md")):
        ru = Path(str(en).replace(".en.md", ".ru.md"))
        if not ru.exists():
            continue
        left, right = paragraphs(en), paragraphs(ru)
        if min(len(left), len(right)) < 3:
            continue
        for index in range(min(len(left), len(right))):
            if len(left[index]) < min_chars or len(right[index]) < min_chars:
                continue
            records.append(
                {
                    "system": SYSTEM,
                    "user": left[index],
                    "reply": right[index],
                    "source": f"{en.parent.name}/{en.name}#p{index}",
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en", required=True, type=Path, help="directory with *.en.md chapters")
    parser.add_argument("--out", required=True, type=Path, help="JSONL to write")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = parser.parse_args()

    records = build(args.en, args.min_chars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
