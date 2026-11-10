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


def build(en_dir: Path, min_chars: int, groups: tuple[int, ...] = ()) -> list[dict[str, str]]:
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
        # A model cannot keep a line break it has never seen: windows of consecutive
        # paragraphs teach the shape the system prompt already asks for.
        for size in groups:
            for start in range(0, min(len(left), len(right)) - size + 1):
                window_left = left[start : start + size]
                window_right = right[start : start + size]
                if any(len(line) < min_chars // 2 for line in window_left + window_right):
                    continue
                records.append(
                    {
                        "system": SYSTEM,
                        "user": "\n".join(window_left),
                        "reply": "\n".join(window_right),
                        "source": f"{en.parent.name}/{en.name}#p{start}-{start + size - 1}",
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en", required=True, type=Path, help="directory with *.en.md chapters")
    parser.add_argument("--out", required=True, type=Path, help="JSONL to write")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument(
        "--groups",
        metavar="SIZES",
        default="",
        help="comma-separated window sizes for multi-line samples, e.g. 2,3",
    )
    args = parser.parse_args()

    groups = tuple(int(part) for part in args.groups.split(",") if part.strip())
    records = build(args.en, args.min_chars, groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
