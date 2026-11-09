#!/usr/bin/env python3
"""Rebuild a glossary from an editorial name table, and stop pinning non-names.

    python3 scripts/apply_names.py --glossary chapters/glossary.json \\
        --names chapters/names.json [--dry-run]

Why this exists: `runner:glossary` extracts terms with a heuristic, and the heuristic is loose —
on book2 it proposed `March`, `Lock`, `Rain`, `Bring`, `He's`, plus `Rusk Marine` rendered as
«Российский морской пехотинец». Every one of those gets pinned into the prompt by
`tagWithGlossary`, so a bad entry corrupts the translation more reliably than a good one helps.

The table is the source of truth for proper nouns. This pass therefore:

  * writes `ru` for every listed name, including the ones the extractor missed entirely
    (`Kestrel Point`, `Marrow Bay`, …), creating the entry when needed;
  * rewrites possessives (`Fiona's`) with the table's genitive;
  * marks every other entry `ignored: true`, which the pipeline honours — it stays in the file
    as a record but is never injected into a prompt.

Idempotent. Writes atomically.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


def load_table(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, dict] = {}
    for group, entries in raw.items():
        if group.startswith("_") or not isinstance(entries, dict):
            continue
        for en, meta in entries.items():
            table[en] = meta
    return table


def resolve(term: str, table: dict[str, dict]) -> tuple[str, str] | None:
    """(ru, type) for a term, handling possessives and the `the`-less article forms."""
    cleaned = term.strip().rstrip(".,;:")
    if cleaned in table:
        return table[cleaned]["ru"], table[cleaned]["type"]
    for suffix in ("'s", "’s"):
        if cleaned.endswith(suffix):
            base = cleaned[: -len(suffix)]
            if base in table:
                meta = table[base]
                return meta.get("gen") or meta["ru"], meta["type"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary", type=Path, required=True)
    ap.add_argument("--names", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.glossary.read_text(encoding="utf-8"))
    terms: list[dict] = data.setdefault("terms", [])
    table = load_table(args.names)
    by_term = {t["term"]: t for t in terms}

    rewritten, silenced, added = [], [], []

    for t in terms:
        hit = resolve(t["term"], table)
        if hit:
            ru, kind = hit
            if t.get("ru") != ru and not t.get("ignored"):
                rewritten.append((t["term"], t.get("ru"), ru))
            if t.get("ru") != ru:
                t["ru"] = ru
            t["type"] = kind
            t.pop("ignored", None)
        else:
            if not t.get("ignored"):
                silenced.append((t["term"], t.get("ru")))
            t["ignored"] = True

    for en, meta in table.items():
        if en in by_term:
            continue
        terms.append(
            {
                "id": slug(en),
                "term": en,
                "type": meta["type"],
                "count": 0,
                "firstChapter": 0,
                "ru": meta["ru"],
                "aliases": [],
                "references": [],
                "note": "added from names.json: the extractor missed it",
            }
        )
        added.append(en)

    for term, old, new in rewritten:
        print(f"  rewrite  {term:26} {old!r:24} -> {new!r}")
    for term, old in silenced:
        print(f"  ignore   {term:26} {old!r}")
    for term in added:
        print(f"  add      {term:26} -> {table[term]['ru']!r}")
    print(
        f"{len(rewritten)} rewritten, {len(silenced)} ignored, {len(added)} added, "
        f"active terms: {sum(1 for t in terms if not t.get('ignored'))}"
    )

    if not args.dry_run:
        tmp = args.glossary.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.glossary)
        print(f"wrote {args.glossary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
