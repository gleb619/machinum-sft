"""Tests for `scripts/pairs_from_book.py` — the corpus builder, no files outside tmp_path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pairs_from_book.py"


def load_module():
    """Import the script by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("pairs_from_book", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def book(tmp_path: Path) -> Path:
    """One chapter with four aligned paragraphs; the last is too short for a single pair."""
    en = tmp_path / "book"
    en.mkdir()
    lines_en = [
        "The first paragraph is long enough to survive the character filter.",
        "The second paragraph is also long enough to be worth training on.",
        "The third paragraph closes the chapter and is long enough as well.",
        "Short one.",
    ]
    lines_ru = [
        "Первый абзац достаточно длинный, чтобы пройти фильтр по символам.",
        "Второй абзац тоже достаточно длинный для обучения.",
        "Третий абзац закрывает главу и тоже достаточно длинный.",
        "Короткий.",
    ]
    for suffix, lines in (("en", lines_en), ("ru", lines_ru)):
        # chapter file: frontmatter, a heading, then paragraphs
        body = "---\ntitle: x\n---\n# Глава\n\n" + "\n\n".join(lines) + "\n"
        (en / f"chapter-1.{suffix}.md").write_text(body, encoding="utf-8")
    return en


def test_single_line_pairs_skip_short_paragraphs(book: Path) -> None:
    module = load_module()
    records = module.build(book, min_chars=40)
    assert len(records) == 3, "the two-word paragraph is dropped, the headings never counted"
    assert all("\n" not in record["user"] for record in records)
    assert records[0]["source"].endswith("#p0")


def test_groups_emit_windows_with_line_breaks(book: Path) -> None:
    module = load_module()
    records = module.build(book, min_chars=40, groups=(2,))
    grouped = [record for record in records if "\n" in record["user"]]
    assert len(grouped) == 2, "three long paragraphs hold two windows; the short one vetoes its own"
    for record in grouped:
        assert record["user"].count("\n") == 1
        assert record["reply"].count("\n") == 1
        assert record["source"].startswith("book/chapter-1.en.md#p")


def test_cli_writes_jsonl_readable_by_data_build(tmp_path: Path, book: Path) -> None:
    module = load_module()
    out = tmp_path / "corpus.jsonl"
    records = module.build(book, min_chars=40, groups=(2, 3))
    out.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows and all({"system", "user", "reply", "source"} <= set(row) for row in rows)
    assert any(row["user"].count("\n") == 2 for row in rows), "a window of three exists"
