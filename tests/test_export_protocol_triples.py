"""Tests for `scripts/export_protocol_triples.py` — torch-free, no network."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "export_protocol_triples.py"
TL254 = Path(
    "/home/boris/WORKSPACE/machinum-pipeline/packages/tools/dist/modification/sft-triples.js"
)
NODE = shutil.which("node")


def load_module():
    """Import the script by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location("export_protocol_triples", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    """A miniature prompt set with distinct blocks and both slot kinds."""
    root = tmp_path / "prompts"
    root.mkdir()
    files = {
        "summary.system.md": "SUMMARY SYSTEM",
        "llm.format.plain.md": "PLAIN FORMAT",
        "summary.output.bullets.md": "BULLET CONTRACT",
        "summary.user.md": "Give {{count}} notes.\n<context>\n{{text}}\n</context>\n"
        "count={{count}}",
        "glossary.system.md": "GLOSSARY SYSTEM",
        "glossary.user.md": "known={{known}} candidates={{candidates}} text={{text}}",
    }
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8")
    return root


def test_summary_prompt_layout_matches_the_pipeline(prompts: Path) -> None:
    module = load_module()
    system, user = module.build_prompts("summary", prompts, "CHAPTER TEXT")
    assert system == "SUMMARY SYSTEM\n\nPLAIN FORMAT\n\nBULLET CONTRACT"
    assert user == "Give 10 notes.\n<context>\nCHAPTER TEXT\n</context>\ncount=10"


def test_ner_prompt_layout_uses_none_lists(prompts: Path) -> None:
    module = load_module()
    system, user = module.build_prompts("ner", prompts, "CHAPTER TEXT")
    assert system == "GLOSSARY SYSTEM"
    assert user == "known=none candidates=none text=CHAPTER TEXT"


def test_render_leaves_unmatched_slots_untouched() -> None:
    module = load_module()
    assert module.render("a {{x}} b {{y}}", {"x": "1"}) == "a 1 b {{y}}"


def test_empty_and_prompt_echo_replies_are_rejected(prompts: Path) -> None:
    module = load_module()
    _system, user = module.build_prompts("summary", prompts, "CHAPTER TEXT")

    ok, reason = module.validate_reply("summary", "", user)
    assert not ok and "empty" in reason

    echoed = user.replace(" ", "\n").upper()
    ok, reason = module.validate_reply("summary", echoed, user)
    assert not ok and "prompt echo" in reason


def test_good_replies_pass() -> None:
    module = load_module()
    summary = "- one fact\n- two facts\n- three facts\n"
    ok, reason = module.validate_reply("summary", summary, "chapter text")
    assert ok, reason

    ner = json.dumps(
        [
            {
                "name": "Kael",
                "category": "Character",
                "description": "A villager.",
                "references": [],
            }
        ]
    )
    ok, reason = module.validate_reply("ner", ner, "chapter text")
    assert ok, reason


def test_fenced_ner_reply_is_cleaned_before_validation() -> None:
    module = load_module()
    fenced = '```json\n[{"name": "Kael", "category": "Character"}]\n```'
    cleaned = module.clean_reply("ner", fenced)
    assert cleaned == '[{"name": "Kael", "category": "Character"}]'
    ok, reason = module.validate_reply("ner", cleaned, "chapter text")
    assert ok, reason


@pytest.mark.parametrize(
    ("protocol", "reply", "needle"),
    [
        ("summary", "- only one line", "bullet lines"),
        ("ner", "not json", "invalid JSON"),
        ("ner", '{"name": "Kael"}', "not a JSON array"),
        ("ner", '[{"category": "Character"}]', "no name"),
        ("ner", "[]", "empty JSON array"),
    ],
)
def test_malformed_replies_are_rejected(protocol: str, reply: str, needle: str) -> None:
    module = load_module()
    ok, reason = module.validate_reply(protocol, reply, "chapter text")
    assert not ok and needle in reason


def test_parse_chapters_is_numeric_and_ignores_non_chapters(tmp_path: Path) -> None:
    module = load_module()
    for name in ("chapter-10.en.md", "chapter-2.en.md", "chapter-1.en.md", "index.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert [number for number, _ in module.parse_chapters(tmp_path)] == [1, 2, 10]
    assert [number for number, _ in module.parse_chapters(tmp_path, limit=2)] == [1, 2]


@pytest.mark.skipif(NODE is None or not TL254.is_file(), reason="node v22 and TL254 are needed")
def test_one_messages_line_per_accepted_item(tmp_path: Path, prompts: Path) -> None:
    """A full run with a fake teacher: three canonical messages per accepted chapter."""
    module = load_module()
    chapters_dir = tmp_path / "chapters"
    chapters_dir.mkdir()
    for number in (1, 2):
        (chapters_dir / f"chapter-{number}.en.md").write_text(
            f"chapter-{number} body", encoding="utf-8"
        )
    replies = {
        1: "- first fact\n- second fact\n- third fact\n",
        2: "- fourth fact\n- fifth fact\n- sixth fact\n",
    }

    def fake_chat(endpoint, teacher, system, user, timeout):
        for number, reply in replies.items():
            if f"chapter-{number} body" in user:
                return reply
        raise AssertionError("unexpected prompt")

    data_root = tmp_path / "data"
    result = module.export_protocol(
        "summary",
        module.parse_chapters(chapters_dir),
        prompts,
        "http://example.invalid:1",
        "fake:tag",
        0,
        tmp_path / "out",
        TL254,
        data_root,
        sleep=0,
        min_records=1,
        chat=fake_chat,
    )
    assert result["ok"], result

    path = data_root / "summaries-v1" / "summaries.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"messages"}
        assert [turn["role"] for turn in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][-1]["content"]
        assert row["messages"][-1]["content"] != row["messages"][-2]["content"]
