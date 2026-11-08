"""Tests for the protocol gates: tiny offline fixtures, no model, no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from msft.config import MsftError
from msft.evaluate import GateResult, run_gates

XML_PROMPT = (
    "## Input format (xml mode)\n"
    "Each line is wrapped in a <result> block with <id> and <text>."
)
PROPS_PROMPT = "## Input format (properties mode)\nEach line is provided as a `N=text` pair."

GOOD_XML = (
    "<result><id>1</id><translation>Привет, мир.</translation></result>\n"
    "<result><id>2</id><translation>Как дела?</translation></result>\n"
    "<terms>[]</terms>"
)


def write_predictions(
    tmp_path: Path, records: list[dict[str, Any]], name: str = "predictions.jsonl"
) -> Path:
    """Write records as JSONL and return the file path."""
    path = tmp_path / name
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def by_gate(results: list[GateResult]) -> dict[str, GateResult]:
    """Index gate results by gate name."""
    return {result.gate: result for result in results}


def record(reply: str, prompt: str = XML_PROMPT, **extra: Any) -> dict[str, Any]:
    """Build one prediction record."""
    return {"case": "tiny", "prompt_logged": prompt, "reply": reply, **extra}


def test_good_xml_passes_every_gate(tmp_path: Path) -> None:
    path = write_predictions(tmp_path, [record(GOOD_XML, expected_lines=2)])
    gates = by_gate(run_gates(path))
    failing = sorted(name for name, gate in gates.items() if not gate.passed)
    assert failing == []
    assert gates["schema_validity"].value == 1.0
    assert "2 result block(s)" in gates["xml_shape"].detail

def test_broken_ids_fail_xml_shape(tmp_path: Path) -> None:
    reply = (
        "<result><id>1</id><translation>а</translation></result>\n"
        "<result><id>3</id><translation>б</translation></result>"
    )
    gates = by_gate(run_gates(write_predictions(tmp_path, [record(reply)])))
    assert gates["xml_shape"].passed is False
    assert "contiguous" in gates["xml_shape"].detail
    assert gates["no_chatter"].passed is True
    assert gates["schema_validity"].passed is False
    assert gates["schema_validity"].value == 0.0


def test_duplicate_ids_fail_xml_shape(tmp_path: Path) -> None:
    reply = (
        "<result><id>1</id><translation>а</translation></result>\n"
        "<result><id>1</id><translation>б</translation></result>"
    )
    gates = by_gate(run_gates(write_predictions(tmp_path, [record(reply)])))
    assert gates["xml_shape"].passed is False
    assert "duplicate" in gates["xml_shape"].detail


def test_chatter_fails_no_chatter(tmp_path: Path) -> None:
    reply = "Here is the translation:\n" + GOOD_XML
    gates = by_gate(run_gates(write_predictions(tmp_path, [record(reply)])))
    assert gates["no_chatter"].passed is False
    assert "opener" in gates["no_chatter"].detail or "preamble" in gates["no_chatter"].detail
    assert gates["xml_shape"].passed is True


def test_code_fence_fails_no_chatter(tmp_path: Path) -> None:
    reply = "```xml\n" + GOOD_XML + "\n```"
    gates = by_gate(run_gates(write_predictions(tmp_path, [record(reply)])))
    assert gates["no_chatter"].passed is False
    assert "fence" in gates["no_chatter"].detail


def test_bad_terms_fail_terms_block(tmp_path: Path) -> None:
    reply = (
        "<result><id>1</id><translation>Привет.</translation></result>\n"
        "<terms>[{oops]</terms>"
    )
    gates = by_gate(run_gates(write_predictions(tmp_path, [record(reply)])))
    assert gates["terms_block"].passed is False
    assert "JSON" in gates["terms_block"].detail
    assert gates["xml_shape"].passed is True


def test_double_terms_block_fails_terms_block(tmp_path: Path) -> None:
    reply = GOOD_XML + "\n<terms>[{\"term\":\"Grog\"}]</terms>"
    gates = by_gate(run_gates(write_predictions(tmp_path, [record(reply)])))
    assert gates["terms_block"].passed is False
    assert "at most one" in gates["terms_block"].detail


def test_wrong_line_count_fails_line_parity(tmp_path: Path) -> None:
    reply = "1=Первая строка.\n2=Вторая строка."
    gates = by_gate(
        run_gates(write_predictions(tmp_path, [record(reply, PROPS_PROMPT, expected_lines=3)]))
    )
    assert gates["line_parity"].passed is False
    assert "expected 3" in gates["line_parity"].detail
    assert gates["properties_shape"].passed is True
    assert gates["xml_shape"].passed is True  # not an xml case: skipped, not failed


def test_properties_ids_must_be_contiguous(tmp_path: Path) -> None:
    reply = "1=Первая.\n3=Третья."
    gates = by_gate(
        run_gates(write_predictions(tmp_path, [record(reply, PROPS_PROMPT, expected_lines=2)]))
    )
    assert gates["properties_shape"].passed is False
    assert "contiguous" in gates["properties_shape"].detail
    assert gates["line_parity"].passed is True


def test_empty_reply_fails_no_chatter_and_xml_shape(tmp_path: Path) -> None:
    gates = by_gate(run_gates(write_predictions(tmp_path, [record("")])))
    assert gates["no_chatter"].passed is False
    assert "empty" in gates["no_chatter"].detail
    assert gates["xml_shape"].passed is False
    assert gates["schema_validity"].passed is False


def test_malformed_jsonl_line_is_a_failed_gate_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"case": "ok", "reply": "1=а"}\nnot json at all\n', encoding="utf-8")
    gates = by_gate(run_gates(path))
    assert gates["schema_validity"].passed is False
    assert gates["schema_validity"].value == 0.5
    assert "invalid JSON" in gates["no_chatter"].detail


def test_record_without_reply_fails_cleanly(tmp_path: Path) -> None:
    path = write_predictions(tmp_path, [{"case": "no-reply"}])
    gates = by_gate(run_gates(path))
    assert gates["schema_validity"].passed is False
    assert "reply" in gates["no_chatter"].detail


def test_missing_file_raises_msft_error(tmp_path: Path) -> None:
    with pytest.raises(MsftError):
        run_gates(tmp_path / "absent.jsonl")


def test_as_json_prints_one_object(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_predictions(tmp_path, [record(GOOD_XML)])
    results = run_gates(path, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(results, list)
    assert payload["ok"] is True
    assert payload["cases"] == 1
    assert [gate["gate"] for gate in payload["gates"]][-1] == "schema_validity"
    assert payload["schema_validity"] == 1.0

def test_thinking_chatter_is_caught() -> None:
    """A small model that narrates its reasoning is chatter, even with no preamble."""
    from msft.evaluate import _CHATTER_RE

    for opening in (
        "Thinking Process:\n1. Analyze the Request: the user wants a translation",
        "Let me translate this properly.",
        "The user wants the text normalised, so I will",
    ):
        assert _CHATTER_RE.match(opening), opening
