"""Tests for `msft probe`: the request it sends and the predictions it writes.

No network: `_post_json` is replaced with a dict lookup keyed on the user turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from msft import probe


def write_cases(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write case records as JSONL and return the path."""
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def install(monkeypatch: pytest.MonkeyPatch, replies: dict[str, str]) -> None:
    """Replace the HTTP POST with a lookup on the user message, recording each request."""

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        user = payload["messages"][-1]["content"]
        return {"message": {"content": replies[user]}}

    monkeypatch.setattr(probe, "_post_json", post)


def test_probe_carries_the_task_into_the_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ner gold reply must pass `msft eval` through the probe too, so `task` has to survive."""
    cases = write_cases(
        tmp_path,
        [
            {"case": "n1", "task": "ner", "prompt": "Extract the terms.", "expected_lines": 1},
            {"case": "t1", "task": "typofix", "prompt": "Fix the typos.", "expected_lines": 2},
        ],
    )
    install(
        monkeypatch,
        {
            "Extract the terms.": '[{"name": "Nadia"}]',
            "Fix the typos.": "He walked in.\nShe nodded.",
        },
    )
    result = probe.run("http://ollama.test:11434", "m:1", cases, system="sys")
    written = [
        json.loads(line)
        for line in Path(result["predictions"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["task"] for row in written] == ["ner", "typofix"]
    assert result["ok"] is True
    assert result["request"]["think"] is False


def test_probe_without_task_keeps_the_translate_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A case with no `task` is still scored as translate: no task key, terms required."""
    cases = write_cases(
        tmp_path, [{"case": "x1", "prompt": "Translate this line.", "expected_lines": 1}]
    )
    install(monkeypatch, {"Translate this line.": "1=Привет."})
    result = probe.run("http://ollama.test:11434", "m:1", cases, system="sys")
    written = [
        json.loads(line)
        for line in Path(result["predictions"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "task" not in written[0]
    assert result["ok"] is False
    terms = next(gate for gate in result["gates"] if gate["gate"] == "terms_block")
    assert terms["passed"] is False
