"""Tests for `msft arena`. Offline: both network seams are replaced, nothing opens a socket."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from msft import arena, metrics
from msft.cli import main
from msft.config import MsftError, redact
from msft.evaluate import run_gates

TAGS: dict[str, Any] = {
    "models": [
        {"name": "a:1", "digest": "sha256:aaa"},
        {"name": "b:1", "digest": "sha256:bbb"},
        {"name": "j:1", "digest": "sha256:jjj"},
        {"name": "c:latest", "digest": "sha256:ccc"},
        {"name": "candidate-model:7b", "digest": "sha256:model"},
    ]
}

CASE = {
    "case": "c1",
    "prompt": "Translate this line.",
    "source": "The deck stands.",
    "expected_lines": 1,
}


def default_reply(payload: dict[str, Any]) -> dict[str, Any]:
    """A well-formed candidate reply, or a well-formed judge reply when `format` is json."""
    if payload.get("format") == "json":
        return {"message": {"content": '{"fidelity": 4, "fluency": 5}'}}
    return {
        "message": {"content": "Привет.", "thinking": "план"},
        "done_reason": "stop",
        "total_duration": 2_000_000_000,
        "load_duration": 1_000_000_000,
        "eval_duration": 500_000_000,
        "eval_count": 10,
        "prompt_eval_count": 7,
    }


class Seam:
    """Stand-in for both module-level network functions, recording every payload."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.reply = default_reply
        self.errors: dict[str, str] = {}

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.posts.append(payload)
        model = str(payload.get("model"))
        if model in self.errors:
            raise MsftError(self.errors[model])
        return self.reply(payload)

    def get(self, url: str, timeout: float) -> dict[str, Any]:
        self.gets.append(url)
        return copy.deepcopy(TAGS)


def install(monkeypatch: pytest.MonkeyPatch) -> Seam:
    """Replace both seams and hand back the recorder."""
    seam = Seam()
    monkeypatch.setattr(arena, "_post_json", seam.post)
    monkeypatch.setattr(arena, "_get_json", seam.get)
    return seam


def write_cases(tmp_path: Path, *records: dict[str, Any] | str) -> Path:
    """Write one JSONL file; strings are written verbatim so a bad line stays bad."""
    path = tmp_path / "cases.jsonl"
    lines = [record if isinstance(record, str) else json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def only_object(stdout: str) -> dict[str, Any]:
    """Parse stdout as exactly one JSON object with no extra output."""
    assert stdout.startswith("{"), stdout
    assert stdout.rstrip("\n").count("\n") == 0, stdout
    parsed = json.loads(stdout)
    assert isinstance(parsed, dict)
    return parsed


def row(
    case: str,
    reply: str,
    chrf: float | None = None,
    *,
    rate: float | None = None,
    total_ms: float | None = None,
    contract: float | None = None,
) -> dict[str, Any]:
    """A successful reply row shaped like `run_model` writes it."""
    script = metrics.script_flags(reply, "source")
    row_metrics: dict[str, Any] = {"script": script}
    if chrf is not None:
        row_metrics["chrf"] = chrf
    payload: dict[str, Any] = {
        "case": case,
        "prompt_logged": "Translate this line.",
        "reply": reply,
        "metrics": row_metrics,
    }
    timing: dict[str, Any] = {}
    if rate is not None:
        timing["tokens_per_s"] = rate
    if total_ms is not None:
        timing["total_ms"] = total_ms
    if timing:
        payload["timing"] = timing
    if contract is not None:
        payload["contract"] = contract
        payload["contract_violations"] = [] if contract == 1.0 else ["stub violation"]
    return payload

def contract_of(reply: str, expect: dict[str, Any]) -> tuple[float, list[str]]:
    """`metrics.contract` for a case that does carry an `expect`, with the score made concrete."""
    score, violations = metrics.contract(reply, expect)
    assert score is not None, "a case with an expect must produce a contract score"
    return score, violations


EXAMPLE_CASES = Path(__file__).resolve().parents[1] / "docs" / "arena-cases.example.jsonl"


# --- case resolution ------------------------------------------------------


def test_resolve_cases_accepts_prompt_user_and_messages(tmp_path: Path) -> None:
    path = write_cases(
        tmp_path,
        {"case": "p", "prompt": "one", "reference": "ref"},
        {"case": "u", "user": "two"},
        {"case": "m", "messages": [{"role": "user", "content": "three"}]},
    )
    cases = arena.resolve_cases(path, [])
    assert [case.prompt for case in cases] == ["one", "two", "three"]
    assert cases[0].reference == "ref"
    assert cases[0].source == "one"
    assert cases[1].source == "two"
    assert all(case.error is None for case in cases)


def test_resolve_cases_bad_json_and_missing_prompt_are_error_cases(tmp_path: Path) -> None:
    path = write_cases(tmp_path, {"case": "ok", "prompt": "x"}, "not json", {"prompt": "  "})
    cases = arena.resolve_cases(path, [])
    assert cases[0].error is None
    assert cases[1].case == "line 2"
    assert cases[1].error is not None and cases[1].error.startswith("invalid JSON")
    assert cases[2].error == "case has no prompt, user or messages text"


def test_resolve_cases_ad_hoc_prompts_are_numbered() -> None:
    cases = arena.resolve_cases(None, ["first", "second"])
    assert [case.case for case in cases] == ["ad-hoc-1", "ad-hoc-2"]
    assert [case.prompt for case in cases] == ["first", "second"]


def test_resolve_cases_rejects_usage_errors(tmp_path: Path) -> None:
    path = write_cases(tmp_path, {"case": "same", "prompt": "x"}, {"case": "same", "prompt": "y"})
    with pytest.raises(MsftError, match="duplicate case"):
        arena.resolve_cases(path, [])
    with pytest.raises(MsftError, match="not both"):
        arena.resolve_cases(path, ["x"])
    with pytest.raises(MsftError, match="no cases"):
        arena.resolve_cases(None, [])


# --- preflight ------------------------------------------------------------


def test_preflight_records_digests_and_resolves_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seam = install(monkeypatch)
    digests = arena.preflight("http://ollama.test", ["a:1", "c"], timeout=5)
    assert digests == {"a:1": "sha256:aaa", "c": "sha256:ccc"}
    assert seam.gets == ["http://ollama.test/api/tags"]


def test_preflight_unknown_model_names_the_available_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch)
    with pytest.raises(MsftError) as excinfo:
        arena.preflight("http://ollama.test", ["missing:1"], timeout=5)
    message = str(excinfo.value)
    assert "missing:1" in message
    assert "a:1" in message


# --- candidates -----------------------------------------------------------


def test_run_model_payload_and_error_row(monkeypatch: pytest.MonkeyPatch) -> None:
    seam = install(monkeypatch)
    cases = [
        arena.Case(
            case="c1",
            prompt="Translate this line.",
            source="The deck.",
            reference="Палуба.",
            expected_lines=1,
        )
    ]
    good = arena.run_model(
        "http://ollama.test",
        "a:1",
        cases,
        system="sys",
        seed=7,
        num_predict=64,
        timeout=5,
    )[0]
    payload = seam.posts[0]
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0, "seed": 7, "num_predict": 64}
    assert payload["messages"][-1]["content"] == "Translate this line."
    assert good["reply"] == "Привет."
    assert good["metrics"]["bleu"] is not None
    assert good["timing"]["tokens_per_s"] == 20.0
    assert good["timing"]["load_ms"] == 1000.0
    assert good["thinking_chars"] == 4
    assert good["done_reason"] == "stop"

    seam.errors["b:1"] = "connection refused"
    failed = arena.run_model(
        "http://ollama.test",
        "b:1",
        cases,
        system="sys",
        seed=7,
        num_predict=64,
        timeout=5,
    )[0]
    assert failed["_error"].startswith("MsftError: connection refused")
    assert "reply" not in failed


def test_replies_file_passes_evaluate_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)
    cases = write_cases(tmp_path, CASE)
    board = arena.run(
        endpoint="http://ollama.test",
        models=["a:1"],
        cases=cases,
        out=tmp_path / "out",
    )
    assert board["ok"] is True
    replies = Path(board["artifacts"]["replies"]["a:1"])
    gates = run_gates(replies)
    assert gates
    assert all(gate.passed for gate in gates)
    out = tmp_path / "out"
    for name in ("cases.jsonl", "judge.jsonl", "leaderboard.json", "leaderboard.md", "meta.json"):
        assert (out / name).is_file()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["models"] == [{"name": "a:1", "digest": "sha256:aaa"}]
    assert meta["cases"] == 1
    assert meta["n_ref"] == 0


# --- aggregation ----------------------------------------------------------


def test_score_failure_is_zero_and_n_ref_only_counts_references() -> None:
    cases = [
        arena.Case("c1", "p", "s", "ref one", None),
        arena.Case("c2", "p", "s", "ref two", None),
        arena.Case("c3", "p", "s", None, None),
    ]
    rows = [
        row("c1", "ref one"),
        {"case": "c2", "prompt_logged": "Translate this line.", "_error": "boom"},
        row("c3", "ok"),
    ]
    stats = arena.score({"m": rows}, cases)["m"]
    assert stats["answered"] == 2
    assert stats["n_ref"] == 2
    assert stats["chrf"] == 50.0
    assert stats["schema_validity"] == pytest.approx(2 / 3, abs=1e-4)


def test_leaderboard_marks_a_model_within_noise_as_near_leader() -> None:
    cases = [
        arena.Case("c1", "p", "s", "ref one", None),
        arena.Case("c2", "p", "s", "ref two", None),
    ]
    rows = [row("c1", "ref one", 100.0), row("c2", "ref two", 100.0)]
    board = arena.leaderboard(
        ["m1", "m2"], {"m1": rows, "m2": copy.deepcopy(rows)}, cases, [], self_judge=None
    )
    assert board["leader"] == "m1"
    assert board["sort_key"].endswith("chrF")
    by_model = {entry["model"]: entry for entry in board["rows"]}
    assert by_model["m2"]["near_leader"] is True
    assert any("within noise" in note for note in board["notes"])


def test_a_protocol_failure_is_never_marked_near_leader() -> None:
    cases = [
        arena.Case("c1", "p", "s", "ref one", 1),
        arena.Case("c2", "p", "s", "ref two", 1),
    ]
    good = [row("c1", "ref one", 100.0), row("c2", "ref two", 100.0)]
    bad = [row("c1", "ref one\nsecond line", 100.0), row("c2", "ref two\nsecond line", 100.0)]
    for entry in good + bad:
        entry["expected_lines"] = 1
    board = arena.leaderboard(["m1", "m2"], {"m1": good, "m2": bad}, cases, [], self_judge=None)
    by_model = {entry["model"]: entry for entry in board["rows"]}
    assert by_model["m1"]["schema_validity"] == 1.0
    assert by_model["m2"]["schema_validity"] == 0.0
    assert by_model["m2"]["near_leader"] is False, (
        "an identical chrF must not hide a line-parity failure"
    )
    assert "≈ chrF" not in arena.format_report(board)


def test_rank_by_sorts_on_one_axis_and_refuses_an_unknown_one() -> None:
    """`--rank-by` answers 'best at X' without a weighted sum deciding it for the user."""
    expect = {"format": "bullet_list", "bullets_min": 3}
    cases = [arena.Case("c1", "p", "s", "ref one", None, task="summarize", expect=expect)]
    # m1 is fast and ignores the format; m2 is slow and obeys it. Each axis must pick one.
    m1 = [row("c1", "one paragraph of prose", rate=300.0, total_ms=100.0, contract=0.0)]
    m2 = [row("c1", "- a\n- b\n- c", rate=100.0, total_ms=900.0, contract=1.0)]
    replies = {"m1": m1, "m2": m2}

    default = arena.leaderboard(["m1", "m2"], replies, cases, [], self_judge=None)
    assert default["sort_key"].startswith("schema_validity"), "the default order is unchanged"

    def ranked(axis: str) -> list[str]:
        board = arena.leaderboard(
            ["m1", "m2"], replies, cases, [], self_judge=None, rank_by=axis
        )
        assert board["sort_key"].startswith(f"requested: {axis}")
        return [entry["model"] for entry in board["rows"]]

    assert ranked("contract") == ["m2", "m1"], "the model that obeys the format wins that axis"
    assert ranked("speed_index") == ["m1", "m2"], "the faster model wins the speed axis"
    assert ranked("p50_reply_s") == ["m1", "m2"], "latency ranks lowest-first"
    with pytest.raises(MsftError, match="rank-by"):
        arena.leaderboard(["m1"], {"m1": m1}, cases, [], self_judge=None, rank_by="vibes")


def test_report_has_no_reference_columns_when_nothing_is_referenced() -> None:
    cases = [arena.Case("c1", "p", "s", None, None)]
    rows = [row("c1", "ok")]
    board = arena.leaderboard(["m1"], {"m1": rows}, cases, [], self_judge=None)
    report = arena.format_report(board)
    assert "| chrF (0) |" in report
    assert "format + judge only" in " ".join(board["notes"])


# --- the instruction contract ---------------------------------------------


def test_contract_is_null_without_expect() -> None:
    assert metrics.contract("- a\n- b\n- c", None) == (None, [])
    cases = [arena.Case("c1", "p", "s", None, None)]
    stats = arena.score({"m": [row("c1", "ok")]}, cases)["m"]
    assert stats["contract"] is None, "a case that asked for no shape cannot fail one"
    assert stats["n_contract"] == 0


def test_contract_bullet_list_passes_and_flags_a_short_list() -> None:
    expect = {"format": "bullet_list", "bullets_min": 3}
    assert metrics.contract("- one\n- two\n- three", expect) == (1.0, [])
    assert metrics.contract("1. one\n2. two\n3. three", expect) == (1.0, [])
    score, violations = contract_of("- one\n- two", expect)
    assert score == pytest.approx(2 / 3, abs=1e-4), "two of three bullets is not a pass"
    assert violations and violations[0].startswith("bullets_min")


def test_contract_bullet_list_flags_prose_around_the_list() -> None:
    expect = {"format": "bullet_list", "bullets_min": 3}
    assert metrics.contract("Harbor incident\n- one\n- two\n- three", expect)[0] == 1.0, (
        "one title line above the list is allowed"
    )
    score, violations = contract_of("First line.\nSecond line.\n- one\n- two\n- three", expect)
    assert score < 1.0
    assert any(violation.startswith("lead_in") for violation in violations)
    score, violations = contract_of("- one\n- two\nand a paragraph\n- three", expect)
    assert score < 1.0
    assert any(violation.startswith("bullets_only") for violation in violations)


def test_contract_json_object_allows_a_fence_but_not_chatter() -> None:
    expect = {"format": "json_object", "keys": ["persons", "places"]}
    fenced = '```json\n{"persons": ["Mara"], "places": ["Klaipeda"]}\n```'
    assert metrics.contract(fenced, expect) == (1.0, []), "a fence is the shape asked for"
    score, violations = contract_of(
        'Here you go:\n{"persons": [], "places": []}\nHope it helps', expect
    )
    assert score < 1.0
    assert any(violation.startswith("no_leading_chatter") for violation in violations)
    assert any(violation.startswith("no_trailing_chatter") for violation in violations)
    score, violations = contract_of('{"persons": []}\n{"places": []}', expect)
    assert score < 1.0, "two objects is not ONE object"
    assert any(violation.startswith("no_trailing_chatter") for violation in violations)


def test_contract_json_object_requires_every_key() -> None:
    expect = {"format": "json_object", "keys": ["persons", "places", "dates"]}
    score, violations = contract_of('{"persons": [], "places": []}', expect)
    assert score == pytest.approx(3 / 4, abs=1e-4)
    assert any("dates" in violation for violation in violations)
    score, violations = contract_of("not json at all", expect)
    assert score == 0.0
    assert any(violation.startswith("json_object") for violation in violations)


def test_contract_term_table_flags_duplicates_and_shape() -> None:
    expect = {"format": "term_table", "pairs_min": 3}
    good = "deck -> палуба\nharbor: гавань\nlantern — фонарь"
    assert metrics.contract(good, expect) == (1.0, [])
    bulleted = "- deck -> палуба\n- harbor: гавань\n- lantern — фонарь"
    assert metrics.contract(bulleted, expect)[0] == 1.0
    score, violations = contract_of("deck -> палуба\ndeck: paluba\nlantern -> фонарь", expect)
    assert score < 1.0
    assert any(violation.startswith("no_duplicate_terms") for violation in violations)
    score, violations = contract_of(
        "deck -> палуба\nI found three terms.\nharbor -> гавань", expect
    )
    assert score < 1.0
    assert any(violation.startswith("pairs_shaped") for violation in violations)
    score, violations = contract_of("deck -> палуба\nharbor -> гавань", expect)
    assert score < 1.0, "pairs_min is a floor, not a suggestion"
    assert any(violation.startswith("pairs_min") for violation in violations)


def test_contract_edit_pairs_flags_an_unchanged_line_that_moved() -> None:
    frozen = "The night shift had already handed over the manifest."
    expect = {"format": "edit_pairs", "pairs_min": 2, "unchanged": True, "unchanged_line": frozen}
    good = f"their -> They're\ngoing too -> going to\n{frozen}"
    assert metrics.contract(good, expect)[0] == 1.0, "an untouched line may sit between the edits"
    score, violations = contract_of("their -> They're\ngoing too -> going to", expect)
    assert score < 1.0
    assert any(violation.startswith("unchanged_line") for violation in violations)

    missing = {"format": "edit_pairs", "unchanged": True}
    score, violations = contract_of("a -> b", missing)
    assert score < 1.0
    assert any("unchanged_line is missing" in violation for violation in violations)

    score, violations = contract_of("teh -> the\nteh -> a", {"format": "edit_pairs"})
    assert score < 1.0
    assert any(violation.startswith("no_conflicting_edits") for violation in violations)


def test_contract_plain_text_rejects_markdown() -> None:
    expect = {"format": "plain_text"}
    assert metrics.contract("Палуба стоит.", expect) == (1.0, [])
    for reply, rule in (
        ("**Палуба** стоит.", "no_bold"),
        ("# Палуба\nстоит.", "no_heading"),
        ("- Палуба стоит.", "no_bullet"),
        ("```\nПалуба стоит.\n```", "no_fence"),
    ):
        score, violations = contract_of(reply, expect)
        assert score == pytest.approx(3 / 4, abs=1e-4), reply
        assert any(violation.startswith(rule) for violation in violations), reply


def test_contract_empty_and_unknown_format_score_zero() -> None:
    assert metrics.contract("", {"format": "plain_text"}) == (0.0, ["non_empty: reply is empty"])
    score, violations = contract_of("anything", {"format": "nonsense"})
    assert score == 0.0
    assert violations == ["unknown format: 'nonsense'"]


def test_run_model_puts_task_and_contract_on_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    cases = [
        arena.Case(
            "s1",
            "List three things.",
            "source",
            None,
            None,
            task="summarize",
            expect={"format": "bullet_list", "bullets_min": 3},
        ),
        arena.Case("t1", "Translate.", "source", None, None, task="translate"),
    ]
    replies = arena.run_model(
        "http://ollama.test", "a:1", cases, system="sys", seed=7, num_predict=64, timeout=5
    )
    assert replies[0]["task"] == "summarize"
    assert replies[0]["contract"] == 0.0, "Привет. is not a bullet list"
    assert replies[0]["contract_violations"], "the reason must be on the row"
    assert "contract" not in replies[1], "a case with no `expect` carries no contract"
    assert replies[1]["task"] == "translate"


def test_score_contract_mean_skips_unasked_cases_and_zeroes_unanswered_ones() -> None:
    cases = [
        arena.Case("s1", "p", "s", None, None, task="summarize", expect={"format": "bullet_list"}),
        arena.Case("s2", "p", "s", None, None, task="summarize", expect={"format": "bullet_list"}),
        arena.Case("t1", "p", "s", None, None, task="translate"),
    ]
    rows = [
        row("s1", "- a\n- b\n- c", contract=1.0),
        {"case": "s2", "prompt_logged": "p", "_error": "boom"},
        row("t1", "ok"),
    ]
    stats = arena.score({"m": rows}, cases)["m"]
    assert stats["n_contract"] == 2
    assert stats["contract"] == 0.5, "1.0 answered + 0.0 failed, and the unasked case is absent"
    assert stats["tasks"]["translate"]["contract"] is None


# --- speed and leaders ----------------------------------------------------


def test_speed_index_is_one_for_the_fastest_and_scales_linearly() -> None:
    cases = [arena.Case("c1", "p", "s", None, None), arena.Case("c2", "p", "s", None, None)]
    replies = {
        "fast": [
            row("c1", "ok", rate=90.0, total_ms=400.0),
            row("c2", "ok", rate=90.0, total_ms=1000.0),
        ],
        "half": [row("c1", "ok", rate=45.0, total_ms=800.0)],
        "slow": [row("c1", "ok", rate=30.0, total_ms=1200.0)],
    }
    stats = arena.score(replies, cases)
    assert stats["fast"]["speed_index"] == 1.0
    assert stats["half"]["speed_index"] == 0.5
    assert stats["slow"]["speed_index"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["fast"]["p50_reply_s"] == 0.7, "median of 0.4 s and 1.0 s"
    assert stats["fast"]["worst_reply_s"] == 1.0
    assert stats["slow"]["worst_reply_s"] == 1.2
    assert stats["fast"]["tokens_per_s"] == 90.0, "the rate stays a rate beside the share"


def test_speed_index_is_none_when_no_timing_arrived() -> None:
    cases = [arena.Case("c1", "p", "s", None, None)]
    stats = arena.score({"m": [row("c1", "ok")]}, cases)["m"]
    assert stats["speed_index"] is None
    assert stats["p50_reply_s"] is None
    assert stats["worst_reply_s"] is None


def test_leaders_name_one_model_per_axis() -> None:
    cases = [
        arena.Case(
            "s1",
            "p",
            "s",
            "ref",
            None,
            task="summarize",
            expect={"format": "bullet_list", "bullets_min": 3},
        )
    ]
    fast = [row("s1", "- a\n- b\n- c", 40.0, rate=100.0, total_ms=500.0, contract=1.0)]
    accurate = [row("s1", "- a\n- b\n- c\n- d", 90.0, rate=25.0, total_ms=2000.0, contract=0.75)]
    board = arena.leaderboard(
        ["fast:1", "accurate:1"],
        {"fast:1": fast, "accurate:1": accurate},
        cases,
        [],
        self_judge=None,
    )
    leaders = board["leaders"]
    assert leaders["contract"] == "fast:1"
    assert leaders["chrf"] == "accurate:1"
    assert leaders["speed_index"] == "fast:1"
    assert leaders["schema_validity"] == "accurate:1", "a tie breaks on the model name"
    assert set(leaders) == {"schema_validity", "contract", "chrf", "speed_index"}, (
        "axes with no value are absent, not filled with a default"
    )
    report = arena.format_report(board)
    assert "who leads on each axis" in report
    assert "| contract | fast:1 | 1.000 |" in report
    assert "not comparable to each other" in report
    assert "composite" not in json.dumps(board).lower(), "no field may sum the axes"


def test_leaders_skip_a_self_judged_column() -> None:
    cases = [arena.Case("c1", "p", "s", "ref one", None)]
    rows = [row("c1", "ref one", 90.0)]
    judge = [
        {"case": "c1", "model": "a:1", "fidelity": 5, "fluency": 5},
        {"case": "c1", "model": "b:1", "fidelity": 3, "fluency": 3},
    ]
    board = arena.leaderboard(
        ["a:1", "b:1"], {"a:1": rows, "b:1": copy.deepcopy(rows)}, cases, judge, self_judge="a:1"
    )
    assert board["leaders"]["judge_fidelity"] == "b:1", "a model grading itself is not evidence"


def test_per_task_breakdown_separates_summarize_from_translate() -> None:
    cases = [
        arena.Case("s1", "p", "s", None, None, task="summarize", expect={"format": "bullet_list"}),
        arena.Case("t1", "p", "s", "Палуба стоит.", None, task="translate"),
    ]
    board = arena.leaderboard(
        ["m"],
        {"m": [row("s1", "- a\n- b\n- c", contract=1.0), row("t1", "Палуба.", 55.0)]},
        cases,
        [],
        self_judge=None,
    )
    tasks = board["rows"][0]["tasks"]
    assert tasks["summarize"]["contract"] == 1.0
    assert tasks["summarize"]["chrf"] is None
    assert tasks["translate"]["contract"] is None
    assert tasks["translate"]["chrf"] == metrics.chrf("Палуба.", "Палуба стоит.")
    report = arena.format_report(board)
    assert "per task" in report
    assert "| summarize (1) |" in report
    assert "| translate (1) |" in report


def test_run_scores_the_contract_and_writes_the_leaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seam = install(monkeypatch)
    seam.reply = lambda payload: {
        "message": {"content": "- a\n- b\n- c"},
        "done_reason": "stop",
        "total_duration": 2_000_000_000,
        "load_duration": 1_000_000_000,
        "eval_duration": 500_000_000,
        "eval_count": 10,
        "prompt_eval_count": 7,
    }
    cases = write_cases(
        tmp_path,
        {
            "case": "s1",
            "prompt": "List three things.",
            "task": "summarize",
            "expect": {"format": "bullet_list", "bullets_min": 3},
        },
        {"case": "t1", "prompt": "Translate.", "reference": "Палуба."},
    )
    board = arena.run(
        endpoint="http://ollama.test", models=["a:1"], cases=cases, out=tmp_path / "out"
    )
    entry = board["rows"][0]
    assert board["n_contract"] == 1
    assert entry["contract"] == 1.0
    assert entry["tasks"]["summarize"]["contract"] == 1.0
    assert entry["speed_index"] == 1.0
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_contract"] == 1
    report = (tmp_path / "out" / "leaderboard.md").read_text(encoding="utf-8")
    assert "who leads on each axis" in report
    assert "| contract | a:1 | 1.000 |" in report
    records = [
        json.loads(line)
        for line in Path(board["artifacts"]["replies"]["a:1"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["task"] == "summarize"
    assert records[0]["contract"] == 1.0
    assert "contract" not in records[1]


def test_example_cases_load_with_every_field_the_loader_needs() -> None:
    text = EXAMPLE_CASES.read_text(encoding="utf-8")
    raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    cases = arena.resolve_cases(EXAMPLE_CASES, [])
    assert len(cases) == len(raw)
    assert all(case.error is None for case in cases), "every example case must load"
    assert all(case.prompt for case in cases)
    assert {"translate", "summarize", "ner", "proofread"} <= {case.task for case in cases}
    by_name = {case.case: case for case in cases}
    assert by_name["summary-bullets"].expect == {"format": "bullet_list", "bullets_min": 3}
    assert by_name["ner-term-table"].expect == {"format": "term_table", "pairs_min": 4}
    assert by_name["ner-json-object"].expect["keys"] == ["persons", "places", "dates"]
    edits = by_name["proofread-edit-pairs"].expect
    assert edits["format"] == "edit_pairs" and edits["unchanged"] is True
    assert edits["unchanged_line"] in by_name["proofread-edit-pairs"].source
    formats = {case.expect["format"] for case in cases if case.expect}
    assert formats == set(metrics.CONTRACT_FORMATS), "the example file exercises every format"
    # The resolved case set must round-trip: cases.jsonl keeps `task` and `expect`.
    for index, case in enumerate(cases, 1):
        again = arena._case_from_record(arena._case_record(case), index)
        assert (again.task, again.expect) == (case.task, case.expect)
    # And the contract scorer agrees with the file on a good reply.
    assert metrics.contract("- a\n- b\n- c", by_name["summary-bullets"].expect) == (1.0, [])
    assert (
        metrics.contract(
            "Mara -> Мара\nKlaipeda -> Клайпеда\nCormac -> Кормак\npier -> пирс",
            by_name["ner-term-table"].expect,
        )[0]
        == 1.0
    )


# --- judge ----------------------------------------------------------------


def test_judge_runs_after_every_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seam = install(monkeypatch)
    cases = write_cases(tmp_path, CASE)
    board = arena.run(
        endpoint="http://ollama.test",
        models=["a:1", "b:1"],
        cases=cases,
        judge="j:1",
        out=tmp_path / "out",
    )
    assert [payload["model"] for payload in seam.posts] == ["a:1", "b:1", "j:1", "j:1"]
    assert seam.posts[-1]["format"] == "json"
    assert board["ok"] is True
    assert board["judge_ran"] is True


def test_self_judge_is_refused_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)
    cases = write_cases(tmp_path, CASE)
    with pytest.raises(MsftError, match="also a candidate"):
        arena.run(
            endpoint="http://ollama.test",
            models=["a:1", "b:1"],
            cases=cases,
            judge="a:1",
            out=tmp_path / "out",
        )


def test_self_judge_is_starred_and_excluded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install(monkeypatch)
    cases = write_cases(tmp_path, CASE)
    board = arena.run(
        endpoint="http://ollama.test",
        models=["a:1", "b:1"],
        cases=cases,
        judge="a:1",
        allow_self_judge=True,
        out=tmp_path / "out",
    )
    by_model = {entry["model"]: entry for entry in board["rows"]}
    assert by_model["a:1"]["judge_star"] is True
    assert by_model["a:1"]["judge_mean"] is None
    assert by_model["b:1"]["judge_mean"] is not None
    assert "self-judged*" in arena.format_report(board)


def test_judge_prompt_never_contains_a_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    seam = install(monkeypatch)
    case = arena.Case("c1", "Translate this line.", "The deck.", "Палуба.", 1)
    replies = {"candidate-model:7b": [{"case": "c1", "reply": "Привет."}]}
    arena.judge_rows("http://ollama.test", "j:1", replies, [case], timeout=5)
    payload = seam.posts[-1]
    text = payload["messages"][1]["content"]
    assert "candidate-model:7b" not in text
    assert "SOURCE:" in text and "CANDIDATE:" in text
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0


def test_unparsable_judge_reply_is_a_row_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seam = install(monkeypatch)
    seam.reply = lambda payload: {"message": {"content": "I think it is fine."}}
    cases = write_cases(tmp_path, CASE)
    board = arena.run(
        endpoint="http://ollama.test",
        models=["a:1"],
        cases=cases,
        judge="j:1",
        out=tmp_path / "out",
    )
    judge_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "judge.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert judge_rows[0]["judge_error"]
    assert board["ok"] is False


# --- CLI ------------------------------------------------------------------


def test_cli_json_prints_exactly_one_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch)
    code = main(
        [
            "arena",
            "--models",
            "a:1,b:1",
            "--prompt",
            "Translate this line.",
            "--endpoint",
            "http://ollama.test",
            "--out",
            str(tmp_path / "out"),
            "--json",
        ]
    )
    payload = only_object(capsys.readouterr().out)
    assert code == 0
    assert {entry["model"] for entry in payload["rows"]} == {"a:1", "b:1"}


def test_cli_exit_0_when_every_case_is_answered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch)
    code = main(
        [
            "arena",
            "--models",
            "a:1",
            "--prompt",
            "Translate this line.",
            "--endpoint",
            "http://ollama.test",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    assert "arena leaderboard" in capsys.readouterr().out


def test_cli_exit_1_when_a_request_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seam = install(monkeypatch)
    seam.errors["b:1"] = "connection refused"
    code = main(
        [
            "arena",
            "--models",
            "a:1,b:1",
            "--prompt",
            "Translate this line.",
            "--endpoint",
            "http://ollama.test",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert (tmp_path / "out" / "leaderboard.md").is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ["--models", "a:1", "--prompt", "x"],
        ["--models", "missing:1", "--prompt", "x", "--endpoint", "http://ollama.test"],
        ["--models", "a:1", "--endpoint", "http://ollama.test"],
        ["--models", "a:1,a_1", "--prompt", "x", "--endpoint", "http://ollama.test"],
        [
            "--models",
            "a:1",
            "--prompt",
            "x",
            "--judge",
            "a:1",
            "--endpoint",
            "http://ollama.test",
        ],
    ],
)
def test_cli_exit_2_on_usage_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    install(monkeypatch)
    monkeypatch.delenv(arena.ENDPOINT_ENV, raising=False)
    assert main(["arena", *argv, "--out", str(tmp_path / "out")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_exit_2_on_duplicate_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch)
    cases = write_cases(tmp_path, {"case": "same", "prompt": "x"}, {"case": "same", "prompt": "y"})
    code = main(
        [
            "arena",
            "--models",
            "a:1",
            "--cases",
            str(cases),
            "--endpoint",
            "http://ollama.test",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert "duplicate case" in capsys.readouterr().err


def test_dry_run_calls_neither_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("--dry-run must not touch the endpoint")

    monkeypatch.setattr(arena, "_post_json", boom)
    monkeypatch.setattr(arena, "_get_json", boom)
    cases = write_cases(tmp_path, CASE, {"case": "bad"})
    code = main(
        [
            "arena",
            "--models",
            "a:1,b:1",
            "--cases",
            str(cases),
            "--judge",
            "j:1",
            "--endpoint",
            "http://ollama.test",
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    text = capsys.readouterr().out
    assert code == 0
    assert "candidate requests: 2" in text
    assert "judge calls: 2" in text
    assert redact(str(tmp_path / "out")) in text
