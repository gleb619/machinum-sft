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


def row(case: str, reply: str, chrf: float | None = None) -> dict[str, Any]:
    """A successful reply row shaped like `run_model` writes it."""
    script = metrics.script_flags(reply, "source")
    row_metrics: dict[str, Any] = {"script": script}
    if chrf is not None:
        row_metrics["chrf"] = chrf
    return {
        "case": case,
        "prompt_logged": "Translate this line.",
        "reply": reply,
        "metrics": row_metrics,
    }


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


def test_report_has_no_reference_columns_when_nothing_is_referenced() -> None:
    cases = [arena.Case("c1", "p", "s", None, None)]
    rows = [row("c1", "ok")]
    board = arena.leaderboard(["m1"], {"m1": rows}, cases, [], self_judge=None)
    report = arena.format_report(board)
    assert "| chrF (0) |" in report
    assert "format + judge only" in " ".join(board["notes"])


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
