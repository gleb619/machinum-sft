"""Tests for `msft export`: the docker/ollama sequence, the template rule, and push errors."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from msft import export
from msft.cli import main
from msft.config import MsftError

LISTING = (
    "NAME                     ID    SIZE    MODIFIED\n"
    "machinum-book-ru:4b  abc  2.6 GB  now\n"
    "me/machinum-book-ru:4b  abc  2.6 GB  now\n"
    # ollama tags a bare name, so `ollama create tiny` shows up as `tiny:latest`.
    "tiny:latest  def  2.6 GB  now\n"
)


class FakeRun:
    """Answers docker/ollama commands like a healthy container and records the argv."""

    def __init__(
        self,
        *,
        listing: str = LISTING,
        fail_on: str | None = None,
        stderr: str = "",
        running: str = "true",
    ) -> None:
        self.listing = listing
        self.fail_on = fail_on
        self.stderr = stderr
        self.running = running
        self.calls: list[list[str]] = []
        self.modelfile: str | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if len(argv) > 2 and argv[1] == "cp":
            source = Path(argv[2])
            if source.name == "Modelfile":
                self.modelfile = source.read_text(encoding="utf-8")
        elif "-f" in argv:
            # Host mode: ollama opens the Modelfile itself, there is no `docker cp` to watch.
            candidate = Path(argv[argv.index("-f") + 1])
            if candidate.is_file():
                self.modelfile = candidate.read_text(encoding="utf-8")
        if self.fail_on is not None and self.fail_on in argv:
            return subprocess.CompletedProcess(argv, 1, "", self.stderr)
        if argv[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, f"{self.running}\n", "")
        if argv[-1] == "list":
            return subprocess.CompletedProcess(argv, 0, self.listing, "")
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture(autouse=True)
def small_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not write 500 MB in a unit test; the production constant is unchanged."""
    monkeypatch.setattr(export, "MIN_GGUF_BYTES", 10)
    monkeypatch.setattr(export.shutil, "which", lambda name: f"/usr/bin/{name}")


def gguf_file(tmp_path: Path, size: int = 20, name: str = "model-4b.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"g" * size)
    return path


def test_container_flow_creates_the_model_without_a_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = gguf_file(tmp_path)
    fake = FakeRun()
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "machinum-book-ru:4b")

    assert result["ok"] is True
    assert result["listed"] is True
    assert result["mode"] == "container"
    assert fake.calls[0] == ["docker", "inspect", "-f", "{{.State.Running}}", "ollama"]
    assert fake.calls[1] == ["docker", "exec", "ollama", "mkdir", "-p", "/models"]
    assert fake.calls[2] == ["docker", "cp", str(path), "ollama:/models/model-4b.gguf"]
    assert fake.calls[3][0:2] == ["docker", "cp"]
    assert fake.calls[3][2].endswith("Modelfile")
    assert fake.calls[3][3].startswith("ollama:/tmp/Modelfile.machinum-book-ru_4b")
    assert fake.calls[4] == [
        "docker",
        "exec",
        "ollama",
        "ollama",
        "create",
        "machinum-book-ru:4b",
        "-f",
        "/tmp/Modelfile.machinum-book-ru_4b",
    ]
    assert fake.calls[5] == [
        "docker",
        "exec",
        "ollama",
        "rm",
        "-f",
        "/models/model-4b.gguf",
        "/tmp/Modelfile.machinum-book-ru_4b",
    ]
    assert fake.calls[-1] == ["docker", "exec", "ollama", "ollama", "list"]

    assert fake.modelfile is not None
    assert "FROM /models/model-4b.gguf" in fake.modelfile
    assert "PARAMETER num_ctx 8192" in fake.modelfile
    assert "PARAMETER temperature 0" in fake.modelfile
    assert "PARAMETER top_p 0.95" in fake.modelfile
    assert "PARAMETER repeat_penalty 1.05" in fake.modelfile
    assert "TEMPLATE" not in fake.modelfile


def test_namespace_copies_then_pushes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = gguf_file(tmp_path)
    fake = FakeRun()
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "machinum-book-ru:4b", namespace="me", push=True)

    assert result["ok"] is True
    assert result["target"] == "me/machinum-book-ru:4b"
    target = "me/machinum-book-ru:4b"
    assert ["docker", "exec", "ollama", "ollama", "cp", "machinum-book-ru:4b", target] in (
        fake.calls
    )
    assert any(call[-2:] == ["push", target] for call in fake.calls)


def test_push_failure_is_reported_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = gguf_file(tmp_path)
    message = "Error: you are not authorized to push to this namespace"
    fake = FakeRun(fail_on="push", stderr=message)
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "machinum-book-ru:4b", namespace="me", push=True)

    assert result["ok"] is False
    assert message in result["error"]
    assert message in export.format_report(result)


def test_refuses_a_fragment_gguf(tmp_path: Path) -> None:
    path = gguf_file(tmp_path, size=5)
    with pytest.raises(MsftError, match="not a finished quant"):
        export.run(path, "tiny")


def test_refuses_push_without_a_namespace(tmp_path: Path) -> None:
    path = gguf_file(tmp_path)
    with pytest.raises(MsftError, match="--namespace"):
        export.run(path, "tiny", push=True)


def test_missing_gguf_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(MsftError, match="no such GGUF"):
        export.run(tmp_path / "absent.gguf", "tiny")


def test_missing_docker_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export.shutil, "which", lambda name: None)
    with pytest.raises(MsftError, match="docker is not on PATH"):
        export.run(gguf_file(tmp_path), "tiny")


def test_host_mode_uses_the_local_ollama_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = gguf_file(tmp_path)
    fake = FakeRun()
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "tiny", container="")

    assert result["ok"] is True
    assert result["mode"] == "host"
    assert fake.calls[0][:3] == ["ollama", "create", "tiny"]
    assert fake.modelfile is not None
    assert f"FROM {path.resolve()}" in fake.modelfile


def test_model_missing_from_ollama_list_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = gguf_file(tmp_path)
    fake = FakeRun(listing="NAME ID SIZE MODIFIED\nsomething-else 1 1 GB now\n")
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "tiny")

    assert result["ok"] is False
    assert "did not appear in `ollama list`" in result["error"]


def test_cli_export_json_prints_one_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = gguf_file(tmp_path)
    monkeypatch.setattr(export.subprocess, "run", FakeRun())

    code = main(["--json", "export", "--gguf", str(path), "--name", "tiny"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["name"] == "tiny"


def test_cli_export_exits_one_on_push_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = gguf_file(tmp_path)
    message = "Error: not authorized to push to this namespace"
    monkeypatch.setattr(export.subprocess, "run", FakeRun(fail_on="push", stderr=message))

    code = main(
        [
            "export",
            "--gguf",
            str(path),
            "--name",
            "tiny",
            "--namespace",
            "me",
            "--push",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert message in captured.out


def test_cli_export_exits_two_for_a_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = gguf_file(tmp_path, size=5)
    monkeypatch.setattr(export.subprocess, "run", FakeRun())

    code = main(["export", "--gguf", str(path), "--name", "tiny"])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "not a finished quant" in captured.err


def test_modelfile_never_contains_a_hand_written_template() -> None:
    text = export.build_modelfile("/models/model.gguf")
    assert "TEMPLATE" not in text
    assert text.startswith("FROM /models/model.gguf\n")
    assert "num_ctx 8192" in text


# --- hygiene and the verify gate -----------------------------------------


def test_verify_gate_refuses_a_failed_artifact_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An artifact that fails `msft verify` must never reach `ollama create`."""
    from msft.config import ModelCfg, RunConfig

    path = gguf_file(tmp_path)
    fake = FakeRun()
    monkeypatch.setattr(export.subprocess, "run", fake)

    import msft.verify as verify_mod

    monkeypatch.setattr(
        verify_mod,
        "run",
        lambda cfg: {"ok": False, "failures": ["gguf"], "checks": []},
    )
    result = export.run(path, "tiny", config=RunConfig(model=ModelCfg(id="org/tiny")))

    assert result["ok"] is False
    assert "verify failed (gguf)" in result["error"]
    assert fake.calls == []
    assert result["verify"] == {"ok": False, "failures": ["gguf"], "checks": []}


def test_a_name_with_shell_characters_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MsftError, match="invalid model name"):
        export.run(gguf_file(tmp_path), "x; rm -rf /")


def test_a_namespace_with_a_slash_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MsftError, match="invalid namespace"):
        export.run(gguf_file(tmp_path), "tiny", namespace="me/other")


def test_ansi_and_long_stderr_are_cleaned_and_tailed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = gguf_file(tmp_path)
    noise = "\n".join(f"line {index}" for index in range(120))
    fake = FakeRun(fail_on="create", stderr=f"\x1b[31mError:\x1b[0m {noise}")
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "tiny")

    assert result["ok"] is False
    assert "\x1b" not in result["error"]
    assert "line 119" in result["error"]
    assert "line 10\n" not in result["error"]
    assert "earlier line(s) omitted" in result["error"]


def test_copied_files_are_removed_even_when_create_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale GGUF left in /models looks current on the next export and is not."""
    path = gguf_file(tmp_path)
    fake = FakeRun(fail_on="create", stderr="create failed")
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "tiny")

    assert result["ok"] is False
    removals = [
        call
        for call in fake.calls
        if call[:3] == ["docker", "exec", "ollama"] and "rm" in call
    ]
    assert removals
    assert f"/models/{path.name}" in removals[-1]
    assert any(arg.startswith("/tmp/Modelfile.") for arg in removals[-1])


def test_a_stopped_container_is_refused_before_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = gguf_file(tmp_path)
    fake = FakeRun(running="false")
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "tiny")

    assert result["ok"] is False
    assert "is not running" in result["error"]
    assert not any(call[1:2] == ["cp"] for call in fake.calls)


def test_a_namespaced_target_must_appear_in_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cp` can succeed while `list` still shows only the bare name; that is not exported."""
    path = gguf_file(tmp_path)
    listing = "NAME ID SIZE MODIFIED\nmachinum-book-ru:4b  abc  2.6 GB  now\n"
    fake = FakeRun(listing=listing)
    monkeypatch.setattr(export.subprocess, "run", fake)

    result = export.run(path, "machinum-book-ru:4b", namespace="me")

    assert result["ok"] is False
    assert "me/machinum-book-ru:4b did not appear" in result["error"]
