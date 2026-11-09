"""Tests for `msft fetch`: retry loop, resume flags, incomplete detection, CLI exit codes."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from msft import fetch
from msft.cli import main
from msft.config import MsftError


def cache_layout(tmp_path: Path, model: str = "org/tiny", *, blob_bytes: int = 10) -> Path:
    """Create a fake HF hub cache and return the snapshot directory it points at."""
    root = tmp_path / "hub" / f"models--{model.replace('/', '--')}"
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "blobs" / "abc").write_bytes(b"x" * blob_bytes)
    snapshot = root / "snapshots" / "rev"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    # Real HF caches symlink snapshot entries to blobs; they must not be counted twice.
    (snapshot / "weights.bin").symlink_to(root / "blobs" / "abc")
    return snapshot


def install_stub(
    monkeypatch: pytest.MonkeyPatch, snapshot_download: Any
) -> dict[str, Any]:
    """Install fake huggingface_hub and torch modules; return the recorded call kwargs."""
    recorded: dict[str, Any] = {}

    def wrapped(model: str, **kwargs: Any) -> str:
        recorded["model"] = model
        recorded["kwargs"] = kwargs
        return snapshot_download(model, **kwargs)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = wrapped  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    return recorded


def test_fetch_returns_complete_snapshot_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = cache_layout(tmp_path)
    recorded = install_stub(monkeypatch, lambda model, **kwargs: str(snapshot))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    result = fetch.run("org/tiny", attempts=3, sleep=lambda seconds: None)

    assert result["ok"] is True
    assert result["attempts"] == 1
    assert result["path"] == str(snapshot)
    # the blob once (not the symlink), plus config.json (not its symlink)
    assert result["bytes"] == 12
    assert result["incomplete"] == []
    assert recorded["kwargs"]["resume_download"] is True
    assert recorded["kwargs"]["max_workers"] == 2
    stderr = capsys.readouterr().err
    assert "attempt 1/3" in stderr
    assert "incomplete: none" in fetch.format_report(result)


def test_fetch_passes_revision_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = cache_layout(tmp_path)
    recorded = install_stub(monkeypatch, lambda model, **kwargs: str(snapshot))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    fetch.run("org/tiny", revision="abc123", attempts=1, sleep=lambda seconds: None)

    assert recorded["kwargs"]["revision"] == "abc123"


def test_fetch_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = cache_layout(tmp_path)
    attempts = {"count": 0}

    def flaky(model: str, **kwargs: Any) -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionError("connection reset")
        return str(snapshot)

    install_stub(monkeypatch, flaky)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    sleeps: list[float] = []

    result = fetch.run("org/tiny", attempts=3, sleep=sleeps.append)

    assert result["ok"] is True
    assert result["attempts"] == 2
    assert sleeps == [fetch.BACKOFF_S]
    stderr = capsys.readouterr().err
    assert "failed: ConnectionError: connection reset" in stderr
    assert "retrying in 15s" in stderr


def test_fetch_all_attempts_failing_returns_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_fail(model: str, **kwargs: Any) -> str:
        raise TimeoutError("stalled")

    install_stub(monkeypatch, always_fail)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    partial = tmp_path / "hub" / "models--org--tiny" / "blobs"
    partial.mkdir(parents=True)
    (partial / "shard.bin.incomplete").write_bytes(b"x" * 7)

    result = fetch.run("org/tiny", attempts=2, sleep=lambda seconds: None)

    assert result["ok"] is False
    assert result["attempts"] == 2
    assert "TimeoutError: stalled" in result["error"]
    assert result["bytes"] == 7


def test_fetch_reports_incomplete_files_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = cache_layout(tmp_path)
    (tmp_path / "hub" / "models--org--tiny" / "blobs" / "half.incomplete").write_bytes(b"x")
    install_stub(monkeypatch, lambda model, **kwargs: str(snapshot))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    result = fetch.run("org/tiny", attempts=1, sleep=lambda seconds: None)

    assert result["ok"] is False
    assert result["incomplete"] == ["blobs/half.incomplete"]
    assert "incomplete file(s) remain" in result["error"]


def test_fetch_without_huggingface_hub_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(fetch, "_import", missing)
    with pytest.raises(MsftError, match="huggingface_hub is not installed"):
        fetch.run("org/tiny", attempts=1)


def test_fetch_without_torch_is_an_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda model, **kwargs: str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def import_or_fail(name: str) -> Any:
        if name == "torch":
            raise ImportError(name)
        return hub

    monkeypatch.setattr(fetch, "_import", import_or_fail)
    with pytest.raises(MsftError, match="torch is not installed"):
        fetch.run("org/tiny", attempts=1)


def test_fetch_rejects_zero_attempts() -> None:
    with pytest.raises(MsftError, match="at least 1"):
        fetch.run("org/tiny", attempts=0)


def test_cli_fetch_json_and_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = cache_layout(tmp_path)
    install_stub(monkeypatch, lambda model, **kwargs: str(snapshot))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    code = main(["--json", "fetch", "--model", "org/tiny", "--attempts", "1"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("{") and captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["model"] == "org/tiny"
    assert payload["bytes"] == 12


def test_cli_fetch_exit_one_when_attempts_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def always_fail(model: str, **kwargs: Any) -> str:
        raise RuntimeError("disk full")

    install_stub(monkeypatch, always_fail)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    code = main(["--json", "fetch", "--model", "org/tiny", "--attempts", "1"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False
    assert "disk full" in payload["error"]


def test_cli_fetch_exit_two_without_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(fetch, "_import", missing)
    code = main(["fetch", "--model", "org/tiny"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "uv pip install -e '.[train]'" in captured.err
