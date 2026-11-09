"""Tests for GGUF provenance sidecars: hashing, snapshots, commits and round-trips."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from msft import provenance
from msft.config import MsftError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never inherit a developer's llama.cpp commit from the shell."""
    monkeypatch.delenv(provenance.COMMIT_ENV, raising=False)


def _merged(tmp_path: Path, *, files: dict[str, int] | None = None) -> Path:
    """A merged directory with a config and the requested weight files."""
    merged = tmp_path / "merged-run"
    merged.mkdir()
    (merged / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    for name, size in (files or {}).items():
        (merged / name).write_bytes(b"w" * size)
    return merged


def _gguf(tmp_path: Path, payload: bytes = b"GGUF" + b"\0" * 12) -> Path:
    path = tmp_path / "model-run.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _strings(value: Any):
    """Every string nested in a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_sha256_file_matches_hashlib_across_chunks(tmp_path: Path) -> None:
    payload = os.urandom(20_000)
    path = tmp_path / "blob.gguf"
    path.write_bytes(payload)

    assert provenance.sha256_file(path, chunk=4096) == hashlib.sha256(payload).hexdigest()


def test_merged_snapshot_lists_sharded_weights_sorted_and_summed(tmp_path: Path) -> None:
    merged = _merged(
        tmp_path,
        files={
            "model-00002-of-00002.safetensors": 20,
            "model-00001-of-00002.safetensors": 10,
            "pytorch_model-00001-of-00002.bin": 5,
            "tokenizer.json": 99,
        },
    )

    snapshot = provenance.merged_snapshot(merged)

    assert snapshot["dir_name"] == "merged-run"
    assert snapshot["config_sha256"] == hashlib.sha256(
        (merged / "config.json").read_bytes()
    ).hexdigest()
    assert snapshot["weights_bytes"] == 35
    assert [entry["name"] for entry in snapshot["weights_files"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "pytorch_model-00001-of-00002.bin",
    ]
    assert all("/" not in entry["name"] for entry in snapshot["weights_files"])


def test_merged_snapshot_accepts_a_single_safetensors_file(tmp_path: Path) -> None:
    merged = _merged(tmp_path, files={"model.safetensors": 7})
    snapshot = provenance.merged_snapshot(merged)
    assert snapshot["weights_files"] == [{"name": "model.safetensors", "bytes": 7}]
    assert snapshot["weights_bytes"] == 7


def test_merged_snapshot_without_config_json_is_an_error(tmp_path: Path) -> None:
    merged = tmp_path / "merged-run"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"w")
    with pytest.raises(MsftError, match="config.json"):
        provenance.merged_snapshot(merged)


def test_llama_cpp_info_prefers_the_env_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "llama.cpp"
    directory.mkdir()
    (directory / provenance.COMMIT_FILE).write_text("from-file\n", encoding="utf-8")
    monkeypatch.setenv(provenance.COMMIT_ENV, "from-env\n")

    assert provenance.llama_cpp_info(directory) == {"commit": "from-env", "source": "env"}


def test_llama_cpp_info_falls_back_to_the_commit_file(tmp_path: Path) -> None:
    directory = tmp_path / "llama.cpp"
    directory.mkdir()
    (directory / provenance.COMMIT_FILE).write_text("from-file\n", encoding="utf-8")

    assert provenance.llama_cpp_info(directory) == {"commit": "from-file", "source": "file"}


def test_llama_cpp_info_asks_git_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "llama.cpp"
    (directory / ".git").mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "deadbeef\n", "")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    assert provenance.llama_cpp_info(directory) == {"commit": "deadbeef", "source": "git"}
    assert calls == [["git", "-C", str(directory), "rev-parse", "HEAD"]]


def test_llama_cpp_info_swallows_a_git_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "llama.cpp"
    (directory / ".git").mkdir(parents=True)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, provenance.GIT_TIMEOUT_S)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    assert provenance.llama_cpp_info(directory) == {"commit": None, "source": None}


def test_llama_cpp_info_is_null_without_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert provenance.llama_cpp_info(None) == {"commit": None, "source": None}
    directory = tmp_path / "no-git"
    directory.mkdir()
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 128, "", "not a repository"),
    )
    assert provenance.llama_cpp_info(directory) == {"commit": None, "source": None}


def test_build_write_load_compare_round_trips(tmp_path: Path) -> None:
    merged = _merged(tmp_path, files={"model.safetensors": 7})
    f16 = tmp_path / "model-run-f16.gguf"
    f16.write_bytes(b"GGUF f16")
    gguf = _gguf(tmp_path)

    record = provenance.build_record(
        config_name="run",
        model_id="org/tiny",
        revision=None,
        merged=provenance.merged_snapshot(merged),
        f16=f16,
        gguf=gguf,
        method="Q4_K_S",
        imatrix=False,
        llama_cpp={"commit": "deadbeef", "source": "git"},
    )
    written = provenance.write(gguf, record)

    loaded, reason = provenance.load(gguf)

    assert written == provenance.sidecar_path(gguf)
    assert reason == ""
    assert loaded == record
    assert record["schema"] == provenance.SCHEMA_VERSION
    assert record["f16"] == {"name": "model-run-f16.gguf", "bytes": 8}
    assert record["gguf"]["sha256"] == hashlib.sha256(gguf.read_bytes()).hexdigest()
    assert (
        provenance.compare(
            loaded,
            gguf=gguf,
            merged=merged,
            f16=f16,
            cfg_name="run",
            method="Q4_K_S",
        )
        == []
    )


def test_the_sidecar_contains_no_absolute_path(tmp_path: Path) -> None:
    merged = _merged(tmp_path, files={"model.safetensors": 7})
    gguf = _gguf(tmp_path)
    record = provenance.build_record(
        config_name="run",
        model_id="org/tiny",
        revision=None,
        merged=provenance.merged_snapshot(merged),
        f16=tmp_path / "model-run-f16.gguf",
        gguf=gguf,
        method="Q4_K_S",
        imatrix=False,
        llama_cpp={"commit": None, "source": None},
    )
    provenance.write(gguf, record)
    text = provenance.sidecar_path(gguf).read_text(encoding="utf-8")

    assert str(tmp_path) not in text
    for value in _strings(json.loads(text)):
        assert not value.startswith("/"), value


def test_load_reports_absent_corrupt_and_unsupported(tmp_path: Path) -> None:
    gguf = _gguf(tmp_path)
    assert provenance.load(gguf) == (None, "absent")

    sidecar = provenance.sidecar_path(gguf)
    sidecar.write_text("{not json", encoding="utf-8")
    record, reason = provenance.load(gguf)
    assert record is None and reason.startswith("corrupt:")

    sidecar.write_text(json.dumps({"schema": 7}), encoding="utf-8")
    assert provenance.load(gguf) == (None, "schema 7 unsupported")

    sidecar.write_text(json.dumps({"schema": 1}), encoding="utf-8")
    record, reason = provenance.load(gguf)
    assert record is None and "corrupt" in reason
