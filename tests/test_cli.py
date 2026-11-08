"""Tests for the CLI: parsing, dispatch, exit codes and JSON output."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import msft.cli as cli
from msft.cli import main

SRC = Path(__file__).resolve().parents[1] / "src"

TINY_CONFIG = """\
name: tiny
model:
  id: org/tiny
  dtype: bfloat16
train:
  max_seq_length: 1024
data:
  val_ratio: 0.05
"""


def write_config(tmp_path: Path, text: str = TINY_CONFIG) -> Path:
    """Write a tiny run config and return its path."""
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def only_object(stdout: str) -> dict[str, Any]:
    """Parse stdout as exactly one JSON object with no extra output."""
    assert stdout.startswith("{"), stdout
    assert stdout.rstrip("\n").count("\n") == 0, stdout
    parsed = json.loads(stdout)
    assert isinstance(parsed, dict)
    return parsed


# --- parsing --------------------------------------------------------------


def test_root_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "usage: msft" in captured.out.lower()
    for command in ("doctor", "train", "merge", "quantize", "eval", "remote", "config"):
        assert command in captured.out


def test_subcommand_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["remote", "--help"]) == 0
    captured = capsys.readouterr()
    assert "preflight" in captured.out
    assert main(["data", "build", "--help"]) == 0


def test_unknown_command_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["frobnicate"]) != 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_no_command_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) != 0


# --- config show ----------------------------------------------------------


def test_config_show_prints_redacted_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path)
    assert main(["--config", str(path), "config", "show"]) == 0
    captured = capsys.readouterr()
    assert "org/tiny" in captured.out
    assert "max_seq_length: 1024" in captured.out
    assert captured.err == ""


def test_config_show_json_is_one_object(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_config(tmp_path)
    assert main(["config", "show", "--json", "--config", str(path)]) == 0
    captured = capsys.readouterr()
    payload = only_object(captured.out)
    assert payload["name"] == "tiny"
    assert payload["model"]["id"] == "org/tiny"
    assert payload["train"]["max_seq_length"] == 1024


def test_missing_config_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "absent.yaml"
    assert main(["--config", str(missing), "config", "show"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: config not found")


def test_error_message_is_redacted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_config(
        tmp_path,
        "model:\n  id: org/tiny\n  10.20.30.40: 1\n",
    )
    assert main(["--config", str(path), "config", "show"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "10.20.30.40" not in captured.err
    assert "<host>" in captured.err


# --- not implemented ------------------------------------------------------


def test_impl_names_the_missing_module() -> None:
    with pytest.raises(cli._NotImplementedYet) as excinfo:
        cli._impl("msft.definitely_not_a_module")
    assert "msft.definitely_not_a_module" in excinfo.value.message
    assert "not implemented yet" in excinfo.value.message


def test_missing_module_exits_three(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str) -> Any:
        exc = ModuleNotFoundError(f"No module named {name!r}")
        exc.name = name
        raise exc

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    path = write_config(tmp_path)
    assert main(["--config", str(path), "train"]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "msft.train is not implemented yet" in captured.err


def test_missing_module_json_is_one_object(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str) -> Any:
        exc = ModuleNotFoundError(f"No module named {name!r}")
        exc.name = name
        raise exc

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    path = write_config(tmp_path)
    assert main(["--json", "--config", str(path), "data", "build"]) == 3
    payload = only_object(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"
    assert payload["module"] == "msft.data"


def test_real_missing_module_exits_three(tmp_path: Path) -> None:
    """The genuine path, while the sibling modules have not landed yet."""
    if importlib.util.find_spec("msft.train") is not None:
        pytest.skip("msft.train exists in this checkout")
    assert main(["--config", str(write_config(tmp_path)), "train"]) == 3


# --- dispatch through a stub module --------------------------------------


def test_train_prints_plan_from_module(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def plan(cfg: Any) -> dict[str, Any]:
        calls["plan"] = cfg.name
        return {"stage": "train", "model": cfg.model.id, "execute": False}

    def run(cfg: Any, *, execute: bool = False) -> dict[str, Any]:
        calls["run"] = execute
        return {"stage": "train", "executed": execute}

    stub = types.ModuleType("msft.train")
    stub.plan = plan  # type: ignore[attr-defined]
    stub.run = run  # type: ignore[attr-defined]
    stub.build_script = lambda cfg: "# script\n"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "msft.train", stub)

    path = write_config(tmp_path)
    assert main(["--config", str(path), "train"]) == 0
    captured = capsys.readouterr()
    assert "train plan" in captured.out
    assert calls["plan"] == "tiny"
    assert "run" not in calls

    assert main(["--config", str(path), "--json", "train"]) == 0
    payload = only_object(capsys.readouterr().out)
    assert payload["model"] == "org/tiny"

    assert main(["--config", str(path), "train", "--execute"]) == 0
    capsys.readouterr()
    assert calls["run"] is True

    assert main(["--config", str(path), "train", "--script"]) == 0
    assert "# script" in capsys.readouterr().out

    assert main(["--config", str(path), "train", "--dry-run"]) == 0
    assert "train plan" in capsys.readouterr().out
    assert calls["run"] is True


def test_dry_run_conflicts_with_execute(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(write_config(tmp_path)), "train", "--dry-run", "--execute"]) == 2
    assert "mutually exclusive" in capsys.readouterr().err


# --- entry point ----------------------------------------------------------


def test_python_m_msft_help_exits_zero() -> None:
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run(
        [sys.executable, "-m", "msft", "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0
    assert "usage: msft" in proc.stdout.lower()
    assert proc.stderr == ""
