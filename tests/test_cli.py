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
quantize:
  output: out/model-tiny.gguf
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

    def run(
        cfg: Any, *, execute: bool = False, inspect_mask: bool = False
    ) -> dict[str, Any]:
        calls["run"] = execute
        return {"stage": "train", "executed": execute}

    stub = types.ModuleType("msft.train")
    stub.plan = plan  # type: ignore[attr-defined]
    stub.run = run  # type: ignore[attr-defined]
    stub.build_script = lambda cfg: "# script\n"  # type: ignore[attr-defined]
    stub.IGNORED_KNOBS = {}  # type: ignore[attr-defined]
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


# --- remote exit codes ----------------------------------------------------


def _remote_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_HOST", "user@buildbox.test")


def test_remote_preflight_exits_one_when_a_check_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import msft.remote as remote

    _remote_env(monkeypatch)
    monkeypatch.setattr(
        remote, "preflight", lambda host, execute=False: [
            {"name": "ssh", "status": "fail", "detail": "unreachable"}
        ]
    )
    code = main(["--json", "remote", "preflight", "--execute"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"][0]["status"] == "fail"


def test_remote_sync_exit_code_follows_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import msft.remote as remote

    _remote_env(monkeypatch)
    path = write_config(tmp_path)
    monkeypatch.setattr(
        remote,
        "sync",
        lambda host, local, *, base_dir=".", delete=False, execute=False: {
            "executed": True,
            "ok": False,
            "returncode": 23,
            "stderr": "rsync failed",
        },
    )
    code = main(["--config", str(path), "remote", "sync", "--execute"])
    assert code == 1
    assert "rsync failed" in capsys.readouterr().out


def test_remote_sync_defaults_local_to_the_config_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import msft.remote as remote

    _remote_env(monkeypatch)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        remote,
        "sync",
        lambda host, local, *, base_dir=".", delete=False, execute=False: (
            seen.update(local=local, base_dir=str(base_dir)) or {"executed": False}
        ),
    )
    path = write_config(tmp_path)
    assert main(["--config", str(path), "remote", "sync"]) == 0
    assert seen["local"] == "data/raw"
    assert seen["base_dir"] == str(tmp_path)


def test_remote_sync_remote_flag_is_a_removed_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import msft.remote as remote

    _remote_env(monkeypatch)
    monkeypatch.setattr(remote, "sync", lambda *args, **kwargs: {"executed": False})
    code = main(["--config", str(write_config(tmp_path)), "remote", "sync", "--remote", "x"])
    assert code == 2
    assert "--remote was removed" in capsys.readouterr().err


def test_remote_run_plan_still_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import msft.remote as remote

    _remote_env(monkeypatch)
    path = write_config(tmp_path)
    monkeypatch.setattr(
        remote,
        "run_remote",
        lambda cfg, host, execute=False, timeout_s=None: {
            "executed": False,
            "host": "masked",
        },
    )
    assert main(["--config", str(path), "remote", "run"]) == 0


def test_worker_signal_prints_the_parsed_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _remote_env(monkeypatch)
    code = main(
        [
            "--json",
            "--config",
            str(write_config(tmp_path)),
            "worker",
            "signal",
            "doctor",
            "--dir",
            str(tmp_path / ".worker"),
            "--timeout",
            "15m",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["timeout"] == "900s"


def test_a_bad_timeout_is_a_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _remote_env(monkeypatch)
    code = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "worker",
            "signal",
            "doctor",
            "--dir",
            str(tmp_path / ".worker"),
            "--timeout",
            "2x",
        ]
    )
    assert code == 2
    assert "invalid --timeout '2x'" in capsys.readouterr().err


def test_remote_status_exits_one_when_ssh_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msft.deploy as deploy

    monkeypatch.setattr(deploy, "load_spec", lambda: deploy.WorkerSpec(host="user@box"))
    monkeypatch.setattr(
        deploy,
        "status",
        lambda spec, execute=False: {
            "executed": True,
            "ok": False,
            "returncode": 255,
            "stdout": "",
            "stderr": "connection refused",
        },
    )
    assert main(["--json", "remote", "status", "--execute"]) == 1


def test_deploy_stops_before_up_when_the_pull_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msft.deploy as deploy

    monkeypatch.setattr(deploy, "load_spec", lambda: deploy.WorkerSpec(host="user@box"))
    monkeypatch.setattr(
        deploy,
        "pull",
        lambda spec, execute=False: {"executed": True, "ok": False, "returncode": 128},
    )

    def boom(*args: object, **kwargs: object) -> dict[str, Any]:
        raise AssertionError("up must not run after a failed pull")

    monkeypatch.setattr(deploy, "up", boom)
    assert main(["--json", "remote", "deploy", "--execute"]) == 1


def test_deploy_up_failure_is_exit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    import msft.deploy as deploy

    monkeypatch.setattr(deploy, "load_spec", lambda: deploy.WorkerSpec(host="user@box"))
    monkeypatch.setattr(
        deploy,
        "pull",
        lambda spec, execute=False: {"executed": True, "ok": True, "returncode": 0},
    )
    monkeypatch.setattr(
        deploy,
        "up",
        lambda spec, execute=False, build=True: {
            "executed": True,
            "ok": False,
            "returncode": 1,
        },
    )
    assert main(["--json", "remote", "deploy", "--execute"]) == 1


# --- unexpected failures keep the JSON contract --------------------------


def test_internal_error_prints_one_failure_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import msft.worker as worker

    def crash(directory: str) -> dict[str, Any]:
        raise RuntimeError("boom from a library")

    monkeypatch.setattr(worker, "status", crash)
    code = main(["--json", "worker", "status", "--dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "internal RuntimeError: boom from a library" in payload["error"]
    assert "Traceback" in captured.err


def test_msft_error_prints_one_failure_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--json", "--config", str(tmp_path / "absent.yaml"), "config", "show"]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out)["ok"] is False
