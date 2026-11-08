"""Tests for `msft.remote`: argv construction, execution and redaction, fully offline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from msft import remote
from msft.config import RunConfig
from msft.doctor import Check

HOST = "user@buildbox.test"
MASKED = "user@bu***"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


class FakeRun:
    """Records argv/kwargs and answers with a canned CompletedProcess, an error, or a responder."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        error: BaseException | None = None,
        responder: object | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.error = error
        self.responder = responder
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        if self.error is not None:
            raise self.error
        if self.responder is not None:
            return self.responder(argv)  # type: ignore[operator]
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)


class Forbidden(FakeRun):
    """Fails the test if anything tries to start a process."""

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"subprocess.run must not be called, got {argv!r}")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep remote paths out of the developer's environment."""
    monkeypatch.delenv("MSFT_REMOTE_DIR", raising=False)
    monkeypatch.delenv("MSFT_REMOTE_CONFIG", raising=False)


def which_all(name: str) -> str:
    """Pretend every tool is installed."""
    return f"/usr/bin/{name}"


def fake_ssh(monkeypatch: pytest.MonkeyPatch, responder: object) -> FakeRun:
    fake = FakeRun(responder=responder)
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def responder_ok(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Answer the four preflight probes like a healthy GPU box would."""
    command = argv[-1]
    if command.startswith("nvidia-smi"):
        return subprocess.CompletedProcess(argv, 0, "NVIDIA L40S, 46068 MiB, 512 MiB\n", "")
    if command.startswith("python3"):
        return subprocess.CompletedProcess(argv, 0, "Python 3.11.8\n", "")
    if command.startswith("df"):
        return subprocess.CompletedProcess(argv, 0, "Avail\n400G\n", "")
    if command.startswith("test -d"):
        return subprocess.CompletedProcess(argv, 0, "yes\n", "")
    raise AssertionError(f"unexpected remote command: {command}")


def by_name(checks: list[Check]) -> dict[str, Check]:
    """Index checks by name."""
    return {check.name: check for check in checks}


# --- masking --------------------------------------------------------------


def test_mask_host_keeps_user_and_two_characters() -> None:
    assert remote.mask_host(HOST) == MASKED
    assert remote.mask_host("buildbox.test") == "bu***"
    assert remote.mask_host("ab") == "***"


# --- preflight ------------------------------------------------------------


def test_preflight_plan_runs_nothing_and_hides_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)
    monkeypatch.setattr(subprocess, "run", Forbidden())
    checks = remote.preflight(HOST)

    assert [check.name for check in checks] == ["ssh", "cuda", "python", "disk", "remote_dir"]
    assert all(check.status == "ok" for check in checks)
    joined = "\n".join(check.detail for check in checks)
    assert joined.count("plan: ") == 4
    assert "-o BatchMode=yes" in joined
    assert "ConnectTimeout=10" in joined
    assert "--format=csv,noheader" in joined
    assert "df -h --output=avail /" in joined
    assert remote.HOST_PLACEHOLDER in joined
    assert "buildbox" not in joined


def test_preflight_without_ssh_is_a_fail_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", Forbidden())
    checks = remote.preflight(HOST)

    assert len(checks) == 5
    assert checks[0].name == "ssh"
    assert checks[0].status == "fail"
    assert "ssh not found" in checks[0].detail
    assert all(check.status == "warn" for check in checks[1:])


def test_preflight_execute_probes_every_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake = fake_ssh(monkeypatch, responder_ok)
    checks = remote.preflight(HOST, execute=True)
    named = by_name(checks)

    assert [check.name for check in checks] == ["ssh", "cuda", "python", "disk", "remote_dir"]
    assert all(check.status == "ok" for check in checks)
    assert "L40S" in named["cuda"].detail
    assert "3.11.8" in named["python"].detail
    assert "400.0 GiB" in named["disk"].detail
    assert "exists" in named["remote_dir"].detail

    assert len(fake.calls) == 4
    for argv, kwargs in fake.calls:
        assert argv[:5] == ["ssh", *SSH_OPTS]
        assert argv[5] == HOST
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == remote.PROBE_TIMEOUT_S
    assert fake.calls[0][0][6] == remote.GPU_COMMAND


def test_preflight_execute_flags_old_remote_python(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "", "Python 3.9.2\n")

    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake_ssh(monkeypatch, responder)
    check = by_name(remote.preflight(HOST, execute=True))["python"]
    assert check.status == "fail"
    assert "3.9.2" in check.detail


def test_preflight_execute_missing_dir_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "no\n", "")

    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake_ssh(monkeypatch, responder)
    check = by_name(remote.preflight(HOST, execute=True))["remote_dir"]
    assert check.status == "warn"
    assert "msft remote sync" in check.detail


def test_preflight_timeout_is_warn_and_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake = FakeRun(error=subprocess.TimeoutExpired("ssh", remote.PROBE_TIMEOUT_S))
    monkeypatch.setattr(subprocess, "run", fake)
    checks = remote.preflight(HOST, execute=True)
    assert all(check.status == "warn" for check in checks[1:])
    assert "timed out" in checks[1].detail


def test_preflight_ssh_error_does_not_leak_host(monkeypatch: pytest.MonkeyPatch) -> None:
    message = f"ssh: Could not resolve hostname {HOST}: nodename nor servname\n"
    fake = FakeRun(stderr=message, returncode=255)
    monkeypatch.setattr(remote.shutil, "which", which_all)
    monkeypatch.setattr(subprocess, "run", fake)
    checks = remote.preflight(HOST, execute=True)
    joined = "\n".join(check.detail for check in checks)
    assert "Could not resolve" in joined
    assert "buildbox" not in joined
    assert MASKED in joined


# --- sync -----------------------------------------------------------------


def test_sync_plan_argv_and_excludes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, tmp_path, "~/proj")

    argv = result["argv"]
    assert argv[0] == "rsync"
    assert "-az" in argv and "--delete-after" in argv
    pairs = set(zip(argv, argv[1:], strict=False))
    for pattern in (".git", ".venv", "out", "data"):
        assert ("--exclude", pattern) in pairs
    assert remote.TRANSPORT in argv
    assert argv[-2] == f"{tmp_path}/"
    assert argv[-1] == f"{remote.HOST_PLACEHOLDER}:~/proj/"

    assert result["executed"] is False
    assert result["ok"] is True
    assert result["host"] == MASKED
    assert result["remote"] == "~/proj"
    assert "buildbox" not in json.dumps(result)
    json.dumps(result)  # the whole plan is JSON-serialisable


def test_sync_execute_runs_rsync_with_real_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake = FakeRun(stdout="sent 12 bytes\n")
    monkeypatch.setattr(subprocess, "run", fake)
    result = remote.sync(HOST, tmp_path, "~/proj", execute=True)

    assert result["executed"] is True
    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"] == "sent 12 bytes"
    argv, kwargs = fake.calls[0]
    assert argv[-1] == f"{HOST}:~/proj/"
    assert argv[-2] == f"{tmp_path}/"
    assert kwargs["timeout"] == remote.RSYNC_TIMEOUT_S
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_sync_without_rsync_fails_softly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        remote.shutil, "which", lambda name: None if name == "rsync" else "/usr/bin/ssh"
    )
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, tmp_path, "~/proj", execute=True)

    assert result["executed"] is False
    assert result["ok"] is False
    assert "rsync not found" in result["error"]


def test_sync_without_ssh_fails_softly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, tmp_path, "~/proj", execute=True)

    assert result["ok"] is False
    assert "ssh not found" in result["error"]


def test_sync_failure_keeps_code_and_scrubs_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake = FakeRun(stderr=f"rsync: connection to {HOST} refused\n", returncode=23)
    monkeypatch.setattr(subprocess, "run", fake)
    result = remote.sync(HOST, tmp_path, "~/proj", execute=True)

    assert result["ok"] is False
    assert result["returncode"] == 23
    assert "exited 23" in result["error"]
    assert "buildbox" not in json.dumps(result)


# --- remote run -----------------------------------------------------------


def expected_command(cfg_name: str = "smoke", directory: str = "~/proj") -> str:
    """The exact remote shell line the contract asks for."""
    return (
        f'cd "$HOME"/{directory[2:]} && . .venv/bin/activate && '
        f"python -m msft train --config configs/{cfg_name}.yaml --execute"
    )


def test_run_remote_plan_composes_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.run_remote(RunConfig(name="smoke"), HOST)

    argv = result["argv"]
    assert argv[:5] == ["ssh", *SSH_OPTS]
    assert argv[5] == remote.HOST_PLACEHOLDER
    assert argv[6] == expected_command()
    assert result["command"].startswith("ssh -o BatchMode=yes -o ConnectTimeout=10 ")
    assert result["executed"] is False
    assert result["host"] == MASKED
    assert result["dir"] == "~/proj"
    assert result["config"] == "configs/smoke.yaml"
    assert "buildbox" not in json.dumps(result)


def test_run_remote_execute_uses_real_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake = FakeRun(stdout="step 10/10 done\n")
    monkeypatch.setattr(subprocess, "run", fake)
    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    assert result["executed"] is True
    assert result["ok"] is True
    assert result["returncode"] == 0
    argv, kwargs = fake.calls[0]
    assert argv[:5] == ["ssh", *SSH_OPTS]
    assert argv[5] == HOST
    assert argv[6] == expected_command()
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert "done" in result["stdout"]


def test_run_remote_without_ssh_fails_softly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    assert result["ok"] is False
    assert result["executed"] is False
    assert "ssh not found" in result["error"]


def test_run_remote_honours_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_CONFIG", "/srv/runs/smoke.yaml")
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.run_remote(RunConfig(name="ignored"), HOST)

    assert result["config"] == "/srv/runs/smoke.yaml"
    assert "--config /srv/runs/smoke.yaml --execute" in result["argv"][6]


def test_run_remote_failure_scrubs_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake = FakeRun(stderr=f"Permission denied ({HOST})\n", returncode=255)
    monkeypatch.setattr(subprocess, "run", fake)
    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    assert result["ok"] is False
    assert result["returncode"] == 255
    assert "buildbox" not in json.dumps(result)
    assert MASKED in result["stderr"]
