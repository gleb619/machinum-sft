"""Tests for `msft.remote`: argv construction, execution and redaction, fully offline."""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import UTC, datetime
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
    monkeypatch.delenv("MSFT_REMOTE_CONFIG", raising=False)
    # Hermetic default: preflight/run resolve the repo from the environment, never from a
    # developer's git-ignored configs/worker.local.yaml.
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")


def which_all(name: str) -> str:
    """Pretend every tool is installed."""
    return f"/usr/bin/{name}"


def fake_ssh(monkeypatch: pytest.MonkeyPatch, responder: object) -> FakeRun:
    fake = FakeRun(responder=responder)
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def fake_popen(
    monkeypatch: pytest.MonkeyPatch, lines: list[str], returncode: int = 0
) -> list[tuple[list[str], dict]]:
    """Stand in for `subprocess.Popen`; `stdout` yields the given lines."""
    calls: list[tuple[list[str], dict]] = []

    class Fake:
        def __init__(self, argv: list[str], **kwargs: object) -> None:
            calls.append((list(argv), dict(kwargs)))
            self.stdout = iter([f"{line}\n" for line in lines])
            self.returncode = returncode

        def wait(self) -> int:
            return returncode

    monkeypatch.setattr(subprocess, "Popen", Fake)
    return calls


def responder_ok(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Answer the four preflight probes like a healthy GPU box would."""
    command = argv[-1]
    if "nvidia-smi" in command:
        return subprocess.CompletedProcess(argv, 0, "NVIDIA L40S, 46068 MiB, 512 MiB\n", "")
    if "python --version" in command:
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
    assert fake.calls[0][0][6] == f"docker exec {remote.CONTAINER} {remote.GPU_COMMAND}"


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


@pytest.fixture
def data_tree(tmp_path: Path) -> Path:
    """A checkout with `data/raw`, `data/other` and a directory outside data."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "other").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "data" / "link").symlink_to(tmp_path / "out")
    return tmp_path


def test_sync_plan_argv_and_excludes(monkeypatch: pytest.MonkeyPatch, data_tree: Path) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/WORKSPACE/machinum-sft")
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, "data/raw", base_dir=data_tree)

    argv = result["argv"]
    assert argv[0] == "rsync"
    assert "-az" in argv and "--protect-args" in argv
    assert not any(part.startswith("--delete") for part in argv)
    pairs = set(zip(argv, argv[1:], strict=False))
    for pattern in remote.ALWAYS_EXCLUDES:
        assert ("--exclude", pattern) in pairs
    assert ("--exclude", "out") not in pairs
    assert ("--exclude", "data") not in pairs
    assert remote.TRANSPORT in argv
    assert "ServerAliveInterval=30" in remote.TRANSPORT
    assert argv[-2] == f"{data_tree / 'data' / 'raw'}/"
    assert argv[-1] == f"{remote.HOST_PLACEHOLDER}:WORKSPACE/machinum-sft/data/raw/"

    assert result["dest"] == "WORKSPACE/machinum-sft/data/raw/"
    assert result["local"] == "data/raw"
    assert result["delete"] is False
    assert result["repo_checked"] is False
    assert result["excludes"] == list(remote.ALWAYS_EXCLUDES)
    assert "remote" not in result
    assert result["executed"] is False
    assert result["ok"] is True
    assert result["host"] == MASKED
    assert "buildbox" not in json.dumps(result)
    json.dumps(result)


def test_sync_delete_uses_delete_after_only(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, "data", base_dir=data_tree, delete=True)

    argv = result["argv"]
    assert argv.count("--delete-after") == 1
    assert not any(
        part.startswith("--delete") and part != "--delete-after" for part in argv
    )
    assert result["delete"] is True
    assert result["dest"] == "proj/data/"


@pytest.mark.parametrize("local", ["/etc", "../x", "data/../out", "data/link", ".", "out"])
def test_sync_rejects_paths_outside_data(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path, local: str
) -> None:
    monkeypatch.setattr(subprocess, "run", Forbidden())
    with pytest.raises(remote.MsftError):
        remote.sync(HOST, local, base_dir=data_tree)


def test_sync_refusal_names_the_data_root(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", Forbidden())
    with pytest.raises(remote.MsftError, match="remote sync only sends paths under") as info:
        remote.sync(HOST, "out", base_dir=data_tree)
    assert "code goes through git pull" in str(info.value)


@pytest.mark.parametrize("local", ["data", "data/raw"])
def test_sync_accepts_directories_under_data(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path, local: str
) -> None:
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, local, base_dir=data_tree)
    expected = "proj/data/" if local == "data" else "proj/data/raw/"
    assert result["dest"] == expected


@pytest.mark.parametrize(
    ("repo", "relative", "dest"),
    [
        ("~/WORKSPACE/machinum-sft", "raw", "WORKSPACE/machinum-sft/data/raw/"),
        ("/srv/m", "raw", "/srv/m/data/raw/"),
        ("~/proj", ".", "proj/data/"),
    ],
)
def test_remote_dest_appends_data(repo: str, relative: str, dest: str) -> None:
    assert remote.remote_dest(repo, Path(relative)) == dest


@pytest.mark.parametrize("repo", ["~", "/", "$HOME/x", "a b", "x/../y", "x;rm"])
def test_remote_dest_rejects_unsafe_repos(repo: str) -> None:
    with pytest.raises(remote.MsftError):
        remote.remote_dest(repo, Path("raw"))


def test_msft_remote_dir_beats_the_worker_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    worker_yaml = configs / "worker.local.yaml"
    worker_yaml.write_text("repo_dir: from-yaml\n", encoding="utf-8")

    monkeypatch.setenv("MSFT_REMOTE_DIR", "from-env")
    assert remote.remote_repo_dir() == "from-env"

    monkeypatch.delenv("MSFT_REMOTE_DIR")
    assert remote.remote_repo_dir() == "from-yaml"

    worker_yaml.unlink()
    with pytest.raises(remote.MsftError, match="no remote repo directory"):
        remote.remote_repo_dir()


def test_sync_execute_probes_the_repo_before_rsync(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path
) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "ssh":
            return subprocess.CompletedProcess(argv, 0, "yes\n", "")
        return subprocess.CompletedProcess(argv, 0, "sent 12 bytes\n", "")

    fake = fake_ssh(monkeypatch, responder)
    result = remote.sync(HOST, "data/raw", base_dir=data_tree, execute=True)

    assert result["executed"] is True
    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["stdout"] == "sent 12 bytes"
    assert result["repo_checked"] is True

    assert fake.calls[0][0][0] == "ssh"
    assert fake.calls[0][0][-1].startswith("test -d ")
    assert ".git" in fake.calls[0][0][-1]
    rsync_calls = [call for call in fake.calls if call[0][0] == "rsync"]
    assert len(rsync_calls) == 1
    argv, kwargs = rsync_calls[0]
    assert argv[-1] == f"{HOST}:proj/data/raw/"
    assert kwargs["timeout"] == remote.RSYNC_TIMEOUT_S
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_sync_execute_refuses_without_a_remote_repo(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path
) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "rsync":
            raise AssertionError("rsync must not run when the repo probe fails")
        return subprocess.CompletedProcess(argv, 0, "no\n", "")

    fake = fake_ssh(monkeypatch, responder)
    with pytest.raises(remote.MsftError, match="not found or not a git clone"):
        remote.sync(HOST, "data/raw", base_dir=data_tree, execute=True)
    assert all(call[0][0] == "ssh" for call in fake.calls)


def test_sync_without_rsync_fails_softly(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path
) -> None:
    monkeypatch.setattr(
        remote.shutil, "which", lambda name: None if name == "rsync" else "/usr/bin/ssh"
    )
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, "data", base_dir=data_tree, execute=True)

    assert result["executed"] is False
    assert result["ok"] is False
    assert "rsync not found" in result["error"]


def test_sync_without_ssh_fails_softly(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path
) -> None:
    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.sync(HOST, "data", base_dir=data_tree, execute=True)

    assert result["ok"] is False
    assert "ssh not found" in result["error"]


def test_sync_failure_keeps_code_and_scrubs_stderr(
    monkeypatch: pytest.MonkeyPatch, data_tree: Path
) -> None:
    monkeypatch.setattr(remote.shutil, "which", which_all)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "ssh":
            return subprocess.CompletedProcess(argv, 0, "yes\n", "")
        return subprocess.CompletedProcess(
            argv, 23, "", f"rsync: connection to {HOST} refused\n"
        )

    fake_ssh(monkeypatch, responder)
    result = remote.sync(HOST, "data", base_dir=data_tree, execute=True)

    assert result["ok"] is False
    assert result["returncode"] == 23
    assert "exited 23" in result["error"]
    assert "buildbox" not in json.dumps(result)


# --- remote run -----------------------------------------------------------


def expected_command(cfg_name: str = "smoke") -> str:
    """The exact container command the contract asks for."""
    return (
        f"docker exec {remote.CONTAINER} python -m msft train "
        f"--config configs/{cfg_name}.yaml --execute"
    )


def test_container_command_never_activates_a_venv() -> None:
    command = remote.container_command("train", "configs/smoke.yaml")
    assert command.startswith("docker exec msft-worker ")
    assert "python -m msft train" in command
    assert "--config configs/smoke.yaml --execute" in command
    assert ".venv" not in command
    assert "cd " not in command


def test_run_remote_plan_composes_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.run_remote(RunConfig(name="smoke"), HOST)

    argv = result["argv"]
    assert argv[:5] == ["ssh", *SSH_OPTS]
    assert "ServerAliveInterval=30" in argv
    assert "ServerAliveCountMax=4" in argv
    assert argv[-2] == remote.HOST_PLACEHOLDER
    assert argv[-1] == remote.verdict_command(expected_command())
    assert argv[-1].startswith("docker exec msft-worker ")
    assert ".venv" not in argv[-1]
    assert "cd " not in argv[-1]
    assert result["command"].startswith("ssh -o BatchMode=yes -o ConnectTimeout=10 ")
    assert "ServerAliveInterval=30" in result["command"]
    assert result["executed"] is False
    assert result["failure"] is None
    assert result["ssh_returncode"] is None
    assert result["host"] == MASKED
    assert result["dir"] == "~/proj"
    assert result["config"] == "configs/smoke.yaml"
    assert result["timeout"] == "none"
    assert "buildbox" not in json.dumps(result)


def test_run_remote_plan_wraps_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.run_remote(RunConfig(name="smoke"), HOST, timeout_s=900)

    assert result["argv"][-1] == remote.verdict_command(
        f"timeout --signal=TERM --kill-after=30 900 {expected_command()}"
    )
    assert result["timeout"] == 900


def test_run_remote_streams_to_stderr_and_keeps_the_remote_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)
    monkeypatch.setattr(remote, "_container_running", lambda host: True)
    calls = fake_popen(monkeypatch, ["Loading weights", "[msft-rc] 3"], returncode=3)

    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Loading weights" in captured.err
    assert result["executed"] is True
    assert result["failure"] == "remote"
    assert result["returncode"] == 3
    assert result["ssh_returncode"] == 3
    assert result["ok"] is False
    assert result["stdout"] == ""
    argv, kwargs = calls[0]
    assert argv[:5] == ["ssh", *SSH_OPTS]
    assert argv[-2] == HOST
    assert "ServerAliveInterval=30" in argv
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["text"] is True
    assert kwargs["bufsize"] == 1


def test_run_remote_sentinel_255_is_remote_not_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)
    monkeypatch.setattr(remote, "_container_running", lambda host: True)
    fake_popen(monkeypatch, ["[msft-rc] 255"], returncode=255)

    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    assert result["failure"] == "remote"
    assert result["returncode"] == 255
    assert result["ssh_returncode"] == 255


def test_run_remote_sentinel_zero_is_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)
    monkeypatch.setattr(remote, "_container_running", lambda host: True)
    fake_popen(monkeypatch, ["done", "[msft-rc] 0"], returncode=0)

    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    assert result["failure"] is None
    assert result["returncode"] == 0
    assert result["ok"] is True


def test_run_remote_missing_sentinel_is_a_transport_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)
    monkeypatch.setattr(remote, "_container_running", lambda host: True)
    fake_popen(monkeypatch, [f"Permission denied ({HOST})"], returncode=255)

    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    assert result["failure"] == "transport"
    assert result["returncode"] is None
    assert result["ssh_returncode"] == 255
    assert "without a [msft-rc] verdict" in result["error"]
    assert "buildbox" not in json.dumps(result)
    assert MASKED in result["stderr"]
    assert "buildbox" not in capsys.readouterr().err


def test_run_remote_requires_a_running_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[-1].startswith("docker inspect"):
            return subprocess.CompletedProcess(argv, 0, "false\n", "")
        raise AssertionError("the run must not start when the container is down")

    fake_ssh(monkeypatch, responder)
    with pytest.raises(remote.MsftError, match="worker container is not running"):
        remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)


def test_run_remote_without_ssh_fails_softly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.run_remote(RunConfig(name="smoke"), HOST, execute=True)

    assert result["ok"] is False
    assert result["executed"] is False
    assert result["failure"] == "transport"
    assert "ssh not found" in result["error"]


def test_run_remote_honours_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_CONFIG", "/srv/runs/smoke.yaml")
    monkeypatch.setattr(subprocess, "run", Forbidden())
    result = remote.run_remote(RunConfig(name="ignored"), HOST)

    assert result["config"] == "/srv/runs/smoke.yaml"
    assert "--config /srv/runs/smoke.yaml --execute" in result["argv"][-1]


# --- tmux sessions --------------------------------------------------------

FROZEN = datetime(2026, 11, 9, 12, 34, 56, tzinfo=UTC)
SESSION = "msft-smoke-20261109T123456Z"


def test_session_name_is_deterministic_and_has_no_dots() -> None:
    name = remote.tmux_session_name("qwen3.5-4b.lora", now=FROZEN)
    assert name == "msft-qwen3_5-4b_lora-20261109T123456Z"
    assert "." not in name
    assert name.startswith("msft-")
    assert remote.TMUX_SESSION_RE.fullmatch(name)


@pytest.mark.parametrize(
    "name",
    ["worker", "msft-", "msft-a.b", "msft-a b", "msft-" + "x" * 81, "msft-a;rm"],
)
def test_validate_session_rejects_foreign_names(name: str) -> None:
    with pytest.raises(remote.MsftError):
        remote.validate_session(name)


def test_launch_tmux_plan_builds_a_parseable_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    result = remote.launch_tmux(RunConfig(name="smoke"), HOST, session=SESSION)
    command = result["argv"][-1]

    assert "command -v tmux" in command
    assert "docker exec -it msft-worker python -m msft train" in command
    assert f"has-session -t ={SESSION}" in command
    assert f"tee -a .worker/tmux/{SESSION}.log" in command
    assert "timeout --signal" not in command
    assert shlex.split(shlex.join(result["argv"])) == result["argv"]
    session_cmd = next(tok for tok in shlex.split(command) if tok.startswith("bash -c "))
    assert shlex.split(session_cmd)[:2] == ["bash", "-c"]
    assert "docker exec -it" in shlex.split(session_cmd)[2]
    assert subprocess.run(["bash", "-n", "-c", command], check=False).returncode == 0

    assert result["state"] == "plan"
    assert result["session"] == SESSION
    assert result["log"] == f".worker/tmux/{SESSION}.log"
    assert result["rc_file"] == f".worker/tmux/{SESSION}.rc"
    assert any("--follow" in hint for hint in result["hints"])
    assert "ServerAliveInterval=30" in result["argv"]
    assert "buildbox" not in json.dumps(result)


def test_launch_tmux_timeout_prefix_only_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    without = remote.launch_tmux(RunConfig(name="smoke"), HOST, session=SESSION)
    with_timeout = remote.launch_tmux(
        RunConfig(name="smoke"), HOST, session=SESSION, timeout_s=900
    )

    assert "timeout --signal" not in without["argv"][-1]
    assert (
        "timeout --signal=TERM --kill-after=30 900 docker exec -it msft-worker"
        in with_timeout["argv"][-1]
    )


def test_launch_tmux_missing_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake_ssh(
        monkeypatch,
        lambda argv: subprocess.CompletedProcess(
            argv, 127, "", f"{remote.TMUX_MISSING}\n"
        ),
    )

    with pytest.raises(remote.MsftError, match="tmux is not installed on the box"):
        remote.launch_tmux(
            RunConfig(name="smoke"), HOST, session=SESSION, force=True, execute=True
        )


def test_launch_tmux_refuses_a_second_gpu_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)
    fake = fake_ssh(
        monkeypatch,
        lambda argv: subprocess.CompletedProcess(argv, 0, "train-existing.json\n", ""),
    )

    with pytest.raises(remote.MsftError, match="two GPU jobs OOM"):
        remote.launch_tmux(RunConfig(name="smoke"), HOST, session=SESSION, execute=True)
    assert len(fake.calls) == 1
    assert "running/*.json" in fake.calls[0][0][-1]


def test_launch_tmux_force_skips_the_gpu_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(remote.shutil, "which", which_all)

    def responder(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if "ls -1" in argv[-1]:
            raise AssertionError("--force must skip the running/ probe")
        return subprocess.CompletedProcess(argv, 0, "", "")

    fake_ssh(monkeypatch, responder)
    result = remote.launch_tmux(
        RunConfig(name="smoke"), HOST, session=SESSION, force=True, execute=True
    )

    assert result["state"] == "started"
    assert result["ok"] is True
    assert result["executed"] is True


# --- logs -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "state", "rc", "tail"),
    [
        (
            f"{remote.TMUX_STATE_MARK} exited 0\n{remote.TMUX_TAIL_MARK}\nhello\n",
            "exited",
            0,
            "hello",
        ),
        (
            f"{remote.TMUX_STATE_MARK} running\n{remote.TMUX_TAIL_MARK}\nhello\n",
            "running",
            None,
            "hello",
        ),
        (f"{remote.TMUX_STATE_MARK} stopped\n{remote.TMUX_TAIL_MARK}\n", "stopped", None, ""),
        (f"{remote.TMUX_STATE_MARK} lost\n{remote.TMUX_TAIL_MARK}\n", "lost", None, ""),
    ],
)
def test_logs_tmux_parses_each_state(
    monkeypatch: pytest.MonkeyPatch, body: str, state: str, rc: int | None, tail: str
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    fake_ssh(monkeypatch, lambda argv: subprocess.CompletedProcess(argv, 0, body, ""))
    result = remote.logs_tmux(HOST, SESSION)

    assert result["state"] == state
    assert result["returncode"] == rc
    assert result["tail"] == tail
    assert result["ok"] is (state != "lost")
    assert f"has-session -t ={SESSION}" in result["command"]


def test_logs_tmux_wait_stops_at_the_rc_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    bodies = iter(
        [
            f"{remote.TMUX_STATE_MARK} running\n{remote.TMUX_TAIL_MARK}\n",
            f"{remote.TMUX_STATE_MARK} exited 7\n{remote.TMUX_TAIL_MARK}\nboom\n",
        ]
    )
    fake_ssh(monkeypatch, lambda argv: subprocess.CompletedProcess(argv, 0, next(bodies), ""))
    sleeps: list[float] = []

    result = remote.logs_tmux(
        HOST, SESSION, wait=True, sleep=sleeps.append
    )

    assert result["state"] == "exited"
    assert result["returncode"] == 7
    assert sleeps == [remote.WAIT_INTERVAL_S]


def test_logs_job_reads_the_done_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    body = (
        f"{remote.TMUX_STATE_MARK} done\n"
        '{"id": "train-1", "status": "done", "returncode": 0, "timed_out": false}\n'
        f"{remote.TMUX_TAIL_MARK}\nline one\nline two\n"
    )
    fake_ssh(monkeypatch, lambda argv: subprocess.CompletedProcess(argv, 0, body, ""))

    result = remote.logs_job(HOST, "train-1")

    assert result["state"] == "done"
    assert result["returncode"] == 0
    assert result["tail"] == "line one\nline two"
    assert "done/train-1.json" in result["command"]


def test_logs_job_wait_polls_until_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    bodies = iter(
        [
            f"{remote.TMUX_STATE_MARK} pending\n{remote.TMUX_TAIL_MARK}\n",
            f"{remote.TMUX_STATE_MARK} done\n"
            '{"status": "failed", "returncode": 1, "timed_out": true}\n'
            f"{remote.TMUX_TAIL_MARK}\nfailed\n",
        ]
    )
    fake_ssh(monkeypatch, lambda argv: subprocess.CompletedProcess(argv, 0, next(bodies), ""))
    sleeps: list[float] = []

    result = remote.logs_job(HOST, "train-1", wait=True, sleep=sleeps.append)

    assert result["state"] == "failed"
    assert result["timed_out"] is True
    assert result["ok"] is False
    assert sleeps == [remote.WAIT_INTERVAL_S]


# --- stop -----------------------------------------------------------------


def test_stop_tmux_sends_ctrl_c_then_reports_graceful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    fake = fake_ssh(
        monkeypatch, lambda argv: subprocess.CompletedProcess(argv, 0, "graceful\n", "")
    )

    result = remote.stop_tmux(HOST, SESSION, grace=5)

    command = fake.calls[0][0][-1]
    assert f"send-keys -t ={SESSION} C-c" in command
    assert f"kill-session -t ={SESSION}" in command
    assert f"has-session -t ={SESSION}" in command
    assert result["state"] == "graceful"
    assert result["ok"] is True


def test_stop_tmux_missing_session_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    fake_ssh(
        monkeypatch,
        lambda argv: subprocess.CompletedProcess(
            argv, 3, "", f"{remote.TMUX_NO_SESSION}\n"
        ),
    )

    result = remote.stop_tmux(HOST, SESSION)

    assert result["state"] == "missing"
    assert result["ok"] is False
    assert result["error"] == "no such session"


def test_stop_tmux_refuses_a_foreign_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSFT_REMOTE_DIR", "~/proj")
    monkeypatch.setattr(subprocess, "run", Forbidden())
    with pytest.raises(remote.MsftError, match="invalid tmux session"):
        remote.stop_tmux(HOST, "someone-elses-session")
