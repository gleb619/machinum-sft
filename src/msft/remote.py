"""Remote GPU orchestration over ssh and rsync.

Every entry point is a plan until `execute=True`: the returned mapping (or the `Check` details)
describes exactly what would run, and the host is never echoed verbatim — plans use the
`$MSFT_REMOTE_HOST` placeholder and human strings go through `_scrub`, so printing a plan cannot
leak the box. A missing `ssh`/`rsync` binary is reported, never raised.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import MsftError, RunConfig, redact
from .doctor import (
    DISK_FAIL_GIB,
    DISK_WARN_GIB,
    GPU_QUERY,
    PYTHON_MIN,
    Check,
    describe_gpus,
    parse_gpu_output,
)

SSH = "ssh"
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
SSH_RUN_OPTS = (*SSH_OPTS, "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4")
TRANSPORT = " ".join((SSH, *SSH_RUN_OPTS))
HOST_PLACEHOLDER = "$MSFT_REMOTE_HOST"
CONTAINER = "msft-worker"
CONTAINER_REPO = "/workspace"
WORKER_CONFIG_CANDIDATES = ("configs/worker.local.yaml", "worker.local.yaml")
ALWAYS_EXCLUDES = (
    "/.git/",
    "/.venv/",
    "/.worker/",
    "/out/",
    "/configs/*.local.yaml",
    ".env",
    ".env.*",
    "__pycache__/",
    ".git",
    ".venv",
)
REMOTE_REPO_RE = re.compile(r"^(?:~/)?[A-Za-z0-9._@+/-]+$")
PROBE_NAMES = ("cuda", "python", "disk", "remote_dir")
GPU_COMMAND = " ".join(GPU_QUERY)
DISK_COMMAND = "df -h --output=avail /"
PROBE_TIMEOUT_S = 20
RSYNC_TIMEOUT_S = 1800
MAX_OUTPUT = 4000
TAIL_LINES = 200
WAIT_INTERVAL_S = 10.0
TMUX_STATE_DIR = ".worker/tmux"
TMUX_MISSING = "MSFT_ERR=tmux-missing"
TMUX_NO_SESSION = "MSFT_ERR=no-such-session"
TMUX_STATE_MARK = "__MSFT_STATE__"
TMUX_TAIL_MARK = "__MSFT_TAIL__"
SENTINEL_RE = re.compile(r"\[msft-rc\]\s+(-?\d+)")
TMUX_SESSION_RE = re.compile(r"^msft-[A-Za-z0-9_-]{1,80}$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMGTP]?)", re.IGNORECASE)
_SIZE_GIB = {
    "": 1.0 / 1024**3,
    "K": 1.0 / 1024**2,
    "M": 1.0 / 1024,
    "G": 1.0,
    "T": 1024.0,
    "P": 1024.0 * 1024.0,
}

__all__ = [
    "ALWAYS_EXCLUDES",
    "CONTAINER",
    "CONTAINER_REPO",
    "HOST_PLACEHOLDER",
    "SSH_OPTS",
    "SSH_RUN_OPTS",
    "build_sync_argv",
    "container_command",
    "job_id",
    "launch_tmux",
    "logs_job",
    "logs_tmux",
    "mask_host",
    "preflight",
    "remote_config",
    "remote_dest",
    "remote_repo_dir",
    "resolve_sync_source",
    "run_remote",
    "ssh_argv",
    "ssh_run_argv",
    "stop_tmux",
    "sync",
    "tmux_session_name",
    "validate_session",
    "verdict_command",
]


# --- redaction ------------------------------------------------------------


def mask_host(host: str) -> str:
    """Mask an ssh target for display, keeping the user and two hostname characters."""
    user, sep, name = host.rpartition("@")
    if not sep:
        user, name = "", host
    head = name[:2] if len(name) > 2 else ""
    prefix = f"{user}@" if sep else ""
    return f"{prefix}{head}***"


def _scrub(text: str, host: str) -> str:
    """Replace the real target (and hostname) with its mask, then apply config.redact."""
    masked = mask_host(host)
    _user, sep, name = host.rpartition("@")
    for needle in (host, name if sep else ""):
        if needle and len(needle) >= 4:
            text = text.replace(needle, masked)
    return redact(text)


def _first_line(text: str, fallback: str) -> str:
    """First non-empty line of tool output, else `fallback`."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return fallback


def _tail(text: str | None, limit: int = MAX_OUTPUT) -> str:
    """Trim captured output to its last `limit` characters."""
    if not text:
        return ""
    stripped = text.strip()
    return stripped if len(stripped) <= limit else "..." + stripped[-limit:]


# --- remote paths and commands --------------------------------------------


def remote_repo_dir() -> str:
    """Repo directory on the box: `$MSFT_REMOTE_DIR`, else `repo_dir` from the worker config."""
    env = os.environ.get("MSFT_REMOTE_DIR")
    if env:
        return env
    for candidate in WORKER_CONFIG_CANDIDATES:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - malformed local file
            raise MsftError(f"cannot read {path}: {exc}") from exc
        if isinstance(data, dict) and data.get("repo_dir"):
            return str(data["repo_dir"])
    raise MsftError(
        "no remote repo directory: set repo_dir in configs/worker.local.yaml "
        "or export MSFT_REMOTE_DIR"
    )


def remote_host() -> str:
    """The ssh target: `$MSFT_REMOTE_HOST`, else `host` from the worker config.

    One source of truth. `remote run` used to demand the environment variable while every other
    command read the git-ignored worker config, so the same box was reachable from one command and
    invisible to the next.
    """
    env = os.environ.get("MSFT_REMOTE_HOST", "").strip()
    if env:
        return env
    for candidate in WORKER_CONFIG_CANDIDATES:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - malformed local file
            raise MsftError(f"cannot read {path}: {exc}") from exc
        if isinstance(data, dict) and str(data.get("host") or "").strip():
            return str(data["host"]).strip()
    raise MsftError(
        "no remote host: set host in configs/worker.local.yaml or export MSFT_REMOTE_HOST"
    )


def remote_config(cfg: RunConfig) -> str:
    """Config path on the remote host: `$MSFT_REMOTE_CONFIG`, else `configs/<name>.yaml`."""
    return os.environ.get("MSFT_REMOTE_CONFIG") or f"configs/{cfg.name}.yaml"


def remote_path(path: str) -> str:
    """Quote a remote path for a shell, keeping a leading `~` expandable."""
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return f'"$HOME"/{shlex.quote(path[2:])}'
    return shlex.quote(path)


def container_command(
    kind: str,
    config_rel: str | None,
    extra_args: Sequence[str] = (),
    timeout_s: int | None = None,
    *,
    tty: bool = False,
) -> str:
    """`docker exec [it] msft-worker python -m msft <kind> --config … --execute`, quoted.

    The builder stays wrapper-free: `run_remote` applies any `timeout` around the result.
    `tty` only controls `-i`, which keeps docker attached to a stdin we feed from /dev/null.
    A terminal (`-t`) is never requested: a detached tmux pane has no foreground process group,
    so the first read from its pty stops docker dead with SIGTTIN (state `T`), and `-t` refuses a
    redirected stdin outright ("cannot attach stdin to a TTY-enabled container").
    """
    action = "data build" if kind == "data" else kind
    parts = ["docker", "exec"]
    if tty:
        parts.append("-i")
    parts += [CONTAINER, "python", "-m", "msft", *action.split()]
    if config_rel:
        parts += ["--config", config_rel]
    parts += list(extra_args)
    if "--execute" not in parts:
        parts.append("--execute")
    return shlex.join(parts)


def verdict_command(command: str) -> str:
    """Wrap a remote command so its real exit code arrives in the stream, not in ssh's."""
    return f"{command} ; rc=$? ; printf '\\n[msft-rc] %s\\n' \"$rc\" ; exit \"$rc\""


def ssh_argv(host: str, command: str) -> list[str]:
    """Build a probe ssh argv that never prompts for a password and gives up in 10s."""
    return [SSH, *SSH_OPTS, host, command]


def ssh_run_argv(host: str, command: str) -> list[str]:
    """Build a long-running ssh argv; keepalives detect a dropped link within ~2 minutes."""
    return [SSH, *SSH_RUN_OPTS, host, command]


def _display(argv: list[str]) -> str:
    """Shell-style join that leaves the host placeholder expandable."""
    parts = [
        part
        if part == HOST_PLACEHOLDER or part.startswith(f"{HOST_PLACEHOLDER}:")
        else shlex.quote(part)
        for part in argv
    ]
    return " ".join(parts)


def _with_slash(path: str) -> str:
    return path if path.endswith("/") else f"{path}/"


def build_sync_argv(
    host: str, local: Path | str, dest: str, *, delete: bool = False
) -> list[str]:
    """Build the rsync argv sending `local` to `host:dest` with the always-on excludes."""
    argv = ["rsync", "-az", "--protect-args"]
    if delete:
        argv.append("--delete-after")
    for pattern in ALWAYS_EXCLUDES:
        argv.extend(["--exclude", pattern])
    argv.extend(["-e", TRANSPORT, _with_slash(str(local)), f"{host}:{_with_slash(dest)}"])
    return argv


def resolve_sync_source(local: str | Path, base_dir: str | Path) -> tuple[Path, Path]:
    """Resolve `local` under `<base_dir>/data/`; return (source, path relative to data)."""
    base = Path(base_dir)
    data_root = (base / "data").resolve()
    candidate = Path(local)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise MsftError(f"sync source not found: {local}")
    if not candidate.is_dir():
        raise MsftError("sync source must be a directory")
    try:
        relative = candidate.relative_to(data_root)
    except ValueError as exc:
        raise MsftError(
            f"refusing to sync {local}: remote sync only sends paths under {data_root}/ "
            "(code goes through git pull, never rsync)"
        ) from exc
    return candidate, relative


def remote_dest(repo: str, relative: Path) -> str:
    """Destination inside the remote repo: `<repo>/data/<relative>/`, with no `~` expansion."""
    if (
        not repo
        or repo in ("~", "/")
        or ".." in repo.split("/")
        or REMOTE_REPO_RE.fullmatch(repo) is None
    ):
        raise MsftError(
            f"invalid remote repo directory {repo!r}: use a relative path, ~/relative/path "
            "or /absolute/path"
        )
    base = repo[2:] if repo.startswith("~/") else repo
    base = base.rstrip("/")
    rel = "" if relative == Path(".") else relative.as_posix()
    return f"{base}/data/{rel}/" if rel else f"{base}/data/"


# --- ssh execution --------------------------------------------------------


def _run_ssh(host: str, command: str) -> tuple[int, str, str]:
    """Run one short ssh probe; returns (returncode, stdout, stderr) and never raises."""
    try:
        proc = subprocess.run(
            ssh_argv(host, command),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {PROBE_TIMEOUT_S}s"
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _run_ssh_run(
    host: str, command: str, *, timeout_s: float = PROBE_TIMEOUT_S
) -> tuple[int, str, str]:
    """Run one ssh control command with keepalives; returns (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ssh_run_argv(host, command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout_s:g}s"
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _container_exec(command: str) -> str:
    """Run one command inside the worker container, where torch and the GPU live."""
    return f"docker exec {CONTAINER} {command}"


def _probe_cuda(host: str) -> Check:
    code, out, err = _run_ssh(host, _container_exec(GPU_COMMAND))
    if code != 0:
        reason = _first_line(err, f"exit {code}")
        return Check("cuda", "warn", _scrub(f"nvidia-smi failed on remote: {reason}", host))
    status, detail = describe_gpus(parse_gpu_output(out))
    return Check("cuda", status, _scrub(detail, host))


def _probe_python(host: str) -> Check:
    code, out, err = _run_ssh(host, _container_exec("python --version"))
    text = (out or err).strip()
    match = _VERSION_RE.search(text)
    if match is None:
        reason = _first_line(err, f"exit {code}")
        return Check("python", "warn", _scrub(f"python --version failed: {reason}", host))
    version = tuple(int(part) for part in match.groups() if part is not None)
    label = ".".join(str(part) for part in version)
    if version[:2] >= PYTHON_MIN:
        return Check("python", "ok", f"remote python3 {label}")
    floor = f"{PYTHON_MIN[0]}.{PYTHON_MIN[1]}"
    return Check("python", "fail", f"remote python3 {label} is older than {floor}")


def _gib_from_df(text: str) -> float | None:
    """First `df -h` avail cell parsed into GiB."""
    for raw in text.splitlines():
        line = raw.strip()
        match = _SIZE_RE.match(line)
        if match is None:
            continue
        factor = _SIZE_GIB.get(match.group(2).upper())
        if factor is None:
            continue
        return float(match.group(1)) * factor
    return None


def _probe_disk(host: str) -> Check:
    code, out, err = _run_ssh(host, DISK_COMMAND)
    if code != 0:
        reason = _first_line(err, f"exit {code}")
        return Check("disk", "warn", _scrub(f"df failed on remote: {reason}", host))
    gib = _gib_from_df(out)
    if gib is None:
        return Check("disk", "warn", _scrub(f"unparsed df output: {out.strip()[:80]}", host))
    detail = f"{gib:.1f} GiB available on /"
    if gib < DISK_FAIL_GIB:
        return Check("disk", "fail", f"{detail} (under {DISK_FAIL_GIB:.0f} GiB)")
    if gib < DISK_WARN_GIB:
        return Check("disk", "warn", f"{detail} (under {DISK_WARN_GIB:.0f} GiB)")
    return Check("disk", "ok", detail)


def _probe_dir(host: str) -> Check:
    directory = remote_repo_dir()
    command = f"test -d {remote_path(directory)} && echo yes || echo no"
    code, out, err = _run_ssh(host, command)
    answer = out.strip().lower()
    if code == 0 and answer.endswith("yes"):
        return Check("remote_dir", "ok", f"{directory} exists")
    if code == 0 and answer.endswith("no"):
        return Check("remote_dir", "warn", f"{directory} missing; run `msft remote sync` first")
    reason = _first_line(err, f"exit {code}")
    return Check("remote_dir", "warn", _scrub(f"cannot check {directory}: {reason}", host))


def _probes() -> list[tuple[str, str]]:
    """(check name, remote command) pairs, built per call so the dir honours the env."""
    return [
        ("cuda", _container_exec(GPU_COMMAND)),
        ("python", _container_exec("python --version")),
        ("disk", DISK_COMMAND),
        (
            "remote_dir",
            f"test -d {remote_path(remote_repo_dir())} && echo yes || echo no",
        ),
    ]


def preflight(host: str, *, execute: bool = False) -> list[Check]:
    """Check ssh, GPU, python3, disk and the remote dir; a dry run only lists the plan."""
    binary = shutil.which(SSH)
    if binary is None:
        skipped = [Check(name, "warn", "skipped: ssh unavailable") for name in PROBE_NAMES]
        return [Check("ssh", "fail", "ssh not found on PATH; install openssh-client"), *skipped]
    checks = [Check("ssh", "ok", f"{binary} (BatchMode=yes, ConnectTimeout=10)")]
    if execute:
        checks.append(_probe_cuda(host))
        checks.append(_probe_python(host))
        checks.append(_probe_disk(host))
        checks.append(_probe_dir(host))
        return checks
    for name, command in _probes():
        plan = _display(ssh_argv(HOST_PLACEHOLDER, command))
        checks.append(Check(name, "ok", f"plan: {plan}"))
    return checks


# --- rsync ----------------------------------------------------------------


def _require_remote_repo(host: str, repo: str) -> None:
    """Refuse before rsync when the box has no git checkout at `repo`."""
    command = f"test -d {remote_path(f'{repo}/.git')} && echo yes || echo no"
    code, out, _err = _run_ssh(host, command)
    if code == 0 and out.strip().lower().endswith("yes"):
        return
    raise MsftError(
        f"remote repo {repo} not found or not a git clone on the box; "
        "run `msft remote deploy --execute` first"
    )


def sync(
    host: str,
    local: str | Path,
    *,
    base_dir: str | Path = ".",
    delete: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Send a directory under `<base_dir>/data/` to `<remote repo>/data/…` over rsync."""
    source, relative = resolve_sync_source(local, base_dir)
    repo = remote_repo_dir()
    dest = remote_dest(repo, relative)
    real_argv = build_sync_argv(host, source, dest, delete=delete)
    preview = build_sync_argv(HOST_PLACEHOLDER, source, dest, delete=delete)
    result: dict[str, Any] = {
        "host": mask_host(host),
        "local": str(local),
        "dest": dest,
        "delete": bool(delete),
        "repo_checked": False,
        "excludes": list(ALWAYS_EXCLUDES),
        "argv": preview,
        "command": redact(_display(preview)),
        "executed": False,
        "ok": True,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not execute:
        return result
    missing = [name for name in (SSH, "rsync") if shutil.which(name) is None]
    if missing:
        result.update(ok=False, error="; ".join(f"{name} not found on PATH" for name in missing))
        return result
    _require_remote_repo(host, repo)
    result["repo_checked"] = True
    try:
        proc = subprocess.run(
            real_argv,
            capture_output=True,
            text=True,
            timeout=RSYNC_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result.update(executed=True, ok=False, error=f"rsync timed out after {RSYNC_TIMEOUT_S}s")
        return result
    except OSError as exc:
        result.update(ok=False, error=_scrub(str(exc), host))
        return result
    result.update(
        executed=True,
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=_scrub(_tail(proc.stdout), host),
        stderr=_scrub(_tail(proc.stderr), host),
    )
    if proc.returncode != 0:
        result["error"] = f"rsync exited {proc.returncode}"
    return result


# --- remote run -----------------------------------------------------------


def _container_running(host: str) -> bool:
    """True when `docker inspect` reports the worker container as running."""
    command = f"docker inspect -f '{{{{.State.Running}}}}' {CONTAINER}"
    code, out, _err = _run_ssh(host, command)
    return code == 0 and out.strip().lower() == "true"


def run_remote(
    cfg: RunConfig,
    host: str,
    *,
    kind: str = "train",
    extra_args: Sequence[str] = (),
    execute: bool = False,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Plan, and with `execute=True` stream, one pipeline stage inside the worker container.

    The wrapper prints `[msft-rc] N` after the container command, so the remote verdict is
    distinguishable from ssh's own exit code: a sentinel always wins (even 255), and a
    missing sentinel is a transport failure, never a green run.
    """
    config_rel = remote_config(cfg)
    command = container_command(kind, config_rel, extra_args)
    if timeout_s is not None:
        command = f"timeout --signal=TERM --kill-after=30 {timeout_s} {command}"
    wrapped = verdict_command(command)
    argv = ssh_run_argv(HOST_PLACEHOLDER, wrapped)
    result: dict[str, Any] = {
        "host": mask_host(host),
        "dir": remote_repo_dir(),
        "config": config_rel,
        "kind": kind,
        "timeout": "none" if timeout_s is None else timeout_s,
        "argv": argv,
        "command": redact(_display(argv)),
        "executed": False,
        "ok": True,
        "failure": None,
        "returncode": None,
        "ssh_returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not execute:
        return result
    if shutil.which(SSH) is None:
        result.update(
            ok=False,
            failure="transport",
            error="ssh not found on PATH; install openssh-client",
        )
        return result
    if not _container_running(host):
        raise MsftError(
            "the worker container is not running on the box; run `msft remote deploy --execute` "
            "or `msft remote up --execute`"
        )
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    try:
        proc = subprocess.Popen(
            ssh_run_argv(host, wrapped),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        result.update(ok=False, failure="transport", error=_scrub(str(exc), host))
        return result
    stream = proc.stdout
    if stream is not None:
        for line in stream:
            tail.append(line)
            print(_scrub(line.rstrip("\n"), host), file=sys.stderr)
    ssh_rc = proc.wait()
    text = "".join(tail)
    codes = SENTINEL_RE.findall(text)
    if codes:
        remote_rc: int | None = int(codes[-1])
        failure = None if remote_rc == 0 else "remote"
        error = None if remote_rc == 0 else f"remote exited {remote_rc}"
    else:
        remote_rc = None
        failure = "transport"
        error = f"ssh exited {ssh_rc} without a [msft-rc] verdict"
    result.update(
        executed=True,
        ok=failure is None,
        failure=failure,
        returncode=remote_rc,
        ssh_returncode=ssh_rc,
        stderr=_scrub(text.strip(), host),
        error=error,
    )
    return result


# --- tmux sessions --------------------------------------------------------


def slug(name: str) -> str:
    """Replace every character tmux would mangle; dots are illegal in session names."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def tmux_session_name(name: str, *, now: datetime | None = None) -> str:
    """`msft-<slug>-<UTC stamp>`; deterministic when `now` is supplied."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"msft-{slug(name)[:63] or 'run'}-{stamp}"


def validate_session(name: str) -> str:
    """Refuse anything `logs`/`stop` could use to touch a foreign tmux session."""
    if not isinstance(name, str) or TMUX_SESSION_RE.fullmatch(name) is None:
        raise MsftError(
            f"invalid tmux session {name!r}: use `msft-` followed by 1-80 letters, digits, "
            "'_' or '-'"
        )
    return name


def job_id(kind: str, *, now: datetime | None = None) -> str:
    """A queue id generated locally, so the caller knows it without parsing remote output."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{stamp}-{secrets.token_hex(2)}"


def _tmux_paths(session: str) -> tuple[str, str, str, str]:
    """(log, rc, metadata, stopped marker) relative to the repo root."""
    base = f"{TMUX_STATE_DIR}/{session}"
    return f"{base}.log", f"{base}.rc", f"{base}.json", f"{base}.stopped"


def _running_jobs(host: str, repo: str) -> list[str]:
    """Names of the queue's running jobs, used to refuse a second GPU job."""
    directory = f"{repo.rstrip('/')}/.worker/running"
    command = f"ls -1 {remote_path(directory)}/*.json 2>/dev/null || true"
    _code, out, _err = _run_ssh(host, command)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _tmux_script(session: str, command: str, timeout_s: int | None) -> str:
    """The bash script a session runs, written to a file so it is quoted exactly once.

    Building this inline means quoting a command that already contains quotes, and the second
    round turns `exec` into literal text. A file has no such problem: the single quotes below are
    just characters in it.
    """
    log, rc, _meta, _marker = _tmux_paths(session)
    pidfile = f"{TMUX_STATE_DIR}/{session}.pid"
    inner = command.split(f" {CONTAINER} ", 1)[-1]  # `python -m msft <kind> …`
    pid_prefix = f"echo $$ > {pidfile}; exec "
    container = f"docker exec -i {CONTAINER} sh -c '{pid_prefix}{inner}'"
    if timeout_s is not None:
        container = f"timeout --signal=TERM --kill-after=30 {timeout_s} {container}"
    return (
        "#!/bin/bash\n"
        "set -o pipefail\n"
        f"{container} < /dev/null 2>&1 | tee -a {log}\n"
        "rc=${PIPESTATUS[0]}\n"
        f'echo "$rc" > {rc}.tmp && mv {rc}.tmp {rc}\n'
        f'printf \'\\n[msft-rc] %s\\n\' "$rc" >> {log}\n'
    )


def _tmux_launch_command(session: str, repo: str, record: dict[str, Any]) -> str:
    """One shell command: check tmux, record the session, upload its script, start it detached."""
    _log, _rc, meta, _marker = _tmux_paths(session)
    command = str(record["command"])
    script = _tmux_script(session, command, record.get("timeout_s"))
    encoded = base64.b64encode(script.encode()).decode()
    script_path = f"{TMUX_STATE_DIR}/{session}.sh"
    payload = shlex.quote(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return (
        f"command -v tmux >/dev/null || {{ echo '{TMUX_MISSING}' >&2; exit 127; }}; "
        f"cd {remote_path(repo)} || exit 1; "
        f"if tmux has-session -t ={shlex.quote(session)} 2>/dev/null; then "
        f"echo 'MSFT_ERR=session-exists' >&2; exit 4; fi; "
        f"mkdir -p {shlex.quote(TMUX_STATE_DIR)}; "
        f"printf '%s\n' {payload} > {shlex.quote(meta)}; "
        f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(script_path)}; "
        f"chmod +x {shlex.quote(script_path)}; "
        f"tmux new-session -d -s {shlex.quote(session)} -c {remote_path(repo)} "
        f"{shlex.quote('bash ' + script_path)}"
    )


def launch_tmux(
    cfg: RunConfig,
    host: str,
    *,
    kind: str = "train",
    extra_args: Sequence[str] = (),
    timeout_s: int | None = None,
    session: str | None = None,
    force: bool = False,
    execute: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Start a detached tmux session that runs the stage through the container's tty.

    `docker exec -it` inside tmux means `send-keys C-c` reaches the training process, which
    is what `remote stop` relies on.
    """
    name = validate_session(session or tmux_session_name(cfg.name, now=now))
    repo = remote_repo_dir()
    config_rel = remote_config(cfg)
    log, rc, _meta, _marker = _tmux_paths(name)
    container = container_command(kind, config_rel, extra_args, tty=True)
    if timeout_s is not None:
        container = f"timeout --signal=TERM --kill-after=30 {timeout_s} {container}"
    record = {
        "session": name,
        "config": config_rel,
        "kind": kind,
        "started": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "timeout_s": timeout_s,
        "extra_args": list(extra_args),
        "command": container,
    }
    command = _tmux_launch_command(name, repo, record)
    argv = ssh_run_argv(HOST_PLACEHOLDER, command)
    hints = [
        f"msft remote logs --tmux {name} --follow",
        f"msft remote logs --tmux {name} --wait",
        f"msft remote stop --tmux {name}",
        f"tmux attach -t ={name}",
    ]
    result: dict[str, Any] = {
        "host": mask_host(host),
        "session": name,
        "config": config_rel,
        "kind": kind,
        "state": "plan",
        "log": log,
        "rc_file": rc,
        "timeout": "none" if timeout_s is None else timeout_s,
        "argv": argv,
        "command": redact(_display(argv)),
        "hints": hints,
        "executed": False,
        "ok": True,
        "returncode": None,
    }
    if not execute:
        return result
    if shutil.which(SSH) is None:
        result.update(
            ok=False,
            state="failed",
            error="ssh not found on PATH; install openssh-client",
        )
        return result
    if not force and _running_jobs(host, repo):
        raise MsftError(
            "the worker is running a job; two GPU jobs OOM a 16 GB card. "
            "Use --queue or pass --force"
        )
    code, out, err = _run_ssh_run(host, command)
    combined = f"{out}\n{err}"
    if code == 127 and TMUX_MISSING in combined:
        raise MsftError(
            "tmux is not installed on the box (apt install tmux). Use --queue for a "
            "disconnect-proof run without tmux, or drop --tmux for a foreground run"
        )
    if code != 0:
        reason = _first_line(err or out, f"exit {code}")
        result.update(
            ok=False,
            state="failed",
            returncode=code,
            error=_scrub(f"tmux launch failed: {reason}", host),
        )
        return result
    result.update(
        executed=True,
        state="started",
        ok=True,
        returncode=0,
        started=record["started"],
    )
    return result


def _parse_tmux_probe(output: str) -> tuple[str, int | None, str]:
    """Split `<state line> … __MSFT_TAIL__ <log>` into (state, remote rc, tail)."""
    state = "lost"
    returncode: int | None = None
    tail: list[str] = []
    in_tail = False
    for line in output.splitlines():
        if not in_tail and line.startswith(TMUX_STATE_MARK):
            value = line[len(TMUX_STATE_MARK) :].strip()
            parts = value.split()
            state = parts[0] if parts else "lost"
            if state == "exited" and len(parts) > 1:
                try:
                    returncode = int(parts[1])
                except ValueError:
                    returncode = None
        elif line.strip() == TMUX_TAIL_MARK:
            in_tail = True
        elif in_tail:
            tail.append(line)
    return state, returncode, "\n".join(tail)


def _tmux_result(
    host: str, session: str, state: str, returncode: int | None, tail: str, command: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": mask_host(host),
        "mode": "tmux",
        "session": session,
        "state": state,
        "returncode": returncode,
        "tail": tail,
        "executed": True,
        "ok": state != "lost",
        "command": redact(_display(ssh_run_argv(HOST_PLACEHOLDER, command))),
    }
    if state == "lost":
        result["error"] = _scrub(
            f"{session} has no rc file, no live session and no stop marker", host
        )
    return result


def _probe_tmux(host: str, session: str, lines: int) -> dict[str, Any]:
    """One short ssh: state of the session plus the requested tail."""
    repo = remote_repo_dir()
    log, rc, _meta, marker = _tmux_paths(session)
    command = (
        f"cd {remote_path(repo)} || exit 1; "
        f"if [ -f {shlex.quote(rc)} ]; then "
        f"echo '{TMUX_STATE_MARK} exited '$(cat {shlex.quote(rc)}); "
        f"elif tmux has-session -t ={shlex.quote(session)} 2>/dev/null; then "
        f"echo '{TMUX_STATE_MARK} running'; "
        f"elif [ -f {shlex.quote(marker)} ]; then "
        f"echo '{TMUX_STATE_MARK} stopped'; "
        f"else echo '{TMUX_STATE_MARK} lost'; fi; "
        f"echo {TMUX_TAIL_MARK}; "
        f"tail -n {int(lines)} {shlex.quote(log)} 2>/dev/null"
    )
    code, out, err = _run_ssh_run(host, command)
    state, returncode, tail = _parse_tmux_probe(out)
    if code != 0 and not out.strip():
        # A dead link is not a lost session: `--wait` must keep polling through it.
        result = _tmux_result(host, session, "lost", None, "", command)
        result.update(
            ok=False,
            transport=True,
            error=_scrub(f"ssh exited {code}: {_first_line(err, 'no output')}", host),
        )
        return result
    return _tmux_result(host, session, state, returncode, tail, command)


def _follow_tmux(host: str, session: str, lines: int) -> dict[str, Any]:
    """Stream `tail -F` to stderr; Ctrl-C closes the tail and never touches the job."""
    repo = remote_repo_dir()
    log, _rc, _meta, _marker = _tmux_paths(session)
    command = (
        f"cd {remote_path(repo)} || exit 1; "
        f"tail -n {int(lines)} -F {shlex.quote(log)}"
    )
    try:
        proc = subprocess.Popen(
            ssh_run_argv(host, command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise MsftError(f"ssh failed: {_scrub(str(exc), host)}") from exc
    stream = proc.stdout
    if stream is not None:
        for line in stream:
            print(_scrub(line.rstrip("\n"), host), file=sys.stderr)
    proc.wait()
    return {
        "host": mask_host(host),
        "mode": "tmux",
        "session": session,
        "state": "follow-ended",
        "tail": "",
        "executed": True,
        "ok": True,
    }


def logs_tmux(
    host: str,
    session: str,
    *,
    lines: int = 40,
    follow: bool = False,
    wait: bool = False,
    interval: float = WAIT_INTERVAL_S,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """State plus tail of a tmux session; `--wait` polls the rc file over short ssh calls."""
    validate_session(session)
    if follow:
        return _follow_tmux(host, session, lines)
    if not wait:
        return _probe_tmux(host, session, lines)
    while True:
        result = _probe_tmux(host, session, lines)
        if result.get("transport"):
            sleep(interval)
            continue
        if result["state"] in ("exited", "stopped", "lost"):
            return result
        sleep(interval)


def _validate_job_id(job: str) -> str:
    if not isinstance(job, str) or JOB_ID_RE.fullmatch(job) is None:
        raise MsftError(
            f"invalid job id {job!r}: use letters, digits, '.', '_' or '-' (at most 80)"
        )
    return job


def _probe_job(host: str, job: str, lines: int) -> dict[str, Any]:
    """Read `done/<id>.json` on the host and tail the job log through the container."""
    repo = remote_repo_dir()
    done = f".worker/done/{job}.json"
    command = (
        f"cd {remote_path(repo)} || exit 1; "
        f"if [ -f {shlex.quote(done)} ]; then "
        f"echo '{TMUX_STATE_MARK} done'; cat {shlex.quote(done)}; "
        f"else echo '{TMUX_STATE_MARK} pending'; fi; "
        f"echo {TMUX_TAIL_MARK}; "
        f"docker exec {CONTAINER} python -m msft worker log --id {shlex.quote(job)} "
        f"--lines {int(lines)} --dir {CONTAINER_REPO}/.worker 2>/dev/null"
    )
    code, out, err = _run_ssh_run(host, command)
    head, _sep, tail = out.partition(TMUX_TAIL_MARK)
    state = "pending"
    payload: dict[str, Any] = {}
    first, _nl, rest = head.partition("\n")
    if first.startswith(TMUX_STATE_MARK):
        state = first[len(TMUX_STATE_MARK) :].strip() or "pending"
        raw = rest.strip()
        if raw:
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                payload = loaded
    status = str(payload.get("status") or state)
    result: dict[str, Any] = {
        "host": mask_host(host),
        "mode": "job",
        "job": job,
        "state": status,
        "returncode": payload.get("returncode"),
        "timed_out": bool(payload.get("timed_out", False)),
        "tail": tail.strip("\n"),
        "executed": True,
        "ok": status != "failed",
        "command": redact(_display(ssh_run_argv(HOST_PLACEHOLDER, command))),
    }
    if code != 0 and not out.strip():
        result.update(
            state="pending",
            ok=False,
            transport=True,
            error=_scrub(f"ssh exited {code}: {_first_line(err, 'no output')}", host),
        )
    return result


def logs_job(
    host: str,
    job: str,
    *,
    lines: int = 40,
    wait: bool = False,
    interval: float = WAIT_INTERVAL_S,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Tail a queued job; `--wait` polls `done/<id>.json` until it appears."""
    _validate_job_id(job)
    if not wait:
        return _probe_job(host, job, lines)
    while True:
        result = _probe_job(host, job, lines)
        if result.get("transport"):
            sleep(interval)
            continue
        if result["state"] in ("done", "failed"):
            return result
        sleep(interval)


def stop_tmux(
    host: str, session: str, *, grace: int = 30, execute: bool = True
) -> dict[str, Any]:
    """Send C-c to the session's pty, then kill it after `grace` seconds if it survives."""
    validate_session(session)
    if grace < 0:
        raise MsftError(f"invalid --grace {grace}: use 0 or more seconds")
    repo = remote_repo_dir()
    _log, _rc, _meta, marker = _tmux_paths(session)
    pidfile = f"{TMUX_STATE_DIR}/{session}.pid"
    _stop_inner = f"p=$(cat {pidfile} 2>/dev/null); [ -n \"$p\" ] && kill -TERM \"$p\""
    target = shlex.quote(session)
    command = (
        f"cd {remote_path(repo)} || exit 1; "
        f"if ! tmux has-session -t ={target} 2>/dev/null; then "
        f"echo '{TMUX_NO_SESSION}' >&2; exit 3; fi; "
        # No tty in the pane, so C-c has nothing to forward: read the pid the job wrote and
        # signal that one process, then let the session finish on its own.
        f"docker exec {CONTAINER} bash -lc "
        f"{shlex.quote(_stop_inner)} 2>/dev/null; "
        f"i=0; while tmux has-session -t ={target} 2>/dev/null; do "
        f'if [ "$i" -ge {int(grace)} ]; then '
        f"tmux kill-session -t ={target} 2>/dev/null; "
        f"mkdir -p {shlex.quote(TMUX_STATE_DIR)}; : > {shlex.quote(marker)}; "
        f"echo killed; exit 0; fi; "
        f"sleep 1; i=$((i+1)); done; "
        f"mkdir -p {shlex.quote(TMUX_STATE_DIR)}; : > {shlex.quote(marker)}; "
        f"echo graceful"
    )
    result: dict[str, Any] = {
        "host": mask_host(host),
        "session": session,
        "state": "plan",
        "executed": False,
        "ok": True,
        "returncode": None,
        "command": redact(_display(ssh_argv(HOST_PLACEHOLDER, command))),
    }
    if not execute:
        return result
    code, out, err = _run_ssh_run(host, command, timeout_s=max(60, grace + 30))
    text = out.strip()
    state = "graceful" if text.endswith("graceful") else (
        "killed" if text.endswith("killed") else "failed"
    )
    if code == 3 and TMUX_NO_SESSION in err:
        result.update(state="missing", returncode=code, ok=False, error="no such session")
        return result
    result.update(
        executed=True,
        state=state,
        returncode=code,
        ok=code == 0 and state in ("graceful", "killed"),
    )
    if not result["ok"]:
        result["error"] = _scrub(_first_line(err or out, f"exit {code}"), host)
    return result
