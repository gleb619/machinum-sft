"""Remote GPU orchestration over ssh and rsync.

Every entry point is a plan until `execute=True`: the returned mapping (or the `Check` details)
describes exactly what would run, and the host is never echoed verbatim — plans use the
`$MSFT_REMOTE_HOST` placeholder and human strings go through `_scrub`, so printing a plan cannot
leak the box. A missing `ssh`/`rsync` binary is reported, never raised.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
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
TRANSPORT = " ".join(
    (SSH, *SSH_OPTS, "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4")
)
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
    "build_sync_argv",
    "container_command",
    "mask_host",
    "preflight",
    "remote_config",
    "remote_dest",
    "remote_repo_dir",
    "resolve_sync_source",
    "run_remote",
    "ssh_argv",
    "sync",
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
) -> str:
    """`docker exec msft-worker python -m msft <kind> --config … --execute`, shell-quoted.

    The builder stays wrapper-free: `run_remote` applies any `timeout` around the result.
    """
    parts = ["docker", "exec", CONTAINER, "python", "-m", "msft", kind]
    if config_rel:
        parts += ["--config", config_rel]
    parts += list(extra_args)
    if "--execute" not in parts:
        parts.append("--execute")
    return shlex.join(parts)


def ssh_argv(host: str, command: str) -> list[str]:
    """Build an ssh argv that never prompts for a password and gives up in 10s."""
    return [SSH, *SSH_OPTS, host, command]


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
    cfg: RunConfig, host: str, *, execute: bool = False, timeout_s: int | None = None
) -> dict[str, Any]:
    """Plan, and with `execute=True` run, `msft train` inside the worker container."""
    config_rel = remote_config(cfg)
    command = container_command("train", config_rel)
    if timeout_s is not None:
        command = f"timeout --signal=TERM --kill-after=30 {timeout_s} {command}"
    result: dict[str, Any] = {
        "host": mask_host(host),
        "dir": remote_repo_dir(),
        "config": config_rel,
        "timeout": "none" if timeout_s is None else timeout_s,
        "argv": ssh_argv(HOST_PLACEHOLDER, command),
        "command": redact(_display(ssh_argv(HOST_PLACEHOLDER, command))),
        "executed": False,
        "ok": True,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if not execute:
        return result
    if shutil.which(SSH) is None:
        result.update(ok=False, error="ssh not found on PATH; install openssh-client")
        return result
    if not _container_running(host):
        raise MsftError(
            "the worker container is not running on the box; run `msft remote deploy --execute` "
            "or `msft remote up --execute`"
        )
    try:
        proc = subprocess.run(ssh_argv(host, command), capture_output=True, text=True, check=False)
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
        result["error"] = f"ssh exited {proc.returncode}"
    return result
