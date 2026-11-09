"""Remote deployment orchestration: the docker-compose worker on the GPU box.

The laptop is the source of truth: code is committed and pushed here, the box pulls, and the
container serves the job queue. This module keeps that loop in one place, and keeps the host
out of the repository — it comes from a git-ignored `configs/worker.local.yaml` (copy of
`configs/worker.example.yaml`) or from MSFT_REMOTE_HOST.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msft.config import MsftError, redact
from msft.remote import mask_host, remote_path, remote_repo_dir

CONFIG_CANDIDATES = ("configs/worker.local.yaml", "worker.local.yaml")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


@dataclass
class WorkerSpec:
    """Where the worker lives. Host is never written to the repo."""

    host: str
    repo_dir: str = "WORKSPACE/machinum-sft"
    compose_file: str = "deploy/remote/docker-compose.yml"
    container: str = "msft-worker"

    @property
    def repo(self) -> str:
        return self.repo_dir if self.repo_dir.startswith(("/", "~")) else f"~/{self.repo_dir}"


def load_spec(path: str | Path | None = None) -> WorkerSpec:
    """Read the git-ignored worker config, falling back to MSFT_REMOTE_HOST."""
    if path is None:
        for candidate in CONFIG_CANDIDATES:
            if Path(candidate).exists():
                path = candidate
                break
    data: dict[str, Any] = {}
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
        try:
            import yaml

            data = yaml.safe_load(raw) or {}
        except Exception as exc:  # pragma: no cover - malformed file
            raise MsftError(f"cannot parse {path}: {exc}") from exc
    host = str(data.get("host") or os.environ.get("MSFT_REMOTE_HOST") or "")
    if not host:
        raise MsftError(
            "no remote host: copy configs/worker.example.yaml to configs/worker.local.yaml "
            "and set host, or export MSFT_REMOTE_HOST"
        )
    return WorkerSpec(
        host=host,
        repo_dir=str(data.get("repo_dir") or remote_repo_dir()),
        compose_file=str(data.get("compose_file") or WorkerSpec.compose_file),
        container=str(data.get("container") or WorkerSpec.container),
    )


def _ssh(spec: WorkerSpec, command: str, *, execute: bool) -> dict[str, Any]:
    argv = ["ssh", *SSH_OPTS, spec.host, command]
    plan = {"host": mask_host(spec.host), "command": redact(command), "argv": "ssh ... (hidden)"}
    if not execute:
        return {**plan, "executed": False}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MsftError(f"ssh failed: {exc}") from exc
    result = {
        **plan,
        "executed": True,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": redact(proc.stdout)[-4000:],
        "stderr": redact(proc.stderr)[-2000:],
    }
    return result


def pull(spec: WorkerSpec, *, execute: bool = False) -> dict[str, Any]:
    """Fast-forward the box to the pushed branch (single source of truth)."""
    safe = '-c safe.directory="$PWD"'
    return _ssh(
        spec,
        f"cd {remote_path(spec.repo)} && git {safe} fetch --all --prune"
        f" && git {safe} pull --ff-only",
        execute=execute,
    )


def up(spec: WorkerSpec, *, execute: bool = False, build: bool = True) -> dict[str, Any]:
    """Bring the compose worker up (builds the image the first time)."""
    flag = "--build" if build else "--no-build"
    return _ssh(
        spec,
        f"cd {remote_path(spec.repo)} && docker compose -f {shlex.quote(spec.compose_file)}"
        f" up -d {flag} worker",
        execute=execute,
    )


def restart(spec: WorkerSpec, *, execute: bool = False) -> dict[str, Any]:
    """Restart the worker container in place (after a pull that changed worker code)."""
    return _ssh(
        spec,
        f"cd {remote_path(spec.repo)} && docker compose -f {shlex.quote(spec.compose_file)}"
        " restart worker",
        execute=execute,
    )


def down(spec: WorkerSpec, *, execute: bool = False) -> dict[str, Any]:
    """Stop and remove the worker container (the image and the mount stay)."""
    return _ssh(
        spec,
        f"cd {remote_path(spec.repo)} && docker compose -f {shlex.quote(spec.compose_file)}"
        " down",
        execute=execute,
    )


def signal(
    spec: WorkerSpec,
    kind: str,
    *,
    config: str | None = None,
    job_id: str | None = None,
    extra_args: list[str] | None = None,
    timeout_s: int | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Queue a job by writing the request file through the container (no python on the host)."""
    args = [shlex.quote(kind)]
    if config:
        args.append(f"--config {shlex.quote(config)}")
    if job_id:
        args.append(f"--id {shlex.quote(job_id)}")
    for extra in extra_args or []:
        # `--arg=--inspect-mask`, never `--arg --inspect-mask`: argparse reads an
        # option-looking token as a flag, not as the value.
        args.append(f"--arg={shlex.quote(extra)}")
    if timeout_s is not None:
        args.append(f"--timeout {int(timeout_s)}")
    inner = (
        "docker exec "
        f"{shlex.quote(spec.container)} python -m msft worker signal {' '.join(args)} "
        "--dir /workspace/.worker"
    )
    return _ssh(spec, f"cd {remote_path(spec.repo)} && {inner}", execute=execute)


def status(spec: WorkerSpec, *, execute: bool = False) -> dict[str, Any]:
    """Container health plus the queue summary from inside the worker."""
    compose = (
        f"cd {remote_path(spec.repo)} && docker compose -f {shlex.quote(spec.compose_file)}"
        " ps --format json"
    )
    queue = (
        f"docker exec {shlex.quote(spec.container)} python -m msft worker status"
        " --dir /workspace/.worker --json"
    )
    return _ssh(spec, f"{compose}; echo '---'; {queue}", execute=execute)


def spec_json(spec: WorkerSpec) -> dict[str, Any]:
    """The spec with the host masked, safe to print."""
    return {
        "host": mask_host(spec.host),
        "repo": spec.repo,
        "compose_file": spec.compose_file,
        "container": spec.container,
    }


def dump_spec(spec: WorkerSpec) -> str:
    return json.dumps(spec_json(spec), indent=2)
