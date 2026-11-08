"""Worker mode: a job queue that lives on the remote GPU box.

The laptop (or CI, or a human) *signals* a job by dropping a JSON file into
``<dir>/inbox``; the worker *serves* that directory, executing one job at a time and
writing a log plus a result file for each. Nothing about the worker is model-specific, so
the same loop also drives `merge`, `quantiate`-style stages and `eval`.

Layout of the worker directory::

    <dir>/inbox/<job-id>.json     job requests (written atomically by `signal`)
    <dir>/running/<job-id>.json   the job currently executing (moved from inbox)
    <dir>/done/<job-id>.json      finished jobs, with rc, duration and artifacts
    <dir>/logs/<job-id>.log       combined stdout/stderr of the child process
    <dir>/heartbeat.json          {"pid", "updated", "job", "state"} — polled by healthchecks

Design rules: stdlib only, no network, no git. `git pull` is the deploy step, done by the
caller (or by the container's entrypoint), never by the worker itself.
"""

from __future__ import annotations

import json
import os
import shutil
import signal as _sig
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from msft.config import MsftError, redact

JOB_KINDS = ("train", "merge", "quantize", "eval", "data", "doctor")
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def worker_dir(path: str | Path) -> Path:
    """Create the worker directory layout, returning its root."""
    root = Path(path).expanduser()
    for child in ("inbox", "running", "done", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Job:
    """One unit of work for the remote box."""

    id: str
    kind: str
    config: str | None = None
    extra_args: list[str] = field(default_factory=list)
    created: str = field(default_factory=_now)
    status: str = PENDING
    started: str | None = None
    finished: str | None = None
    returncode: int | None = None
    duration_s: float | None = None
    log: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def signal(
    directory: str | Path,
    kind: str,
    *,
    job_id: str | None = None,
    config: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Drop a job request into the inbox; returns its id.

    Written to a temporary file and renamed, so the worker never reads a half-written job.
    """
    if kind not in JOB_KINDS:
        raise MsftError(f"unknown job kind '{kind}'; pick one of {', '.join(JOB_KINDS)}")
    root = worker_dir(directory)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ident = job_id or f"{kind}-{stamp}"
    job = Job(id=ident, kind=kind, config=config, extra_args=list(extra_args or []))
    target = root / "inbox" / f"{ident}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return ident


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MsftError(f"cannot read job file {path.name}: {exc}") from exc


def _next_job(root: Path) -> Path | None:
    """Oldest pending job, by creation stamp in the name then mtime."""
    pending = sorted((root / "inbox").glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    return pending[0] if pending else None


def _heartbeat(root: Path, state: str, job: str | None = None) -> None:
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "updated": _now(),
        "state": state,
        "job": job,
        "python": sys.version.split()[0],
    }
    tmp = root / "heartbeat.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, root / "heartbeat.json")


def build_command(job: Job, argv0: list[str]) -> list[str]:
    """`python -m msft <kind> [--config ...] --execute` plus any extra args.

    `data` needs its `build` action; the other kinds are leaves of the CLI.
    """
    cmd = [*argv0, "-m", "msft", *([job.kind, "build"] if job.kind == "data" else [job.kind])]
    if job.config:
        cmd += ["--config", job.config]
    cmd += list(job.extra_args)
    if "--execute" not in cmd:
        cmd.append("--execute")
    return cmd


def run_job(
    root: Path, job_file: Path, *, argv0: list[str] | None = None, cwd: str | None = None
) -> Job:
    """Execute one job file, moving it through running → done, logging everything."""
    data = _read(job_file)
    job = Job(**{key: data[key] for key in Job.__dataclass_fields__ if key in data})
    running = root / "running" / job_file.name
    os.replace(job_file, running)
    job.status = RUNNING
    job.started = _now()
    log_path = root / "logs" / f"{job.id}.log"
    cmd = build_command(job, argv0 or [sys.executable])
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {_now()} {job.kind} {job.config or ''}\n# cmd: {redact(' '.join(cmd))}\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            job.returncode = proc.returncode
        except OSError as exc:  # missing interpreter, permissions
            log.write(f"error: {exc}\n")
            job.returncode = 127
            job.error = str(exc)
    job.duration_s = round(time.monotonic() - started, 1)
    job.finished = _now()
    job.log = str(log_path)
    job.status = DONE if job.returncode == 0 else FAILED
    target = root / "done" / job_file.name
    target.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
    running.unlink(missing_ok=True)
    return job


def gpu_note() -> str:
    """A one-line, console-visible statement about where jobs will actually run.

    Silence is the bug here: a worker that quietly falls back to the CPU looks healthy while
    a 3-minute job runs for days. Every serve start prints this line, and `status` carries it.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return (
            "WARNING: CUDA_VISIBLE_DEVICES is empty — jobs would run on the CPU "
            "(refused unless MSFT_ALLOW_CPU=1)"
        )
    if shutil.which("nvidia-smi") is None:
        return "WARNING: nvidia-smi not found — no GPU here; training jobs will refuse to start"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"WARNING: cannot query the GPU ({exc}); training jobs will refuse to start"
    gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not gpus:
        return "WARNING: nvidia-smi reported no GPU; training jobs will refuse to start"
    return "GPU: " + " | ".join(gpus)


def _install_stop_handler() -> Any:
    """Turn SIGTERM/SIGINT into a clean loop exit so a job is not left 'running'."""
    stop = {"flag": False}

    def handler(_signum: int, _frame: Any) -> None:
        stop["flag"] = True

    _sig.signal(_sig.SIGTERM, handler)
    _sig.signal(_sig.SIGINT, handler)
    return stop


def serve(
    directory: str | Path,
    *,
    interval: float = 15.0,
    once: bool = False,
    argv0: list[str] | None = None,
    cwd: str | None = None,
) -> int:
    """Poll the inbox forever (or until `once`), running jobs one at a time."""
    root = worker_dir(directory)
    stop = _install_stop_handler()
    _heartbeat(root, "idle")
    print(f"worker: serving {root} (interval {interval:g}s)", flush=True)
    print(f"worker: {gpu_note()}", flush=True)
    while not stop["flag"]:
        job_file = _next_job(root)
        if job_file is None:
            if once:
                break
            _heartbeat(root, "idle")
            time.sleep(interval)
            continue
        job = run_job(root, job_file, argv0=argv0, cwd=cwd)
        state = "idle" if job.status == DONE else job.status
        _heartbeat(root, state, job.id)
        print(
            f"worker: {job.id} -> {job.status} rc={job.returncode} in {job.duration_s:g}s"
            f" (log {job.log})",
            flush=True,
        )
        if once:
            break
    _heartbeat(root, "stopped")
    return 0


def status(directory: str | Path) -> dict[str, Any]:
    """Summarise the queue: heartbeat, pending, running and recent results."""
    root = Path(directory).expanduser()
    heartbeat: dict[str, Any] | None = None
    if (root / "heartbeat.json").exists():
        try:
            heartbeat = _read(root / "heartbeat.json")
        except MsftError:
            heartbeat = None
    if heartbeat and heartbeat.get("updated"):
        try:
            stamp = datetime.fromisoformat(str(heartbeat["updated"]))
            heartbeat["age_s"] = round((datetime.now(UTC) - stamp).total_seconds(), 1)
        except ValueError:
            heartbeat["age_s"] = None

    def jobs(folder: str) -> list[dict[str, Any]]:
        out = []
        for path in sorted((root / folder).glob("*.json")) if (root / folder).exists() else []:
            try:
                data = _read(path)
            except MsftError as exc:
                data = {"id": path.stem, "error": str(exc)}
            out.append(data)
        return out

    pending = jobs("inbox")
    running = jobs("running")
    done = jobs("done")
    return {
        "dir": str(root),
        "heartbeat": heartbeat,
        "pending": pending,
        "running": running,
        "done": done[-10:],
        "gpu": gpu_note(),
        "counts": {
            "pending": len(pending),
            "running": len(running),
            "done": sum(1 for job in done if job.get("status") == DONE),
            "failed": sum(1 for job in done if job.get("status") == FAILED),
        },
    }


def tail_log(directory: str | Path, job_id: str | None = None, lines: int = 40) -> str:
    """Return the tail of a job log (latest job when `job_id` is omitted)."""
    root = Path(directory).expanduser()
    logs = sorted((root / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return ""
    path = root / "logs" / f"{job_id}.log" if job_id else logs[-1]
    if not path.exists():
        raise MsftError(f"no log for job '{job_id}'")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:])


def reset(directory: str | Path, *, keep_logs: bool = True) -> int:
    """Clear inbox/running/done (used between experiments). Logs are kept by default."""
    root = Path(directory).expanduser()
    removed = 0
    for folder in ("inbox", "running", "done"):
        path = root / folder
        if not path.exists():
            continue
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
                removed += 1
            elif item.is_dir():
                shutil.rmtree(item)
                removed += 1
    if not keep_logs and (root / "logs").exists():
        for item in (root / "logs").iterdir():
            item.unlink()
            removed += 1
    return removed
