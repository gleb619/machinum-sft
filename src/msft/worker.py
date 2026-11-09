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

import contextlib
import json
import os
import re
import secrets
import shutil
import signal as _sig
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from msft.config import MsftError, redact

JOB_KINDS = ("train", "merge", "quantize", "eval", "data", "doctor", "fetch", "verify")
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
HEARTBEAT_INTERVAL_S = 30.0
STOP_POLL_S = 1.0
STOP_GRACE_S = 30.0
MAX_TIMEOUT_S = 2_592_000
INTERRUPTED_ERROR = "interrupted (worker restarted)"

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_DURATION_RE = re.compile(r"^([0-9]+)([smhd]?)$")
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _validate_id(ident: str) -> str:
    """Reject ids that cannot safely become a filename inside the queue directory."""
    if not isinstance(ident, str) or not _ID_RE.fullmatch(ident):
        raise MsftError(
            f"invalid job id {ident!r}: use letters, digits, '.', '_' or '-' "
            "(at most 80 characters, starting with a letter or digit)"
        )
    return ident


def parse_duration(text: str) -> int:
    """Parse `90`, `90s`, `45m`, `2h` or `1d` into seconds; reject signs and fractions."""
    match = _DURATION_RE.fullmatch(text.strip()) if isinstance(text, str) else None
    if match is not None:
        seconds = int(match.group(1)) * _DURATION_UNITS[match.group(2)]
        if 1 <= seconds <= MAX_TIMEOUT_S:
            return seconds
    raise MsftError(f"invalid --timeout {text!r}: use e.g. 900, 15m, 2h")


def _check_timeout(value: Any) -> int | None:
    """Accept None or a whole number of seconds in range; reject everything else."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MsftError(
            f"invalid timeout_s {value!r}: use a positive whole number of seconds"
        )
    if not 1 <= value <= MAX_TIMEOUT_S:
        raise MsftError(
            f"invalid timeout_s {value!r}: use a positive whole number of seconds"
        )
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON record atomically, so a reader never sees half a file."""
    # The heartbeat thread and the job loop can write the same path; a shared temp name would
    # let one rename the file the other is still writing.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


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
    pid: int | None = None
    timeout_s: int | None = None
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def signal(
    directory: str | Path,
    kind: str,
    *,
    job_id: str | None = None,
    config: str | None = None,
    extra_args: list[str] | None = None,
    timeout_s: int | None = None,
) -> str:
    """Drop a job request into the inbox; returns its id.

    Written to a temporary file and renamed, so the worker never reads a half-written job.
    """
    if kind not in JOB_KINDS:
        raise MsftError(f"unknown job kind '{kind}'; pick one of {', '.join(JOB_KINDS)}")
    timeout_s = _check_timeout(timeout_s)
    if job_id is not None:
        _validate_id(job_id)
    root = worker_dir(directory)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ident = job_id or f"{kind}-{stamp}-{secrets.token_hex(2)}"
    job = Job(
        id=ident,
        kind=kind,
        config=config,
        extra_args=list(extra_args or []),
        timeout_s=timeout_s,
    )
    target = root / "inbox" / f"{ident}.json"
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
    try:
        os.link(tmp, target)
    except FileExistsError as exc:
        raise MsftError(
            f"job '{ident}' already exists in {root / 'inbox'}; pass a different --id"
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)
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


def _heartbeat(root: Path, state: str, job: str | None = None, pid: int | None = None) -> None:
    payload = {
        "pid": os.getpid(),
        "child_pid": pid,
        "host": socket.gethostname(),
        "updated": _now(),
        "state": state,
        "job": job,
        "python": sys.version.split()[0],
    }
    _write_json(root / "heartbeat.json", payload)


class _Heartbeat(threading.Thread):
    """Refresh the heartbeat while a job runs, so `status` and healthchecks stay honest."""

    def __init__(
        self, root: Path, job_id: str, interval: float = HEARTBEAT_INTERVAL_S
    ) -> None:
        super().__init__(name=f"msft-heartbeat-{job_id}", daemon=True)
        self.root = root
        self.job_id = job_id
        self.interval = interval
        self.pid: int | None = None
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            _heartbeat(self.root, RUNNING, self.job_id, pid=self.pid)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=max(1.0, self.interval))


def build_command(job: Job, argv0: list[str]) -> list[str]:
    """`python -m msft <kind> [--config ...] --execute` plus any extra args.

    `data` needs its `build` action; `fetch` and `verify` are leaves of the CLI like the
    training stages, so the generic mapping covers them.
    """
    cmd = [*argv0, "-m", "msft", *([job.kind, "build"] if job.kind == "data" else [job.kind])]
    if job.config:
        cmd += ["--config", job.config]
    cmd += list(job.extra_args)
    if "--execute" not in cmd:
        cmd.append("--execute")
    return cmd


def _signal_group(proc: subprocess.Popen[Any], sig: int) -> bool:
    """Signal the child's whole process group; False when `killpg` does not exist."""
    if not hasattr(os, "killpg"):
        return False
    # The group is already gone, or we may not signal it; either way there is nothing
    # further to escalate here.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)
    return True


def _terminate_group(proc: subprocess.Popen[Any]) -> tuple[int, bool]:
    """SIGTERM the process group, SIGKILL it after the grace period; return (rc, escalated)."""
    if not _signal_group(proc, _sig.SIGTERM):
        proc.terminate()
    escalated = False
    try:
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        escalated = True
        if not _signal_group(proc, _sig.SIGKILL):
            proc.kill()
        proc.wait()
    return (proc.returncode if proc.returncode is not None else -1), escalated


def _wait_for_child(
    proc: subprocess.Popen[Any],
    stop: threading.Event | None,
    deadline: float | None = None,
) -> tuple[int, str | None]:
    """Wait for a child; a stop request or the deadline terminates its process group.

    Returns the child's real return code and the reason it was killed: None, "stop" or
    "timeout". A stop wins over a timeout when both are set: a worker shutdown must never
    be reported as a job that ran too long.
    """
    while True:
        try:
            return proc.wait(timeout=STOP_POLL_S), None
        except subprocess.TimeoutExpired:
            reason: str | None = None
            if stop is not None and stop.is_set():
                reason = "stop"
            elif deadline is not None and time.monotonic() >= deadline:
                reason = "timeout"
            if reason is None:
                continue
            returncode, _escalated = _terminate_group(proc)
            return returncode, reason


def _record_failure(
    root: Path, job_file: Path, ident: str, kind: str, error: str
) -> Job:
    """Move a job that could not run to done/ with status=failed and the reason."""
    job = Job(id=ident, kind=kind)
    job.status = FAILED
    job.error = redact(error)
    job.finished = _now()
    _write_json(root / "done" / job_file.name, job.to_dict())
    for folder in ("inbox", "running"):
        (root / folder / job_file.name).unlink(missing_ok=True)
    return job


def _job_details(path: Path) -> tuple[str, str]:
    """Best-effort (id, kind) from a job file, falling back to its filename."""
    try:
        data = _read(path)
    except MsftError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return str(data.get("id") or path.stem), str(data.get("kind") or "unknown")


def run_job(
    root: Path,
    job_file: Path,
    *,
    argv0: list[str] | None = None,
    cwd: str | None = None,
    stop: threading.Event | None = None,
) -> Job:
    """Execute one job file, moving it through running → done, logging everything."""
    data = _read(job_file)
    job = Job(**{key: data[key] for key in Job.__dataclass_fields__ if key in data})
    _validate_id(job.id)
    if job.kind not in JOB_KINDS:
        raise MsftError(
            f"job '{job.id}' has unknown kind '{job.kind}'; "
            f"pick one of {', '.join(JOB_KINDS)}"
        )
    job.timeout_s = _check_timeout(job.timeout_s)
    running = root / "running" / job_file.name
    os.replace(job_file, running)
    job.status = RUNNING
    job.started = _now()
    _write_json(running, job.to_dict())
    log_path = root / "logs" / f"{job.id}.log"
    cmd = build_command(job, argv0 or [sys.executable])
    started = time.monotonic()
    heartbeat = _Heartbeat(root, job.id)
    heartbeat.start()
    _heartbeat(root, RUNNING, job.id)
    reason: str | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                f"# {_now()} {job.kind} {job.config or ''}\n# cmd: {redact(' '.join(cmd))}\n"
            )
            log.flush()
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:  # missing interpreter, permissions
                log.write(f"error: {exc}\n")
                job.returncode = 127
                job.error = str(exc)
            else:
                job.pid = proc.pid
                heartbeat.pid = proc.pid
                # Persist the child pid now that it exists; the file must not say "pending".
                _write_json(running, job.to_dict())
                deadline = (
                    time.monotonic() + job.timeout_s if job.timeout_s is not None else None
                )
                job.returncode, reason = _wait_for_child(proc, stop, deadline=deadline)
                if reason == "timeout":
                    log.write(
                        f"# {_now()} TIMEOUT: exceeded {job.timeout_s}s; "
                        "sending SIGTERM to the process group\n"
                    )
                    log.flush()
                    job.timed_out = True
                    killed = (
                        f" (SIGKILL after {STOP_GRACE_S:g}s grace)"
                        if job.returncode == -_sig.SIGKILL
                        else ""
                    )
                    job.error = (
                        f"timed out after {job.timeout_s}s; the process group was terminated"
                        + killed
                    )
                elif reason == "stop":
                    job.error = "interrupted (worker stopped); the child was terminated"
    finally:
        heartbeat.stop()
    job.duration_s = round(time.monotonic() - started, 1)
    job.finished = _now()
    job.log = str(log_path)
    job.status = DONE if job.returncode == 0 and reason is None else FAILED
    _write_json(root / "done" / job_file.name, job.to_dict())
    running.unlink(missing_ok=True)
    return job


def _recover_running(root: Path) -> int:
    """Fail jobs left in running/ by a previous worker process; never requeue them."""
    recovered = 0
    for path in sorted((root / "running").glob("*.json")):
        ident, kind = _job_details(path)
        job = _record_failure(root, path, ident, kind, INTERRUPTED_ERROR)
        recovered += 1
        print(f"worker: {job.id} was left running; marked failed", flush=True)
    return recovered


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


def _install_stop_handler() -> threading.Event:
    """Turn SIGTERM/SIGINT into a clean loop exit so a job is not left 'running'."""
    stop = threading.Event()

    def handler(_signum: int, _frame: Any) -> None:
        stop.set()

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
    recovered = _recover_running(root)
    if recovered:
        print(
            f"worker: marked {recovered} job(s) left running by a previous process as failed",
            flush=True,
        )
    _heartbeat(root, "idle")
    print(f"worker: serving {root} (interval {interval:g}s)", flush=True)
    print(f"worker: {gpu_note()}", flush=True)
    while not stop.is_set():
        job_file = _next_job(root)
        if job_file is None:
            if once:
                break
            _heartbeat(root, "idle")
            if stop.wait(interval):
                break
            continue
        try:
            job = run_job(root, job_file, argv0=argv0, cwd=cwd, stop=stop)
        except (MsftError, TypeError, KeyError) as exc:
            # A corrupt job file must not crash-loop the worker: record it and move on.
            ident, kind = _job_details(job_file)
            job = _record_failure(root, job_file, ident, kind, f"{type(exc).__name__}: {exc}")
            _heartbeat(root, "idle", job.id)
            print(f"worker: {job.id} -> failed: {job.error}", flush=True)
            if once:
                break
            continue
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
    if job_id is not None:
        _validate_id(job_id)
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
