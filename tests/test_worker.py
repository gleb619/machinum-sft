"""Worker queue tests: signalling, execution bookkeeping, status and logs. Offline only."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from msft import worker
from msft.config import MsftError


def test_signal_creates_job_file(tmp_path: Path) -> None:
    job_id = worker.signal(tmp_path, "train", job_id="j1", config="configs/x.yaml")
    path = tmp_path / "inbox" / "j1.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["id"] == job_id == "j1"
    assert data["kind"] == "train"
    assert data["config"] == "configs/x.yaml"
    assert data["status"] == worker.PENDING
    assert not list(tmp_path.glob("inbox/*.tmp"))


def test_signal_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(MsftError):
        worker.signal(tmp_path, "rm-rf")


def test_build_command_appends_execute_and_config() -> None:
    job = worker.Job(id="j", kind="train", config="c.yaml")
    cmd = worker.build_command(job, [sys.executable])
    assert cmd[:4] == [sys.executable, "-m", "msft", "train"]
    assert cmd[4:6] == ["--config", "c.yaml"]
    assert cmd[-1] == "--execute"
    assert cmd.count("--execute") == 1


def test_run_job_moves_files_and_records_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker.signal(tmp_path, "merge", job_id="j2", config="c.yaml")
    seen: dict[str, object] = {}

    class Done:
        pid = 4242
        returncode: int | None = None

        def __init__(self, cmd, cwd=None, stdout=None, stderr=None, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs
            assert stdout is not None
            stdout.write("pretend output\n")

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", Done)
    job = worker.run_job(tmp_path, tmp_path / "inbox" / "j2.json")

    assert job.status == worker.DONE
    assert job.returncode == 0
    assert job.pid == 4242
    assert job.duration_s is not None and job.duration_s >= 0
    finished = json.loads((tmp_path / "done" / "j2.json").read_text())
    assert finished["pid"] == 4242
    assert finished["status"] == worker.DONE
    assert not (tmp_path / "inbox" / "j2.json").exists()
    assert not (tmp_path / "running" / "j2.json").exists()
    assert "pretend output" in (tmp_path / "logs" / "j2.log").read_text()
    assert seen["cmd"][-1] == "--execute"
    assert seen["kwargs"]["start_new_session"] is True


def test_run_job_marks_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker.signal(tmp_path, "train", job_id="bad")

    class Failed:
        pid = 1
        returncode: int | None = None

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            self.returncode = 3
            return 3

    monkeypatch.setattr(worker.subprocess, "Popen", Failed)
    job = worker.run_job(tmp_path, tmp_path / "inbox" / "bad.json")
    assert job.status == worker.FAILED
    assert job.returncode == 3


def test_serve_once_runs_doctor_end_to_end(tmp_path: Path) -> None:
    """Real child process, no network: `doctor` prints a report and exits 0."""
    worker.signal(tmp_path, "doctor", job_id="d1")
    code = worker.serve(tmp_path, once=True, interval=0.1)
    assert code == 0
    finished = json.loads((tmp_path / "done" / "d1.json").read_text())
    assert finished["status"] == worker.DONE
    log = (tmp_path / "logs" / "d1.log").read_text()
    assert "doctor" in log
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["state"] in {"idle", "stopped"}


def test_status_counts_and_heartbeat(tmp_path: Path) -> None:
    worker.signal(tmp_path, "merge", job_id="a")
    worker.signal(tmp_path, "quantize", job_id="b")
    state = worker.status(tmp_path)
    assert state["counts"]["pending"] == 2
    assert state["counts"]["running"] == 0
    assert state["counts"]["done"] == 0
    assert state["heartbeat"] is None

    worker._heartbeat(worker.worker_dir(tmp_path), "idle")
    state = worker.status(tmp_path)
    assert state["heartbeat"]["state"] == "idle"
    assert isinstance(state["heartbeat"]["age_s"], float)


def test_tail_log_returns_latest(tmp_path: Path) -> None:
    root = worker.worker_dir(tmp_path)
    (root / "logs" / "one.log").write_text("first\nsecond\n")
    assert worker.tail_log(tmp_path, lines=1).strip() == "second"
    assert worker.tail_log(tmp_path, "one", lines=5).startswith("first")
    with pytest.raises(MsftError):
        worker.tail_log(tmp_path, "missing")


def test_reset_clears_queue_keeps_logs(tmp_path: Path) -> None:
    worker.signal(tmp_path, "train", job_id="x")
    root = worker.worker_dir(tmp_path)
    (root / "logs" / "x.log").write_text("kept\n")
    removed = worker.reset(tmp_path)
    assert removed >= 1
    assert worker.status(tmp_path)["counts"]["pending"] == 0
    assert (root / "logs" / "x.log").exists()

def test_gpu_note_reports_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class Proc:
        returncode = 0
        stdout = "0, NVIDIA GeForce RTX 4060 Ti, 16380 MiB\n"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: Proc())
    assert worker.gpu_note().startswith("GPU: 0, NVIDIA GeForce RTX 4060 Ti")


def test_gpu_note_warns_when_there_is_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    assert worker.gpu_note().startswith("WARNING: nvidia-smi not found")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert "CUDA_VISIBLE_DEVICES is empty" in worker.gpu_note()


def test_status_carries_the_gpu_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "gpu_note", lambda: "GPU: 0, test card")
    worker.signal(tmp_path, "doctor", job_id="g")
    assert worker.status(tmp_path)["gpu"] == "GPU: 0, test card"


# --- job ids (collision, traversal) --------------------------------------


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "x" * 81, ".hidden", "sp ace"])
def test_signal_rejects_unsafe_ids(tmp_path: Path, bad: str) -> None:
    with pytest.raises(MsftError, match="invalid job id"):
        worker.signal(tmp_path, "train", job_id=bad)
    assert not list(tmp_path.glob("inbox/*.json"))


def test_signal_never_overwrites_an_existing_job(tmp_path: Path) -> None:
    worker.signal(tmp_path, "train", job_id="same", config="first.yaml")
    with pytest.raises(MsftError, match="already exists"):
        worker.signal(tmp_path, "merge", job_id="same", config="second.yaml")
    data = json.loads((tmp_path / "inbox" / "same.json").read_text())
    assert data["config"] == "first.yaml"


def test_automatic_ids_are_unique(tmp_path: Path) -> None:
    first = worker.signal(tmp_path, "train")
    second = worker.signal(tmp_path, "train")
    assert first != second
    assert {path.stem for path in (tmp_path / "inbox").glob("*.json")} == {first, second}


def test_tail_log_rejects_traversal(tmp_path: Path) -> None:
    worker.worker_dir(tmp_path)
    with pytest.raises(MsftError, match="invalid job id"):
        worker.tail_log(tmp_path, "../heartbeat")


# --- corrupt and interrupted jobs ---------------------------------------


def test_a_corrupt_job_is_recorded_failed_and_the_worker_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-written or hand-edited job file must not crash-loop the worker."""
    root = worker.worker_dir(tmp_path)
    (root / "inbox" / "broken.json").write_text("{not json", encoding="utf-8")
    worker.signal(tmp_path, "doctor", job_id="after")
    stop = worker._install_stop_handler()
    monkeypatch.setattr(worker, "_install_stop_handler", lambda: stop)
    thread = threading.Thread(target=worker.serve, args=(tmp_path,), kwargs={"interval": 0.01})
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if (root / "done" / "broken.json").exists() and (
                root / "done" / "after.json"
            ).exists():
                break
            time.sleep(0.02)
        stop.set()
    finally:
        thread.join(timeout=30)

    broken = json.loads((root / "done" / "broken.json").read_text())
    assert broken["status"] == worker.FAILED
    assert broken["error"]
    assert not (root / "inbox" / "broken.json").exists()
    after = json.loads((root / "done" / "after.json").read_text())
    assert after["status"] == worker.DONE


def test_a_job_left_running_is_failed_not_requeued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a restart the old child is gone; re-running a train job silently would be worse."""
    root = worker.worker_dir(tmp_path)
    job = worker.Job(id="orphan", kind="train")
    (root / "running" / "orphan.json").write_text(json.dumps(job.to_dict()), encoding="utf-8")

    assert worker.serve(tmp_path, once=True, interval=0.01) == 0

    finished = json.loads((root / "done" / "orphan.json").read_text())
    assert finished["status"] == worker.FAILED
    assert finished["error"] == worker.INTERRUPTED_ERROR
    assert not (root / "running" / "orphan.json").exists()
    assert not (root / "inbox" / "orphan.json").exists()


# --- heartbeat and stop forwarding --------------------------------------


def test_run_job_persists_running_state_and_child_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long jobs used to show a stale 'idle' heartbeat and a 'pending' file for hours."""
    worker.signal(tmp_path, "merge", job_id="slow")
    seen: dict[str, object] = {}

    class Slow:
        pid = 99
        returncode: int | None = None

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            data = json.loads((tmp_path / "running" / "slow.json").read_text())
            seen["status"] = data["status"]
            seen["pid"] = data["pid"]
            seen["started"] = data["started"]
            self.returncode = 0
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", Slow)
    worker.run_job(tmp_path, tmp_path / "inbox" / "slow.json")

    assert seen["status"] == worker.RUNNING
    assert seen["pid"] == 99
    assert seen["started"]


def test_heartbeat_thread_refreshes_during_a_job(tmp_path: Path) -> None:
    root = worker.worker_dir(tmp_path)
    beat = worker._Heartbeat(root, "long", interval=0.01)
    beat.pid = 77
    beat.start()
    try:
        deadline = time.monotonic() + 5
        data: dict[str, object] = {}
        while time.monotonic() < deadline:
            if (root / "heartbeat.json").is_file():
                data = json.loads((root / "heartbeat.json").read_text())
                if data.get("state") == worker.RUNNING:
                    break
            time.sleep(0.01)
    finally:
        beat.stop()
    assert data["state"] == worker.RUNNING
    assert data["job"] == "long"
    assert data["child_pid"] == 77


def test_a_stop_request_terminates_the_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Proc:
        pid = 4321
        returncode: int | None = None

        def __init__(self) -> None:
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            self.returncode = -15
            return -15

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(worker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    stop = threading.Event()
    stop.set()
    returncode, reason = worker._wait_for_child(Proc(), stop)
    assert reason == "stop"
    assert returncode == -15
    assert signals == [(4321, worker._sig.SIGTERM)]


# --- timeouts -------------------------------------------------------------


def test_wait_for_child_reports_timeout_when_the_deadline_has_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Proc:
        pid = 4321
        returncode: int | None = None

        def __init__(self) -> None:
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            self.returncode = -15
            return -15

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(worker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    returncode, reason = worker._wait_for_child(
        Proc(), None, deadline=time.monotonic() - 1
    )
    assert reason == "timeout"
    assert returncode == -15
    assert signals == [(4321, worker._sig.SIGTERM)]


def test_wait_for_child_without_a_deadline_does_not_time_out() -> None:
    class Proc:
        pid = 1
        returncode: int | None = None

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    assert worker._wait_for_child(Proc(), None, deadline=None) == (0, None)


def test_a_stop_request_wins_over_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class Proc:
        pid = 4321
        returncode: int | None = None

        def __init__(self) -> None:
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            self.returncode = -15
            return -15

    monkeypatch.setattr(worker.os, "killpg", lambda pid, sig: None)
    stop = threading.Event()
    stop.set()
    _returncode, reason = worker._wait_for_child(
        Proc(), stop, deadline=time.monotonic() - 1
    )
    assert reason == "stop"


def test_run_job_times_out_a_real_sleeping_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker.signal(tmp_path, "doctor", job_id="slow", timeout_s=1)
    monkeypatch.setattr(
        worker,
        "build_command",
        lambda job, argv0: [sys.executable, "-c", "import time; time.sleep(30)"],
    )

    started = time.monotonic()
    job = worker.run_job(tmp_path, tmp_path / "inbox" / "slow.json")
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert job.status == worker.FAILED
    assert job.timed_out is True
    assert job.returncode == -worker._sig.SIGTERM
    assert job.error == "timed out after 1s; the process group was terminated"
    log = (tmp_path / "logs" / "slow.log").read_text()
    assert "TIMEOUT: exceeded 1s" in log
    assert "sending SIGTERM to the process group" in log


def test_a_timeout_kills_the_whole_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pidfile = tmp_path / "grandchild.pid"
    script = f"sleep 30 & echo $! > {shlex.quote(str(pidfile))}; wait"
    worker.signal(tmp_path, "doctor", job_id="tree", timeout_s=1)
    monkeypatch.setattr(worker, "build_command", lambda job, argv0: ["sh", "-c", script])

    job = worker.run_job(tmp_path, tmp_path / "inbox" / "tree.json")
    assert job.timed_out is True

    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {pid} survived the process-group kill")


@pytest.mark.parametrize("bad", [0, -1, True, "5"])
def test_signal_rejects_an_invalid_timeout(tmp_path: Path, bad: object) -> None:
    with pytest.raises(MsftError, match="invalid timeout_s"):
        worker.signal(tmp_path, "train", timeout_s=bad)  # type: ignore[arg-type]
    assert not list(tmp_path.glob("inbox/*.json"))


def test_signal_writes_a_valid_timeout(tmp_path: Path) -> None:
    worker.signal(tmp_path, "train", job_id="t", timeout_s=7200)
    data = json.loads((tmp_path / "inbox" / "t.json").read_text())
    assert data["timeout_s"] == 7200


def test_a_job_file_without_a_timeout_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = worker.worker_dir(tmp_path)
    payload = worker.Job(id="old", kind="doctor").to_dict()
    payload.pop("timeout_s")
    payload.pop("timed_out")
    (root / "inbox" / "old.json").write_text(json.dumps(payload), encoding="utf-8")

    class Done:
        pid = 1
        returncode: int | None = None

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", Done)
    job = worker.run_job(root, root / "inbox" / "old.json")
    assert job.status == worker.DONE
    assert job.timeout_s is None
    assert job.timed_out is False


def test_a_job_with_an_invalid_timeout_is_quarantined(tmp_path: Path) -> None:
    root = worker.worker_dir(tmp_path)
    payload = worker.Job(id="badto", kind="doctor").to_dict()
    payload["timeout_s"] = "abc"
    (root / "inbox" / "badto.json").write_text(json.dumps(payload), encoding="utf-8")
    worker.signal(tmp_path, "doctor", job_id="zz-after")

    assert worker.serve(tmp_path, once=True, interval=0.01) == 0
    broken = json.loads((root / "done" / "badto.json").read_text())
    assert broken["status"] == worker.FAILED
    assert "invalid timeout_s" in broken["error"]

    # The worker survived the hand-edited file and still runs the next job.
    assert worker.serve(tmp_path, once=True, interval=0.01) == 0
    after = json.loads((root / "done" / "zz-after.json").read_text())
    assert after["status"] == worker.DONE


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90", 90), ("90s", 90), ("45m", 2700), ("2h", 7200), ("1d", 86400), ("30m", 1800)],
)
def test_parse_duration_accepts_whole_units(text: str, seconds: int) -> None:
    assert worker.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["0", "-5", "1.5h", "2x", "", "1/2", "99999d"])
def test_parse_duration_rejects_everything_else(text: str) -> None:
    with pytest.raises(MsftError, match="invalid --timeout"):
        worker.parse_duration(text)


def test_run_job_rejects_an_unknown_kind(tmp_path: Path) -> None:
    root = worker.worker_dir(tmp_path)
    payload = worker.Job(id="evil", kind="rm-rf").to_dict()
    (root / "inbox" / "evil.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MsftError, match="unknown kind"):
        worker.run_job(root, root / "inbox" / "evil.json")


def test_compose_allows_a_long_job_to_be_stopped_gracefully() -> None:
    """SIGTERM then 60 s before SIGKILL: training checkpoints take longer than the default 10 s."""
    import yaml

    path = Path(__file__).resolve().parents[1] / "deploy" / "remote" / "docker-compose.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["services"]["worker"]["stop_grace_period"] == "60s"
