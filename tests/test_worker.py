"""Worker queue tests: signalling, execution bookkeeping, status and logs. Offline only."""

from __future__ import annotations

import json
import sys
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
        returncode = 0

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, check=False):
        seen["cmd"] = cmd
        assert stdout is not None
        stdout.write("pretend output\n")
        return Done()

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    job = worker.run_job(tmp_path, tmp_path / "inbox" / "j2.json")

    assert job.status == worker.DONE
    assert job.returncode == 0
    assert job.duration_s is not None and job.duration_s >= 0
    assert (tmp_path / "done" / "j2.json").exists()
    assert not (tmp_path / "inbox" / "j2.json").exists()
    assert not (tmp_path / "running" / "j2.json").exists()
    assert "pretend output" in (tmp_path / "logs" / "j2.log").read_text()
    assert seen["cmd"][-1] == "--execute"


def test_run_job_marks_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker.signal(tmp_path, "train", job_id="bad")

    class Failed:
        returncode = 3

    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **k: Failed())
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
