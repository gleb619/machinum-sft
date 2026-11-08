"""Tests for `msft.doctor`: no GPU, no network, no torch, every probe monkeypatched."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import namedtuple
from dataclasses import asdict
from pathlib import Path

import pytest

from msft import doctor
from msft.config import DataCfg, ModelCfg, RunConfig

DiskUsage = namedtuple("DiskUsage", "total used free")

CHECK_NAMES = [
    "python",
    "dep:torch",
    "dep:unsloth",
    "dep:trl",
    "dep:peft",
    "dep:datasets",
    "cuda",
    "disk",
    "dataset",
    "hf_token",
]


class FakeRun:
    """Records argv/kwargs and returns a canned CompletedProcess, or raises a canned error."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        error: BaseException | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.error = error
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def cfg(tmp_path: Path) -> RunConfig:
    """A config whose dataset input really exists under tmp_path."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    return RunConfig(
        name="smoke",
        model=ModelCfg(id="Qwen/Qwen3.5-9B"),
        data=DataCfg(input="data/raw"),
        output_dir="out",
        base_dir=str(tmp_path),
    )


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    """No training deps, no nvidia-smi, roomy disk: every probe takes its safe branch."""
    monkeypatch.setattr(doctor, "_find_spec", lambda name: None)
    monkeypatch.setattr(doctor, "_pkg_version", lambda name: None)
    fake = FakeRun(error=FileNotFoundError(2, "nvidia-smi: not found"))
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(
        doctor.shutil,
        "disk_usage",
        lambda path: DiskUsage(500 * doctor.GIB, 100 * doctor.GIB, 400 * doctor.GIB),
    )
    return fake


def by_name(checks: list[doctor.Check]) -> dict[str, doctor.Check]:
    """Index checks by name."""
    return {check.name: check for check in checks}


# --- shape and interpreter ------------------------------------------------


def test_check_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        doctor.Check("x", "maybe", "boom")


def test_run_doctor_shape_is_stable(cfg: RunConfig, offline: FakeRun) -> None:
    checks = doctor.run_doctor(cfg)
    assert [check.name for check in checks] == CHECK_NAMES
    assert all(check.status in {"ok", "warn", "fail"} for check in checks)
    assert json.loads(json.dumps([asdict(check) for check in checks]))


def test_run_doctor_without_config_uses_defaults(offline: FakeRun) -> None:
    checks = doctor.run_doctor(None)
    assert [check.name for check in checks] == CHECK_NAMES


def test_python_version_gate_ok(cfg: RunConfig, offline: FakeRun) -> None:
    assert by_name(doctor.run_doctor(cfg))["python"].status == "ok"


def test_python_version_gate_fails_below_310(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "python_version", lambda: (3, 9, 18))
    check = by_name(doctor.run_doctor(cfg))["python"]
    assert check.status == "fail"
    assert "3.9.18" in check.detail
    assert "3.10" in check.detail


# --- dependencies ---------------------------------------------------------


def test_dependencies_use_find_spec_and_metadata(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(doctor, "_find_spec", lambda name: seen.append(name) or object())
    monkeypatch.setattr(
        doctor, "_pkg_version", lambda name: "2.5.1" if name == "torch" else None
    )
    checks = by_name(doctor.run_doctor(cfg))
    assert seen == ["torch", "unsloth", "trl", "peft", "datasets"]
    assert checks["dep:torch"].status == "ok"
    assert "2.5.1" in checks["dep:torch"].detail
    assert checks["dep:trl"].status == "ok"
    assert "unknown" in checks["dep:trl"].detail
    assert "torch" not in sys.modules  # find_spec only: nothing was actually imported


def test_missing_dependency_is_warn(cfg: RunConfig, offline: FakeRun) -> None:
    check = by_name(doctor.run_doctor(cfg))["dep:datasets"]
    assert check.status == "warn"
    assert "pip install" in check.detail


# --- GPU ------------------------------------------------------------------


def test_cuda_runs_nvidia_smi_with_timeout(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRun(stdout="NVIDIA A100-SXM4-80GB, 81920 MiB, 1024 MiB\n")
    monkeypatch.setattr(subprocess, "run", fake)
    check = by_name(doctor.run_doctor(cfg))["cuda"]
    assert check.status == "ok"
    assert "A100" in check.detail and "81920 MiB" in check.detail
    argv, kwargs = fake.calls[0]
    assert argv == [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader",
    ]
    assert kwargs["timeout"] == 5
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_cuda_absent_is_warn_not_crash(cfg: RunConfig, offline: FakeRun) -> None:
    check = by_name(doctor.run_doctor(cfg))["cuda"]
    assert check.status == "warn"
    assert "nvidia-smi" in check.detail


def test_cuda_timeout_is_warn(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRun(error=subprocess.TimeoutExpired("nvidia-smi", 5))
    monkeypatch.setattr(subprocess, "run", fake)
    check = by_name(doctor.run_doctor(cfg))["cuda"]
    assert check.status == "warn"
    assert "timed out" in check.detail


def test_cuda_nonzero_exit_is_warn(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRun(stderr="No devices were found\n", returncode=9)
    monkeypatch.setattr(subprocess, "run", fake)
    check = by_name(doctor.run_doctor(cfg))["cuda"]
    assert check.status == "warn"
    assert "No devices were found" in check.detail


def test_cuda_without_rows_is_warn(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRun(stdout="\n")
    monkeypatch.setattr(subprocess, "run", fake)
    check = by_name(doctor.run_doctor(cfg))["cuda"]
    assert check.status == "warn"
    assert "no GPUs" in check.detail


def test_parse_gpu_output_skips_junk() -> None:
    rows = doctor.parse_gpu_output("NVIDIA L40S, 46068 MiB, 12 MiB\nbroken row\n\n")
    assert rows == [("NVIDIA L40S", 46068, 12)]
    assert doctor.parse_gpu_output("") == []


# --- disk -----------------------------------------------------------------


def test_disk_measures_nearest_existing_output_parent(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def usage(path: str | Path) -> DiskUsage:
        seen.append(Path(path))
        return DiskUsage(200 * doctor.GIB, 50 * doctor.GIB, 150 * doctor.GIB)

    monkeypatch.setattr(doctor.shutil, "disk_usage", usage)
    check = by_name(doctor.run_doctor(cfg))["disk"]
    assert seen == [Path(cfg.base_dir)]
    assert check.status == "ok"
    assert "150.0 GiB free" in check.detail


def test_disk_warns_and_fails_on_low_space(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def usage(total_gib: float, used_gib: float, free_gib: float) -> DiskUsage:
        return DiskUsage(total_gib * doctor.GIB, used_gib * doctor.GIB, free_gib * doctor.GIB)

    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda path: usage(200, 190, 10))
    assert by_name(doctor.run_doctor(cfg))["disk"].status == "warn"
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda path: usage(200, 198, 2))
    assert by_name(doctor.run_doctor(cfg))["disk"].status == "fail"


def test_disk_error_is_warn(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: str | Path) -> DiskUsage:
        raise PermissionError("denied")

    monkeypatch.setattr(doctor.shutil, "disk_usage", broken)
    check = by_name(doctor.run_doctor(cfg))["disk"]
    assert check.status == "warn"
    assert "cannot measure" in check.detail


# --- dataset and token ----------------------------------------------------


def test_dataset_present_is_ok(cfg: RunConfig, offline: FakeRun) -> None:
    check = by_name(doctor.run_doctor(cfg))["dataset"]
    assert check.status == "ok"
    assert "data/raw" in check.detail


def test_dataset_missing_is_warn(cfg: RunConfig, offline: FakeRun) -> None:
    cfg.data.input = "data/does-not-exist"
    check = by_name(doctor.run_doctor(cfg))["dataset"]
    assert check.status == "warn"
    assert "data build" in check.detail


def test_gemma_without_hf_token_warns(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg.model.id = "google/gemma-3-12b-it"
    check = by_name(doctor.run_doctor(cfg))["hf_token"]
    assert check.status == "warn"
    assert "HF_TOKEN unset" in check.detail


def test_gemma_with_hf_token_is_ok_and_token_is_never_echoed(
    cfg: RunConfig, offline: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_testonlyvalue1234567890")
    cfg.model.id = "google/gemma-3-4b-it"
    checks = doctor.run_doctor(cfg)
    assert by_name(checks)["hf_token"].status == "ok"
    report = doctor.format_report(checks, False)
    assert "hf_testonlyvalue" not in report


def test_token_not_required_for_other_models(cfg: RunConfig, offline: FakeRun) -> None:
    check = by_name(doctor.run_doctor(cfg))["hf_token"]
    assert check.status == "ok"
    assert "not required" in check.detail


# --- report ---------------------------------------------------------------


def test_format_report_json_is_single_object(cfg: RunConfig, offline: FakeRun) -> None:
    checks = doctor.run_doctor(cfg)
    payload = json.loads(doctor.format_report(checks, True))
    assert payload["status"] == doctor.overall_status(checks)
    assert payload["checks"][0]["name"] == "python"
    assert sum(payload["summary"].values()) == len(checks)
    assert set(payload["summary"]) == {"ok", "warn", "fail"}


def test_format_report_text_has_every_check(cfg: RunConfig, offline: FakeRun) -> None:
    checks = doctor.run_doctor(cfg)
    text = doctor.format_report(checks, False)
    assert text.startswith("doctor: ")
    assert text.splitlines()[-1] == f"result: {doctor.overall_status(checks)}"
    for check in checks:
        assert check.name in text


def test_format_report_redacts_paths_and_tokens() -> None:
    checks = [
        doctor.Check("x", "warn", "/home/alice/proj"),
        doctor.Check("y", "fail", "key hf_abcdefgh12345678"),
    ]
    for output in (doctor.format_report(checks, False), doctor.format_report(checks, True)):
        assert "/home/alice" not in output
        assert "hf_abcdefgh12345678" not in output
