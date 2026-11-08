"""Environment probing for `msft doctor`.

Nothing here imports torch, unsloth or any other training package: presence is read with
`importlib.util.find_spec` and versions with `importlib.metadata`, so the module stays cheap and
`msft --help` / `msft doctor` work on a bare interpreter. A missing GPU is a warning, never a
crash.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from importlib.util import find_spec as _find_spec
from pathlib import Path

from .config import RunConfig, redact

PYTHON_MIN = (3, 10)
TRAINING_DEPS = ("torch", "unsloth", "trl", "peft", "datasets")
GPU_QUERY = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"]
GPU_TIMEOUT_S = 5
GIB = 1024**3
DISK_WARN_GIB = 20.0
DISK_FAIL_GIB = 5.0
STATUSES = ("ok", "warn", "fail")

__all__ = [
    "DISK_FAIL_GIB",
    "DISK_WARN_GIB",
    "GIB",
    "GPU_QUERY",
    "PYTHON_MIN",
    "TRAINING_DEPS",
    "Check",
    "describe_gpus",
    "format_report",
    "overall_status",
    "parse_gpu_output",
    "python_version",
    "run_doctor",
]


@dataclass
class Check:
    """One probe result; `status` is always one of `ok`, `warn` or `fail`."""

    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")


def python_version() -> tuple[int, int, int]:
    """Return the running interpreter version as (major, minor, micro)."""
    info = sys.version_info
    return (info.major, info.minor, info.micro)


def _check_python() -> Check:
    version = python_version()
    text = ".".join(str(part) for part in version)
    floor = f"{PYTHON_MIN[0]}.{PYTHON_MIN[1]}"
    if version[:2] >= PYTHON_MIN:
        return Check("python", "ok", f"{text} (>= {floor})")
    return Check("python", "fail", f"{text} is older than {floor}; upgrade the interpreter")


def _package_version(name: str) -> str | None:
    """Read an installed distribution version without importing the package."""
    try:
        return _pkg_version(name)
    except (PackageNotFoundError, ValueError):
        return None


def _check_package(name: str) -> Check:
    label = f"dep:{name}"
    try:
        spec = _find_spec(name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is None:
        return Check(label, "warn", f"{name} not installed; pip install -e '.[train]'")
    version = _package_version(name)
    detail = f"{version} installed" if version else "installed (version unknown)"
    return Check(label, "ok", detail)


def parse_gpu_output(text: str) -> list[tuple[str, int, int]]:
    """Parse `nvidia-smi --query-gpu` csv rows into (name, total_mib, used_mib) tuples."""
    rows: list[tuple[str, int, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total = int(float(parts[1].split()[0]))
            used = int(float(parts[2].split()[0]))
        except (IndexError, ValueError):
            continue
        rows.append((parts[0], total, used))
    return rows


def describe_gpus(rows: list[tuple[str, int, int]]) -> tuple[str, str]:
    """Return (status, detail) for parsed GPU rows; an empty list means no visible GPU."""
    if not rows:
        return "warn", "nvidia-smi reported no GPUs"
    name, total, used = rows[0]
    percent = (used / total * 100.0) if total else 0.0
    detail = f"{name}, {total} MiB total, {used} MiB used ({percent:.0f}%)"
    if len(rows) > 1:
        detail += f"; {len(rows) - 1} more GPU(s) visible"
    return "ok", detail


def _check_cuda() -> Check:
    try:
        proc = subprocess.run(
            GPU_QUERY,
            capture_output=True,
            text=True,
            timeout=GPU_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Check("cuda", "warn", f"nvidia-smi timed out after {GPU_TIMEOUT_S}s; CPU-only?")
    except OSError as exc:
        return Check("cuda", "warn", f"nvidia-smi unavailable ({exc.__class__.__name__}); CPU-only")
    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        reason = lines[0] if lines else f"exit {proc.returncode}"
        return Check("cuda", "warn", f"nvidia-smi failed: {reason}")
    status, detail = describe_gpus(parse_gpu_output(proc.stdout or ""))
    return Check("cuda", status, detail)


def _nearest_existing(path: Path) -> Path:
    """Walk up until a parent that exists, so an unwritten output dir can still be measured."""
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _check_disk(path: Path) -> Check:
    target = _nearest_existing(path)
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return Check("disk", "warn", f"cannot measure free space at {target}: {exc}")
    free = usage.free / GIB
    total = usage.total / GIB
    detail = f"{free:.1f} GiB free of {total:.1f} GiB at {target}"
    if free < DISK_FAIL_GIB:
        return Check("disk", "fail", f"{detail} (under {DISK_FAIL_GIB:.0f} GiB)")
    if free < DISK_WARN_GIB:
        return Check("disk", "warn", f"{detail} (under {DISK_WARN_GIB:.0f} GiB)")
    return Check("disk", "ok", detail)


def _dataset_path(cfg: RunConfig) -> Path:
    raw = Path(cfg.data.input)
    return raw if raw.is_absolute() else Path(cfg.base_dir) / raw


def _check_dataset(cfg: RunConfig) -> Check:
    path = _dataset_path(cfg)
    if path.exists():
        return Check("dataset", "ok", f"{path} present")
    return Check("dataset", "warn", f"{path} missing; run `msft data build` first")


def _check_hf_token(cfg: RunConfig) -> Check:
    model = cfg.model.id.lower()
    if not model.startswith("google/gemma"):
        return Check("hf_token", "ok", "not required for this model")
    if os.environ.get("HF_TOKEN"):
        return Check("hf_token", "ok", "HF_TOKEN is set")
    return Check("hf_token", "warn", f"HF_TOKEN unset; {cfg.model.id} is gated on Hugging Face")


def run_doctor(cfg: RunConfig | None = None) -> list[Check]:
    """Probe interpreter, training deps, GPU, disk, dataset and gated-model token; never raises."""
    config = cfg or RunConfig()
    checks = [_check_python()]
    checks.extend(_check_package(name) for name in TRAINING_DEPS)
    checks.append(_check_cuda())
    checks.append(_check_disk(config.output_path))
    checks.append(_check_dataset(config))
    checks.append(_check_hf_token(config))
    return checks


def overall_status(checks: list[Check]) -> str:
    """Worst status across checks: fail beats warn beats ok."""
    found = {check.status for check in checks}
    for status in ("fail", "warn"):
        if status in found:
            return status
    return "ok"


def format_report(checks: list[Check], as_json: bool = False) -> str:
    """Render checks as a human table, or as a single JSON object when `as_json` is set."""
    counts = {status: sum(check.status == status for check in checks) for status in STATUSES}
    status = overall_status(checks)
    if as_json:
        payload = {
            "status": status,
            "summary": counts,
            "checks": [asdict(check) for check in checks],
        }
        return redact(json.dumps(payload, ensure_ascii=False, indent=2))
    lines = [f"doctor: {counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail"]
    lines.extend(f"  {check.status:<4}  {check.name:<12}  {check.detail}" for check in checks)
    lines.append(f"result: {status}")
    return redact("\n".join(lines))
