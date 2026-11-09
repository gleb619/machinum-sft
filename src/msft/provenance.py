"""Provenance sidecars for quantised GGUF artifacts.

A sidecar sits next to a ``.gguf`` and records what produced it: the run config, the merged
directory's digest and file list, the f16 intermediate, the quantisation method and the
llama.cpp commit. It is written atomically by ``msft quantize`` after the GGUF is renamed into
place, so a crash can leave a GGUF without a sidecar but never a sidecar vouching for a file
that is not there.

Only file names are stored, never absolute paths or hosts. stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msft import __version__
from msft.config import MsftError

SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".provenance.json"
COMMIT_ENV = "MSFT_LLAMA_CPP_COMMIT"
COMMIT_FILE = ".msft-commit"
GIT_TIMEOUT_S = 5
CHUNK_BYTES = 8 << 20


def sidecar_path(gguf: str | Path) -> Path:
    """`out/model.gguf` -> `out/model.gguf.provenance.json`."""
    return Path(str(gguf) + SIDECAR_SUFFIX)


def sha256_file(path: str | Path, chunk: int = CHUNK_BYTES) -> str:
    """Hex sha256 of a file, read `chunk` bytes at a time so a 5 GB GGUF fits in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_bytes(path: Path) -> int:
    """Size of a regular file, or 0 when it is missing or unreadable."""
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _weight_names(merged: Path) -> list[str]:
    """Sorted file names of the merged weights, never paths."""
    names: list[str] = []
    for entry in merged.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix == ".safetensors" or (
            entry.name.startswith("pytorch_model") and entry.suffix == ".bin"
        ):
            names.append(entry.name)
    return sorted(names)


def merged_snapshot(merged_dir: str | Path) -> dict[str, Any]:
    """Digest `config.json` and list every merged weight file with its size."""
    merged = Path(merged_dir)
    config = merged / "config.json"
    if not config.is_file():
        raise MsftError(
            f"{merged.name} has no config.json; cannot snapshot the merged model"
        )
    files = [
        {"name": name, "bytes": _file_bytes(merged / name)} for name in _weight_names(merged)
    ]
    return {
        "dir_name": merged.name,
        "config_sha256": sha256_file(config),
        "weights_bytes": sum(entry["bytes"] for entry in files),
        "weights_files": files,
    }


def llama_cpp_info(convert_dir: str | Path | None) -> dict[str, Any]:
    """The llama.cpp commit: env var, then `.msft-commit`, then `git rev-parse HEAD`."""
    env_commit = os.environ.get(COMMIT_ENV, "").strip()
    if env_commit:
        return {"commit": env_commit, "source": "env"}
    if convert_dir is None:
        return {"commit": None, "source": None}
    directory = Path(convert_dir)
    try:
        file_commit = (directory / COMMIT_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        file_commit = ""
    if file_commit:
        return {"commit": file_commit, "source": "file"}
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "source": None}
    commit = (completed.stdout or "").strip()
    if completed.returncode != 0 or not commit:
        return {"commit": None, "source": None}
    return {"commit": commit, "source": "git"}


def build_record(
    *,
    config_name: str,
    model_id: str,
    revision: str | None,
    merged: dict[str, Any],
    f16: str | Path,
    gguf: str | Path,
    method: str,
    imatrix: bool,
    llama_cpp: dict[str, Any],
    created: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the schema-1 provenance record for one quantised GGUF."""
    f16_path = Path(f16)
    gguf_path = Path(gguf)
    stamp = created or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return {
        "schema": SCHEMA_VERSION,
        "msft_version": __version__,
        "created": stamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "config_name": config_name,
        "model": {"id": model_id, "revision": revision},
        "merged": dict(merged),
        "f16": {"name": f16_path.name, "bytes": _file_bytes(f16_path)},
        "gguf": {
            "name": gguf_path.name,
            "bytes": _file_bytes(gguf_path),
            "sha256": sha256_file(gguf_path),
        },
        "quantize": {"method": method, "imatrix": bool(imatrix)},
        "llama_cpp": dict(llama_cpp),
    }


def write(gguf: str | Path, record: dict[str, Any]) -> Path:
    """Write the sidecar for `gguf` atomically; returns its path."""
    target = sidecar_path(gguf)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    handle_fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


_RECORD_SHAPES: dict[str, type] = {
    "config_name": str,
    "model": dict,
    "merged": dict,
    "f16": dict,
    "gguf": dict,
    "quantize": dict,
}


def load(gguf: str | Path) -> tuple[dict[str, Any] | None, str]:
    """Read the sidecar next to `gguf`; returns ``(record|None, reason)``."""
    path = sidecar_path(gguf)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "absent"
    except OSError as exc:
        return None, f"corrupt: {exc}"
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"corrupt: {exc}"
    if not isinstance(record, dict):
        return None, "corrupt: not a JSON object"
    schema = record.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool):
        return None, "corrupt: missing or non-integer schema"
    if schema != SCHEMA_VERSION:
        return None, f"schema {schema} unsupported"
    for key, kind in _RECORD_SHAPES.items():
        if not isinstance(record.get(key), kind):
            return None, f"corrupt: missing or malformed '{key}'"
    return record, ""


def _short(value: Any) -> str:
    """First eight characters of a digest, with an explicit ellipsis when present."""
    text = str(value) if value is not None else ""
    return f"{text[:8]}…" if text else "<missing>"


def _mapping(value: Any) -> dict[str, Any]:
    """`value` as a dict, or an empty one; a malformed sidecar must not raise here."""
    return value if isinstance(value, dict) else {}


def compare(
    record: dict[str, Any],
    *,
    gguf: str | Path,
    merged: str | Path,
    f16: str | Path | None,
    cfg_name: str,
    method: str,
    deep: bool = False,
) -> list[str]:
    """Human-readable mismatches between a sidecar and the files on disk; empty when they agree."""
    problems: list[str] = []
    if record.get("config_name") != cfg_name:
        problems.append(
            f"config name {cfg_name!r} != recorded {record.get('config_name')!r}"
        )

    recorded_quant = _mapping(record.get("quantize"))
    recorded_method = str(recorded_quant.get("method") or "")
    if recorded_method.upper() != method.upper():
        problems.append(
            f"quantize method {method} != recorded {recorded_method or '<missing>'}"
        )

    recorded_merged = _mapping(record.get("merged"))
    try:
        snapshot = merged_snapshot(merged)
    except MsftError as exc:
        problems.append(f"cannot snapshot the merged directory: {exc}")
    else:
        if snapshot["config_sha256"] != recorded_merged.get("config_sha256"):
            problems.append(
                f"merged config.json sha256 {_short(snapshot['config_sha256'])} != recorded "
                f"{_short(recorded_merged.get('config_sha256'))} (the merged model changed "
                "after this GGUF was quantized); re-run quantize"
            )
        recorded_files = recorded_merged.get("weights_files")
        if not isinstance(recorded_files, list) or snapshot["weights_files"] != recorded_files:
            recorded_count = len(recorded_files) if isinstance(recorded_files, list) else 0
            problems.append(
                f"merged weight files differ from the recorded set "
                f"({len(snapshot['weights_files'])} now, {recorded_count} recorded); the merged "
                "model changed after this GGUF was quantized; re-run quantize"
            )

    gguf_path = Path(gguf)
    recorded_gguf = _mapping(record.get("gguf"))
    actual_bytes = _file_bytes(gguf_path)
    if actual_bytes != recorded_gguf.get("bytes"):
        problems.append(
            f"{gguf_path.name} is {actual_bytes} bytes, recorded "
            f"{recorded_gguf.get('bytes')}"
        )
    if deep:
        actual_sha = sha256_file(gguf_path) if gguf_path.is_file() else ""
        if actual_sha != recorded_gguf.get("sha256"):
            problems.append(
                f"{gguf_path.name} sha256 {_short(actual_sha)} != recorded "
                f"{_short(recorded_gguf.get('sha256'))}"
            )

    if f16 is not None:
        f16_path = Path(f16)
        if f16_path.is_file():
            recorded_f16 = _mapping(record.get("f16"))
            actual_f16 = _file_bytes(f16_path)
            if actual_f16 != recorded_f16.get("bytes"):
                problems.append(
                    f"{f16_path.name} is {actual_f16} bytes, recorded "
                    f"{recorded_f16.get('bytes')}"
                )
    return problems
