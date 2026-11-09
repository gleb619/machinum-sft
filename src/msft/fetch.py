"""Resumable Hugging Face weight downloads: `msft fetch`.

`from_pretrained` alone does not resume a dropped shard; on the first real run the 4B cache fell
from 5.8 GB back to 4.2 GB mid-download. This wraps ``snapshot_download(...,
resume_download=True, max_workers=2)`` in a bounded retry loop, prints what each attempt has on
disk, and only reports success once no ``incomplete`` file remains.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from msft.config import MsftError, redact

BACKOFF_S = 15.0
MAX_WORKERS = 2
DEFAULT_ATTEMPTS = 8
INSTALL_HINT = "install the training extra: uv pip install -e '.[train]'"


def _log(message: str) -> None:
    """Progress line on stderr, so a `--json` run keeps stdout to one object."""
    print(redact(message), file=sys.stderr)


def _import(name: str) -> Any:
    """Import a module at call time; tests replace this to simulate a missing dependency."""
    return importlib.import_module(name)


def require(name: str) -> Any:
    """Import a heavy dependency or fail with an actionable message (exit code 2)."""
    try:
        return _import(name)
    except ImportError as exc:
        raise MsftError(f"{name} is not installed; {INSTALL_HINT}") from exc


def human_bytes(count: int) -> str:
    """Binary size label, e.g. ``9.4 GiB``. Symlinks are not counted twice."""
    value = float(max(count, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{count} B"  # pragma: no cover - the loop always returns


def cache_root(model_id: str) -> Path:
    """HF hub cache directory for a model, honouring ``HF_HUB_CACHE`` and ``HF_HOME``."""
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        hub = Path(explicit)
    elif os.environ.get("HF_HOME"):
        hub = Path(os.environ["HF_HOME"]) / "hub"
    else:
        hub = Path.home() / ".cache" / "huggingface" / "hub"
    return hub / f"models--{model_id.replace('/', '--')}"


def _dir_size(path: Path) -> int:
    """Bytes of regular files under ``path``; snapshot symlinks are skipped, not double-counted."""
    total = 0
    for current, _dirs, files in os.walk(path):
        for name in files:
            entry = Path(current) / name
            try:
                if entry.is_symlink():
                    continue
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def _incomplete(root: Path) -> list[str]:
    """Relative paths of files still mid-download (``incomplete/`` or ``*.incomplete``)."""
    found: list[str] = []
    for current, dirs, files in os.walk(root):
        here = Path(current)
        if here.name == "incomplete":
            found.extend(str((here / name).relative_to(root)) for name in files)
            dirs[:] = []
            continue
        found.extend(
            str((here / name).relative_to(root))
            for name in files
            if name.endswith(".incomplete")
        )
    return sorted(found)


def _snapshot_root(snapshot: Path) -> Path:
    """The ``models--org--name`` cache directory when ``snapshot`` has the HF hub layout."""
    parent = snapshot.parent
    if parent.name == "snapshots" and parent.parent.name.startswith("models--"):
        return parent.parent
    return snapshot


def _fatal(exc: BaseException, hub: Any) -> bool:
    """True for failures another attempt cannot fix: gated, missing, bad revision, bad call."""
    if isinstance(exc, TypeError):
        return True
    for name in ("RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError"):
        error_type = getattr(hub, name, None)
        if isinstance(error_type, type) and isinstance(exc, error_type):
            return True
    http_error = getattr(hub, "HfHubHTTPError", None)
    if isinstance(http_error, type) and isinstance(exc, http_error):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status in (401, 403, 404):
            return True
    return False


def run(
    model: str,
    *,
    revision: str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Download ``model`` with resume and retries; returns a JSON-serialisable result."""
    if attempts < 1:
        raise MsftError("--attempts must be at least 1")
    hub = require("huggingface_hub")
    require("torch")

    root = cache_root(model)
    kwargs: dict[str, Any] = {"resume_download": True, "max_workers": MAX_WORKERS}
    if revision:
        kwargs["revision"] = revision

    result: dict[str, Any] = {
        "ok": False,
        "model": model,
        "revision": revision,
        "path": None,
        "attempts": 0,
        "max_attempts": attempts,
        "bytes": 0,
        "size": human_bytes(0),
        "cache": str(root),
        "cache_bytes": 0,
        "cache_size": human_bytes(0),
        "incomplete": [],
        "max_workers": MAX_WORKERS,
        "error": None,
    }

    snapshot: Path | None = None
    for attempt in range(1, attempts + 1):
        result["attempts"] = attempt
        try:
            snapshot = Path(hub.snapshot_download(model, **kwargs))
        except Exception as exc:  # noqa: BLE001 - retry only what another attempt can fix
            message = redact(f"{type(exc).__name__}: {exc}")
            if _fatal(exc, hub):
                raise MsftError(f"cannot download {model}: {message}") from exc
            result["error"] = message
            partial = _dir_size(root) if root.exists() else 0
            _log(
                f"attempt {attempt}/{attempts}: {human_bytes(partial)} downloaded, "
                f"failed: {result['error']}"
            )
            if attempt < attempts:
                _log(f"retrying in {BACKOFF_S:g}s")
                sleep(BACKOFF_S)
            continue
        break

    if snapshot is None:
        size = _dir_size(root) if root.exists() else 0
        result.update(
            bytes=size,
            size=human_bytes(size),
            cache_bytes=size,
            cache_size=human_bytes(size),
        )
        _log(f"all {attempts} attempt(s) failed; last error: {result['error']}")
        return result

    model_root = _snapshot_root(snapshot)
    size = _dir_size(model_root)
    incomplete = _incomplete(model_root)
    _log(f"attempt {result['attempts']}/{attempts}: {human_bytes(size)} at {snapshot}")
    result.update(
        path=str(snapshot),
        bytes=size,
        size=human_bytes(size),
        cache=str(model_root),
        cache_bytes=size,
        cache_size=human_bytes(size),
        incomplete=incomplete,
    )
    if incomplete:
        result["error"] = (
            f"{len(incomplete)} incomplete file(s) remain, e.g. {incomplete[0]}; "
            "re-run `msft fetch` to resume"
        )
    else:
        result["ok"] = True
    return result


def format_report(result: dict[str, Any]) -> str:
    """Human-readable download report: state, sizes and what is still incomplete."""
    revision = f" @ {result['revision']}" if result.get("revision") else ""
    lines = [
        f"fetch {result['model']}{revision}",
        f"  state: {'complete' if result['ok'] else 'FAILED'}",
        f"  snapshot: {result.get('path') or '-'}",
        f"  size: {result['size']} (cache {result['cache_size']})",
        f"  attempts: {result['attempts']}/{result['max_attempts']}",
        f"  incomplete: {', '.join(result['incomplete']) or 'none'}",
    ]
    if result.get("error"):
        lines.append(f"  error: {result['error']}")
    return redact("\n".join(lines))
