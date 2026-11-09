"""Create (and optionally push) an ollama model from a GGUF: `msft export`.

This replaces ``scripts/ollama-export.sh``. Two rules from the first real run are load-bearing:

* the Modelfile carries **no** ``TEMPLATE`` — the GGUF has the tokenizer's own chat template, and
  a hand-copied one once made a Qwen model answer in Gemma's format and narrate its reasoning;
* a bare model name pushes to ``library/``, which nobody owns, so ``--push`` requires
  ``--namespace``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from msft.config import MsftError, redact

MIN_GGUF_BYTES = 500_000_000
DEFAULT_CONTAINER = "ollama"
NUM_CTX = 8192
TEMPERATURE = 0
TOP_P = 0.95
REPEAT_PENALTY = 1.05


def _log(message: str) -> None:
    """Progress line on stderr, so a `--json` run keeps stdout to one object."""
    print(redact(message), file=sys.stderr)


def _size_label(count: int) -> str:
    """Binary size label, e.g. ``2.6 GiB``."""
    value = float(max(count, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{count} B"  # pragma: no cover - the loop always returns


def build_modelfile(source: str) -> str:
    """Modelfile body with sampling parameters and deliberately no ``TEMPLATE``."""
    return "\n".join(
        [
            f"FROM {source}",
            "",
            f"PARAMETER num_ctx {NUM_CTX}",
            f"PARAMETER temperature {TEMPERATURE:g}",
            f"PARAMETER top_p {TOP_P:g}",
            f"PARAMETER repeat_penalty {REPEAT_PENALTY:g}",
            "",
        ]
    )


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one docker/ollama command and capture its output."""
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _step(argv: list[str], steps: list[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
    """Log and run one step, recording argv and return code for the report."""
    _log(f"$ {' '.join(argv)}")
    try:
        proc = _run(argv)
    except OSError as exc:
        raise MsftError(f"cannot run {argv[0]}: {exc}") from exc
    steps.append(
        {
            "argv": list(argv),
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "").strip(),
        }
    )
    return proc


def _failure(argv: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    """Verbatim failure text: the command, its exit code and its own stderr."""
    detail = (proc.stderr or proc.stdout or "").strip()
    message = f"{' '.join(argv)} exited {proc.returncode}"
    return f"{message}: {detail}" if detail else message


def _listed(listing: str, model: str) -> bool:
    """Whether ``ollama list`` shows ``model``; ``ollama create tiny`` stores ``tiny:latest``.

    Match row names, not the whole table: a substring test over the blob accepted a different
    model whose name merely contains ours.
    """
    rows = [line.split() for line in listing.splitlines()[1:]]
    names = {row[0] for row in rows if row}
    if model in names:
        return True
    return ":" not in model and f"{model}:latest" in names


def _remote_modelfile(name: str) -> str:
    """A path-safe Modelfile name inside the container (model names contain ``/`` and ``:``)."""
    return "/tmp/Modelfile." + re.sub(r"[^A-Za-z0-9._-]", "_", name)


def run(
    gguf: str | Path,
    name: str,
    *,
    namespace: str | None = None,
    push: bool = False,
    container: str = DEFAULT_CONTAINER,
) -> dict[str, Any]:
    """Create ``name`` from ``gguf`` in ollama; returns a JSON-serialisable result."""
    path = Path(gguf)
    if not path.is_file():
        raise MsftError(f"no such GGUF: {path}")
    size = path.stat().st_size
    if size < MIN_GGUF_BYTES:
        raise MsftError(
            f"refusing {path}: {_size_label(size)} is not a finished quant "
            "(a Q4_K_S GGUF is at least 500 MB)"
        )
    if push and not namespace:
        raise MsftError(
            "--push needs --namespace: a bare name targets library/ and ollama refuses it"
        )

    use_container = bool(container)
    binary = "docker" if use_container else "ollama"
    if shutil.which(binary) is None:
        raise MsftError(
            f"{binary} is not on PATH; install it or pass --container '' for host ollama"
        )

    target = f"{namespace}/{name}" if namespace else name
    result: dict[str, Any] = {
        "ok": False,
        "gguf": str(path),
        "bytes": size,
        "size": _size_label(size),
        "name": name,
        "namespace": namespace,
        "target": target,
        "mode": "container" if use_container else "host",
        "container": container if use_container else None,
        "listed": None,
        "steps": [],
        "error": None,
    }
    steps: list[dict[str, Any]] = result["steps"]

    with tempfile.TemporaryDirectory(prefix="msft-export-") as tmp:
        local_modelfile = Path(tmp) / "Modelfile"
        source = f"/models/{path.name}" if use_container else str(path.resolve())
        local_modelfile.write_text(build_modelfile(source), encoding="utf-8")
        if use_container:
            remote_modelfile = _remote_modelfile(name)
            create = ["docker", "exec", container, "ollama", "create", name, "-f", remote_modelfile]
            sequence = [
                ["docker", "exec", container, "mkdir", "-p", "/models"],
                ["docker", "cp", str(path), f"{container}:/models/{path.name}"],
                ["docker", "cp", str(local_modelfile), f"{container}:{remote_modelfile}"],
                create,
            ]
        else:
            sequence = [["ollama", "create", name, "-f", str(local_modelfile)]]

        for argv in sequence:
            proc = _step(argv, steps)
            if proc.returncode != 0:
                result["error"] = _failure(argv, proc)
                _log(f"error: {result['error']}")
                return result

    prefix = ["docker", "exec", container, "ollama"] if use_container else ["ollama"]
    if namespace:
        argv = [*prefix, "cp", name, target]
        proc = _step(argv, steps)
        if proc.returncode != 0:
            result["error"] = _failure(argv, proc)
            _log(f"error: {result['error']}")
            return result
    if push:
        argv = [*prefix, "push", target]
        proc = _step(argv, steps)
        if proc.returncode != 0:
            # Keep ollama's wording verbatim: "not authorized to push to this namespace" is the
            # known failure and it names the cause better than we could.
            result["error"] = _failure(argv, proc)
            _log(f"push failed: {result['error']}")
            return result

    argv = [*prefix, "list"]
    proc = _step(argv, steps)
    if proc.returncode != 0:
        result["error"] = _failure(argv, proc)
        return result
    listing = proc.stdout or ""
    result["listed"] = _listed(listing, name)
    if not result["listed"]:
        result["error"] = f"{name} did not appear in `ollama list`"
        return result
    result["ok"] = True
    return result


def format_report(result: dict[str, Any]) -> str:
    """Step log, whether the model is listed, and the verdict."""
    mode = result["mode"]
    if result.get("container"):
        mode = f"{mode} ({result['container']})"
    lines = [
        f"export {result['gguf']} -> {result['name']}",
        f"  mode: {mode}",
        f"  size: {result['size']}",
        f"  target: {result['target']}",
    ]
    for step in result["steps"]:
        lines.append(f"  rc={step['returncode']} {' '.join(step['argv'])}")
    lines.append(f"  listed: {'yes' if result['listed'] else 'no'}")
    if result.get("error"):
        lines.append(f"  error: {result['error']}")
    lines.append(f"export {'ok' if result['ok'] else 'FAILED'}")
    return redact("\n".join(lines))
