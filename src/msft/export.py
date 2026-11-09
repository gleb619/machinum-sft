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

from msft.config import MsftError, RunConfig, redact

MIN_GGUF_BYTES = 500_000_000
DEFAULT_CONTAINER = "ollama"
NUM_CTX = 8192
TEMPERATURE = 0
TOP_P = 0.95
REPEAT_PENALTY = 1.05
TAIL_LINES = 50

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_TAG_RE = re.compile(r"[A-Za-z0-9._-]+")


def _clean(text: str, limit: int = TAIL_LINES) -> str:
    """Strip ANSI colour and keep the last ``limit`` lines; docker logs are long and noisy."""
    plain = _ANSI_RE.sub("", text or "")
    lines = [line.rstrip() for line in plain.splitlines() if line.strip()]
    if len(lines) > limit:
        dropped = len(lines) - limit
        return "\n".join([f"... {dropped} earlier line(s) omitted", *lines[-limit:]])
    return "\n".join(lines)


def _check_name(value: str, what: str) -> str:
    """Reject names that cannot safely be passed to docker/ollama as one argument."""
    base, sep, tag = value.partition(":")
    if not _NAME_RE.fullmatch(base) or (sep and not _TAG_RE.fullmatch(tag)):
        raise MsftError(
            f"invalid {what} {value!r}: use letters, digits, '.', '_' or '-' "
            "(a single ':' separates an optional tag)"
        )
    return value


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
            "stderr": _clean(proc.stderr or ""),
        }
    )
    return proc


def _failure(argv: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    """Verbatim failure text: the command, its exit code and the tail of its own output."""
    detail = _clean(proc.stderr or proc.stdout or "")
    message = f"{' '.join(argv)} exited {proc.returncode}"
    return redact(f"{message}: {detail}" if detail else message)


def _listed(listing: str, target: str) -> bool:
    """Whether ``ollama list`` shows ``target``; ``ollama create tiny`` stores ``tiny:latest``.

    Match row names, not the whole table: a substring test over the blob accepted a different
    model whose name merely contains ours. When a namespace is used, the namespaced target is
    what must appear, not the bare name it was copied from.
    """
    rows = [line.split() for line in listing.splitlines()[1:]]
    names = {row[0] for row in rows if row}
    if target in names:
        return True
    return ":" not in target and f"{target}:latest" in names


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
    config: RunConfig | None = None,
) -> dict[str, Any]:
    """Create ``name`` from ``gguf`` in ollama; returns a JSON-serialisable result.

    When ``config`` is given, `msft verify` must pass first: an artifact that fails its own
    checks must never reach ``ollama create`` and look healthy in ``ollama list``.
    """
    path = Path(gguf)
    if not path.is_file():
        raise MsftError(f"no such GGUF: {path}")
    name = _check_name(name, "model name")
    if namespace is not None:
        namespace = _check_name(namespace, "namespace")
    if push and not namespace:
        raise MsftError(
            "--push needs --namespace: a bare name targets library/ and ollama refuses it"
        )
    size = path.stat().st_size
    # With a config, verify's band against the merged weights replaces this fixed floor: it is
    # the same rule for a 0.8B smoke run and a 12B release.
    if config is None and size < MIN_GGUF_BYTES:
        raise MsftError(
            f"refusing {path}: {_size_label(size)} is not a finished quant "
            "(a Q4_K_S GGUF is at least 500 MB)"
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
        "verify": None,
        "listed": None,
        "steps": [],
        "error": None,
    }
    steps: list[dict[str, Any]] = result["steps"]

    if config is not None:
        from msft import verify as verify_mod

        verification = verify_mod.run(config)
        result["verify"] = {
            "ok": verification["ok"],
            "failures": verification["failures"],
            "checks": verification["checks"],
        }
        if not verification["ok"]:
            failed = ", ".join(verification["failures"])
            result["error"] = (
                f"verify failed ({failed}); refusing to export {path.name} — "
                "run `msft verify` with the same --config for the details"
            )
            _log(f"error: {result['error']}")
            return result

    if use_container:
        proc = _step(
            ["docker", "inspect", "-f", "{{.State.Running}}", container], steps
        )
        if proc.returncode != 0:
            result["error"] = _failure(
                ["docker", "inspect", "-f", "{{.State.Running}}", container], proc
            )
            _log(f"error: {result['error']}")
            return result
        if (proc.stdout or "").strip() != "true":
            result["error"] = (
                f"container {container!r} is not running; start it before exporting"
            )
            _log(f"error: {result['error']}")
            return result

    with tempfile.TemporaryDirectory(prefix="msft-export-") as tmp:
        local_modelfile = Path(tmp) / "Modelfile"
        source = f"/models/{path.name}" if use_container else str(path.resolve())
        local_modelfile.write_text(build_modelfile(source), encoding="utf-8")
        copied: list[str] = []
        if use_container:
            remote_modelfile = _remote_modelfile(target)
            create = [
                "docker",
                "exec",
                container,
                "ollama",
                "create",
                name,
                "-f",
                remote_modelfile,
            ]
            sequence = [
                ["docker", "exec", container, "mkdir", "-p", "/models"],
                ["docker", "cp", str(path), f"{container}:/models/{path.name}"],
                ["docker", "cp", str(local_modelfile), f"{container}:{remote_modelfile}"],
                create,
            ]
            copied = [f"/models/{path.name}", remote_modelfile]
        else:
            sequence = [["ollama", "create", name, "-f", str(local_modelfile)]]

        try:
            for argv in sequence:
                proc = _step(argv, steps)
                if proc.returncode != 0:
                    result["error"] = _failure(argv, proc)
                    _log(f"error: {result['error']}")
                    return result
        finally:
            if copied:
                # A stale copy in the container looks current and is not; always remove it.
                try:
                    _step(["docker", "exec", container, "rm", "-f", *copied], steps)
                except OSError as exc:  # cleanup must not mask the real failure
                    _log(f"warning: could not remove the container copies: {exc}")

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
    result["listed"] = _listed(listing, target)
    if not result["listed"]:
        result["error"] = f"{target} did not appear in `ollama list`"
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
    verification = result.get("verify")
    if verification is not None:
        verdict = "PASS" if verification["ok"] else f"FAIL ({', '.join(verification['failures'])})"
        lines.append(f"  verify: {verdict}")
    lines.append(f"  listed: {'yes' if result['listed'] else 'no'}")
    if result.get("error"):
        lines.append(f"  error: {result['error']}")
    lines.append(f"export {'ok' if result['ok'] else 'FAILED'}")
    return redact("\n".join(lines))
