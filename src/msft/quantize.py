"""GGUF export: locate llama.cpp, build the exact argv, and run conversion + quantisation.

Only stdlib here. The two tools are looked up from ``$LLAMA_CPP_DIR``, a list of common
checkout locations and ``PATH``; nothing is downloaded or built.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from msft import provenance
from msft.config import MsftError, RunConfig

CONVERT_NAME = "convert_hf_to_gguf.py"
QUANTIZE_NAMES = ("llama-quantize", "quantize")
IMATRIX_NAMES = ("llama-imatrix", "imatrix")
PARTIAL_SUFFIX = ".partial"
GGUF_MAGIC = b"GGUF"
CORPUS_ENV = "MSFT_IMATRIX_CORPUS"
LLAMA_DIR_ENV = "LLAMA_CPP_DIR"
BUILD_DIRS = ("", "build/bin", "build")

SEARCH_ROOTS: tuple[Path, ...] = (
    Path.home() / "llama.cpp",
    Path.home() / "src" / "llama.cpp",
    Path.home() / "code" / "llama.cpp",
    Path.home() / "git" / "llama.cpp",
    Path("/opt/llama.cpp"),
    Path("/usr/local/llama.cpp"),
    Path("/usr/src/llama.cpp"),
    Path.cwd() / "llama.cpp",
)

INSTALL_HINT = (
    "llama.cpp was not found. Either set LLAMA_CPP_DIR to an existing checkout, or build one:\n"
    "  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp\n"
    "  cmake -B ~/llama.cpp/build -S ~/llama.cpp && "
    "cmake --build ~/llama.cpp/build --config Release -j\n"
    "  export LLAMA_CPP_DIR=~/llama.cpp\n"
    "The python side also needs `uv pip install -r ~/llama.cpp/requirements.txt`."
)


def _roots() -> list[Path]:
    """`$LLAMA_CPP_DIR` first, then the well-known checkout locations."""
    roots: list[Path] = []
    env_dir = os.environ.get(LLAMA_DIR_ENV)
    if env_dir:
        roots.append(Path(env_dir).expanduser())
    roots.extend(SEARCH_ROOTS)
    return roots


def _find_convert() -> Path | None:
    """Path to `convert_hf_to_gguf.py`, or None when no checkout provides it."""
    for root in _roots():
        candidate = root / CONVERT_NAME
        if candidate.is_file():
            return candidate
    found = shutil.which(CONVERT_NAME)
    return Path(found) if found else None


def _find_tool(names: tuple[str, ...]) -> Path | None:
    """Path to the first executable called one of `names`, in a checkout or on PATH."""
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    for root in _roots():
        for sub in BUILD_DIRS:
            for name in names:
                candidate = root / sub / name
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
    return None


def output_gguf(cfg: RunConfig) -> Path:
    """The requested quantised GGUF, resolved against `base_dir` when relative."""
    out = Path(cfg.quantize.output)
    return out if out.is_absolute() else Path(cfg.base_dir) / out


def partial_gguf(cfg: RunConfig) -> Path:
    """Where `llama-quantize` writes before the finished file is atomically renamed."""
    out = output_gguf(cfg)
    return out.with_name(out.name + PARTIAL_SUFFIX)


def f16_gguf(cfg: RunConfig) -> Path:
    """Intermediate float16 GGUF written by `convert_hf_to_gguf.py`."""
    out = output_gguf(cfg)
    return out.with_name(f"{out.stem}-f16.gguf")


def imatrix_gguf(cfg: RunConfig) -> Path:
    """Importance matrix written by `llama-imatrix` for `--imatrix` calibration."""
    out = output_gguf(cfg)
    return out.with_name(f"{out.stem}.imatrix.gguf")


def _corpus() -> Path | None:
    """Calibration corpus from `$MSFT_IMATRIX_CORPUS`, expanded, or None."""
    raw = os.environ.get(CORPUS_ENV)
    return Path(raw).expanduser() if raw else None


def _commands(
    cfg: RunConfig, tools: dict[str, Path | None], corpus: Path | None
) -> list[list[str]]:
    """The argv list in run order: convert -> imatrix (optional) -> quantize."""
    convert, quantize_bin, imatrix_bin = tools["convert"], tools["quantize"], tools["imatrix"]
    f16 = f16_gguf(cfg)
    commands: list[list[str]] = []
    if convert is not None:
        commands.append(
            [
                sys.executable,
                str(convert),
                str(cfg.merged_path),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ]
        )
    if cfg.quantize.imatrix and imatrix_bin is not None and corpus is not None:
        commands.append(
            [
                str(imatrix_bin),
                "-m",
                str(f16),
                "-f",
                str(corpus),
                "-o",
                str(imatrix_gguf(cfg)),
            ]
        )
    if quantize_bin is not None:
        argv = [str(quantize_bin)]
        if cfg.quantize.imatrix and corpus is not None:
            argv += ["--imatrix", str(imatrix_gguf(cfg))]
        argv += [str(f16), str(partial_gguf(cfg)), cfg.quantize.method.upper()]
        commands.append(argv)
    return commands


def _missing(cfg: RunConfig, tools: dict[str, Path | None], corpus: Path | None) -> list[str]:
    """Human-readable names of everything that blocks a real run."""
    missing: list[str] = []
    if tools["convert"] is None:
        missing.append(CONVERT_NAME)
    if tools["quantize"] is None:
        missing.append("llama-quantize")
    if cfg.quantize.imatrix:
        if tools["imatrix"] is None:
            missing.append("llama-imatrix")
        if corpus is None:
            missing.append(CORPUS_ENV)
        elif not corpus.is_file():
            missing.append(f"calibration corpus {corpus}")
    return missing


def _hint(missing: list[str]) -> str:
    """Actionable remediation text for a missing-tools list."""
    lines = [f"quantisation needs: {', '.join(missing)}"]
    if CONVERT_NAME in missing or "llama-quantize" in missing:
        lines.append(INSTALL_HINT)
    if CORPUS_ENV in missing:
        lines.append(f"set {CORPUS_ENV} to a plain-text corpus for imatrix calibration")
    if any(item.startswith("calibration corpus") for item in missing):
        lines.append(f"{CORPUS_ENV} points at a file that does not exist")
    return "\n".join(lines)


def plan(cfg: RunConfig, *, require_tools: bool = True) -> dict[str, Any]:
    """argv-level plan for GGUF conversion and quantisation; raises when tools are missing."""
    tools = {
        "convert": _find_convert(),
        "quantize": _find_tool(QUANTIZE_NAMES),
        "imatrix": _find_tool(IMATRIX_NAMES) if cfg.quantize.imatrix else None,
    }
    corpus = _corpus() if cfg.quantize.imatrix else None
    missing = _missing(cfg, tools, corpus)
    if missing and require_tools:
        raise MsftError(_hint(missing))

    notes: list[str] = []
    merged_config = cfg.merged_path / "config.json"
    if not merged_config.is_file():
        notes.append(f"{merged_config} is missing; run `msft merge --execute` first")
    if cfg.quantize.imatrix:
        notes.append("imatrix calibration runs the f16 model once over the corpus; expect minutes")
    if not cfg.quantize.output.lower().endswith(".gguf"):
        notes.append(f"quantize.output does not end in .gguf: {cfg.quantize.output}")

    return {
        "stage": "quantize",
        "name": cfg.name,
        "model_dir": str(cfg.merged_path),
        "output": str(output_gguf(cfg)),
        "partial": str(partial_gguf(cfg)),
        "f16": str(f16_gguf(cfg)),
        "sidecar": str(provenance.sidecar_path(output_gguf(cfg))),
        "method": cfg.quantize.method.upper(),
        "imatrix": (
            {
                "enabled": True,
                "corpus": str(corpus) if corpus else None,
                "output": str(imatrix_gguf(cfg)),
            }
            if cfg.quantize.imatrix
            else {"enabled": False}
        ),
        "tools": {name: (str(path) if path else None) for name, path in tools.items()},
        "commands": _commands(cfg, tools, corpus),
        "missing": missing,
        "hint": _hint(missing) if missing else "",
        "ready": not missing,
        "notes": notes,
    }


def format_plan(plan_dict: dict[str, Any]) -> str:
    """Human-readable rendering of a quantize plan, one command per line."""
    lines = [f"quantize {plan_dict['name']} -> {plan_dict['output']} ({plan_dict['method']})"]
    lines.append(f"  from {plan_dict['model_dir']}")
    if plan_dict["missing"]:
        lines.append("  not ready: " + ", ".join(plan_dict["missing"]))
        if plan_dict.get("hint"):
            lines.append("  " + plan_dict["hint"].replace("\n", "\n  "))
    lines.append(f"  {len(plan_dict['commands'])} command(s):")
    lines += [f"    $ {' '.join(argv)}" for argv in plan_dict["commands"]]
    lines += [f"  note: {note}" for note in plan_dict["notes"]]
    return "\n".join(lines)


def _log(message: str) -> None:
    """Progress line on stderr, so a `--json` run keeps stdout to one object."""
    print(message, file=sys.stderr)


def run(cfg: RunConfig, *, execute: bool = False) -> dict[str, Any]:
    """Print the plan; with `execute=True` run each command in order and check the result.

    A dry run touches nothing and still returns the plan, including what is missing.
    """
    result = plan(cfg, require_tools=False)
    _log(format_plan(result))
    result["executed"] = False
    if not execute:
        return result

    if result["missing"]:
        raise MsftError(_hint(result["missing"]))
    if not (cfg.merged_path / "config.json").is_file():
        raise MsftError(
            f"{cfg.merged_path} holds no merged model; run `msft merge --execute` first"
        )

    out = output_gguf(cfg)
    partial = partial_gguf(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A previous run's artefact must never be exported as if this run produced it. The
    # sidecar goes with it: provenance must never outlive the file it describes.
    out.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    sidecar = provenance.sidecar_path(out)
    sidecar.unlink(missing_ok=True)
    merged_before = provenance.merged_snapshot(cfg.merged_path)
    for index, argv in enumerate(result["commands"], start=1):
        _log(f"[{index}/{len(result['commands'])}] $ {' '.join(argv)}")
        completed = subprocess.run(argv, stdout=sys.stderr, check=False)
        if completed.returncode != 0:
            raise MsftError(
                f"command {index} exited {completed.returncode}: {argv[0]}; read the log "
                "above and re-run `msft quantize --execute`"
            )

    if not partial.is_file():
        raise MsftError(
            f"llama-quantize reported success but {partial} is missing; "
            "check that the method name is supported by your llama.cpp build"
        )
    size = partial.stat().st_size
    with partial.open("rb") as handle:
        magic = handle.read(len(GGUF_MAGIC))
    if size <= 0 or magic != GGUF_MAGIC:
        raise MsftError(
            f"{partial} is not a valid GGUF ({size} bytes, magic {magic!r}); "
            "llama-quantize failed without a non-zero exit"
        )
    os.replace(partial, out)
    # The GGUF now exists; only then may a sidecar vouch for it. A merged directory that
    # changed mid-run means the f16 was converted from weights the snapshot no longer names.
    merged_after = provenance.merged_snapshot(cfg.merged_path)
    if merged_after != merged_before:
        raise MsftError("merged directory changed while quantizing")
    convert = result["tools"].get("convert")
    record = provenance.build_record(
        config_name=cfg.name,
        model_id=cfg.model.id,
        revision=cfg.model.revision,
        merged=merged_before,
        f16=f16_gguf(cfg),
        gguf=out,
        method=cfg.quantize.method.upper(),
        imatrix=bool(cfg.quantize.imatrix),
        llama_cpp=provenance.llama_cpp_info(str(Path(convert).parent) if convert else None),
    )
    provenance.write(out, record)
    result["executed"] = True
    return result
