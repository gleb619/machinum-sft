"""Post-run artifact verification: `msft verify`.

A job that exits 0 is not proof. The first run shipped a 516 MB ``model-4b.gguf`` that was really
the 0.8B model, because both configs merged into one directory. This module checks the things
that would have caught it: the merged model's ``model_type`` matches the base model, its
tokenizer loads, and the GGUF is neither a fragment nor a differently-sized quant.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from msft.config import RunConfig, redact
from msft.quantize import f16_gguf, output_gguf

MIN_GGUF_BYTES = 500_000_000
RATIO_MIN = 0.25
RATIO_MAX = 0.45


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


def _file_size(path: Path) -> int:
    """Bytes of a regular file, or 0 when it is missing or unreadable."""
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _dir_size(path: Path) -> int:
    """Bytes of regular files under a directory; symlinks are counted once."""
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


def _hub_roots() -> list[Path]:
    """Candidate Hugging Face hub cache roots, most explicit first."""
    roots: list[Path] = []
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        roots.append(Path(explicit))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _cached_base_config(model_id: str) -> Path | None:
    """Newest cached ``config.json`` for ``model_id``, or None when it was never downloaded."""
    folder = f"models--{model_id.replace('/', '--')}"
    candidates: list[Path] = []
    for root in _hub_roots():
        candidates.extend((root / folder / "snapshots").glob("*/config.json"))
    present = [path for path in candidates if path.is_file()]
    if not present:
        return None
    return max(present, key=lambda path: path.stat().st_mtime)


def base_model_type(cfg: RunConfig) -> tuple[str | None, str]:
    """The base model's ``model_type`` from the local cache; returns (type, where it came from)."""
    cached = _cached_base_config(cfg.model.id)
    if cached is not None:
        try:
            data = json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"cannot read cached base config {cached}: {exc}"
        if isinstance(data, dict) and data.get("model_type"):
            return str(data["model_type"]), f"HF cache {cached}"
    try:
        from transformers import AutoConfig
    except ImportError:
        return None, (
            f"no cached config.json for {cfg.model.id} and transformers is not installed; "
            "run `msft fetch --config …` on the box first"
        )
    kwargs: dict[str, Any] = {"local_files_only": True}
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    try:
        config = AutoConfig.from_pretrained(cfg.model.id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any failure means the type is unknown
        return None, f"no cached config.json for {cfg.model.id}: {exc}"
    model_type = getattr(config, "model_type", "") or ""
    if not model_type:
        return None, f"{cfg.model.id} config has no model_type"
    return str(model_type), "transformers AutoConfig"


def tokenizer_loads(merged: Path) -> tuple[bool, str]:
    """Try to load the merged tokenizer; transformers is imported lazily."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return False, "transformers is not installed; uv pip install -e '.[train]'"
    try:
        AutoTokenizer.from_pretrained(str(merged))
    except Exception as exc:  # noqa: BLE001 - any load failure breaks the GGUF converter
        return False, f"AutoTokenizer.from_pretrained failed: {exc}"
    return True, "tokenizer loads from the merged directory"


def _check_merged(cfg: RunConfig) -> dict[str, Any]:
    exists = cfg.merged_path.is_dir()
    detail = (
        f"{cfg.merged_path} exists"
        if exists
        else f"{cfg.merged_path} is missing; run `msft merge --execute`"
    )
    return {"name": "merged_dir", "passed": exists, "detail": detail}


def _check_model_type(cfg: RunConfig) -> dict[str, Any]:
    config_path = cfg.merged_path / "config.json"
    if not config_path.is_file():
        return {
            "name": "model_type",
            "passed": False,
            "detail": f"{config_path} is missing; the merged directory has no model config",
        }
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "name": "model_type",
            "passed": False,
            "detail": f"cannot read {config_path}: {exc}",
        }
    merged_type = data.get("model_type") if isinstance(data, dict) else None
    if not merged_type:
        return {
            "name": "model_type",
            "passed": False,
            "detail": f"{config_path} has no model_type",
        }
    base_type, source = base_model_type(cfg)
    if not base_type:
        return {"name": "model_type", "passed": False, "detail": source}
    if str(merged_type) == base_type:
        return {
            "name": "model_type",
            "passed": True,
            "detail": f"merged {merged_type} equals base {base_type} ({source})",
        }
    return {
        "name": "model_type",
        "passed": False,
        "detail": (
            f"merged {merged_type} != base {base_type} ({source}); this is another model "
            "under this artifact's filename"
        ),
    }


def _check_tokenizer(cfg: RunConfig) -> dict[str, Any]:
    if not cfg.merged_path.is_dir():
        return {
            "name": "tokenizer",
            "passed": False,
            "detail": "merged directory is missing; nothing to load",
        }
    passed, detail = tokenizer_loads(cfg.merged_path)
    return {"name": "tokenizer", "passed": passed, "detail": detail}


def _check_gguf(cfg: RunConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """One check plus the size table row for the quantised GGUF and its f16 source."""
    gguf = output_gguf(cfg)
    f16 = f16_gguf(cfg)
    gguf_bytes = _file_size(gguf)
    f16_bytes = _file_size(f16)
    ratio = gguf_bytes / f16_bytes if f16_bytes else None

    problems: list[str] = []
    if not gguf.is_file():
        problems.append(f"{gguf} is missing; run `msft quantize --execute`")
    elif gguf_bytes <= MIN_GGUF_BYTES:
        problems.append(
            f"{gguf} is only {_size_label(gguf_bytes)}; a finished Q4 is at least 500 MB"
        )
    if ratio is not None and not RATIO_MIN <= ratio <= RATIO_MAX:
        problems.append(
            f"{gguf.name} is {ratio:.1%} of {f16.name}; Q4_K_S lands between 25 % and 45 %"
        )
    if problems:
        detail = "; ".join(problems)
    elif ratio is None:
        detail = f"{_size_label(gguf_bytes)}; f16 source absent, ratio not checked"
    else:
        detail = f"{_size_label(gguf_bytes)}, {ratio:.1%} of {f16.name}"

    sizes = {
        "gguf": {"path": str(gguf), "bytes": gguf_bytes, "label": _size_label(gguf_bytes)},
        "f16": {"path": str(f16), "bytes": f16_bytes, "label": _size_label(f16_bytes)},
        "ratio": round(ratio, 4) if ratio is not None else None,
    }
    return {"name": "gguf", "passed": not problems, "detail": detail}, sizes


def run(cfg: RunConfig) -> dict[str, Any]:
    """Check merged config, tokenizer and GGUF; returns a JSON-serialisable verdict."""
    gguf_check, gguf_sizes = _check_gguf(cfg)
    checks = [
        _check_merged(cfg),
        _check_model_type(cfg),
        _check_tokenizer(cfg),
        gguf_check,
    ]
    failures = [check["name"] for check in checks if not check["passed"]]
    merged_bytes = _dir_size(cfg.merged_path) if cfg.merged_path.is_dir() else 0
    sizes = {
        "merged": {
            "path": str(cfg.merged_path),
            "bytes": merged_bytes,
            "label": _size_label(merged_bytes),
        },
        **gguf_sizes,
    }
    for check in checks:
        _log(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}")
    return {
        "ok": not failures,
        "name": cfg.name,
        "model": cfg.model.id,
        "merged": str(cfg.merged_path),
        "gguf": str(output_gguf(cfg)),
        "f16": str(f16_gguf(cfg)),
        "checks": checks,
        "sizes": sizes,
        "failures": failures,
    }


def format_report(result: dict[str, Any]) -> str:
    """Size table plus one PASS/FAIL line per check."""
    sizes = result["sizes"]
    lines = [
        f"verify {result['name']} ({result['model']})",
        f"  {'merged':<6} {sizes['merged']['label']:>9}  {sizes['merged']['path']}",
        f"  {'gguf':<6} {sizes['gguf']['label']:>9}  {sizes['gguf']['path']}",
    ]
    if sizes["f16"]["bytes"]:
        ratio = sizes["ratio"]
        suffix = f"  (gguf = {ratio:.1%} of f16)" if ratio is not None else ""
        lines.append(f"  {'f16':<6} {sizes['f16']['label']:>9}  {sizes['f16']['path']}{suffix}")
    for check in result["checks"]:
        lines.append(
            f"  {'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}"
        )
    failed = len(result["failures"])
    lines.append(
        f"artifacts {'ok' if result['ok'] else 'FAILED'}"
        f" ({len(result['checks'])} checks, {failed} failed)"
    )
    return redact("\n".join(lines))
