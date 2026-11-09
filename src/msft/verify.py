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

from msft import provenance
from msft.config import RunConfig, redact
from msft.quantize import f16_gguf, output_gguf

RATIO_MIN = 0.25
RATIO_MAX = 0.45
MERGED_BAND_MIN = 0.15
MERGED_BAND_MAX = 0.65
GGUF_MAGIC = b"GGUF"
CONFIG_KEYS = ("model_type", "hidden_size", "num_hidden_layers", "vocab_size")


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


def base_config(cfg: RunConfig) -> tuple[dict[str, Any] | None, str]:
    """The base model's ``config.json`` as a dict, and where it came from."""
    cached = _cached_base_config(cfg.model.id)
    if cached is not None:
        try:
            data = json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"cannot read cached base config {cached}: {exc}"
        if isinstance(data, dict):
            return data, f"HF cache {cached}"
        return None, f"cached base config {cached} is not a mapping"
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
    except Exception as exc:  # noqa: BLE001 - any failure means the config is unknown
        return None, f"no cached config.json for {cfg.model.id}: {exc}"
    if not getattr(config, "model_type", ""):
        return None, f"{cfg.model.id} config has no model_type"
    return dict(config.to_dict()), "transformers AutoConfig"


def base_model_type(cfg: RunConfig) -> tuple[str | None, str]:
    """The base model's ``model_type`` from the local cache; returns (type, where it came from)."""
    data, source = base_config(cfg)
    if data is None:
        return None, source
    model_type = data.get("model_type")
    if not model_type:
        return None, f"{cfg.model.id} config has no model_type"
    return str(model_type), source


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
    """Compare the merged config with the base: model_type and the size-bearing fields.

    ``model_type`` alone is identical for every size in a family — that is why a 0.8B model
    once passed as a 4B. ``hidden_size``, ``num_hidden_layers`` and ``vocab_size`` are not.
    """
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
    if not isinstance(data, dict):
        return {
            "name": "model_type",
            "passed": False,
            "detail": f"{config_path} does not hold a JSON object",
        }
    base, source = base_config(cfg)
    if base is None:
        return {"name": "model_type", "passed": False, "detail": source}

    problems: list[str] = []
    compared: list[str] = []
    for key in CONFIG_KEYS:
        expected = base.get(key)
        actual = data.get(key)
        if expected is None:
            continue
        compared.append(key)
        if actual is None:
            problems.append(f"merged config has no {key}")
        elif actual != expected:
            problems.append(f"{key} {actual} != base {expected}")
    if problems:
        return {
            "name": "model_type",
            "passed": False,
            "detail": (
                f"{'; '.join(problems)} ({source}); this is another model under this "
                "artifact's filename"
            ),
        }
    checked = ", ".join(compared) or "no size fields present in the base config"
    return {
        "name": "model_type",
        "passed": True,
        "detail": f"merged matches base for {checked} ({source})",
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


def _gguf_magic(path: Path) -> bytes:
    """The first four bytes of a file, or b'' when it cannot be read."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(GGUF_MAGIC))
    except OSError:
        return b""


def _check_gguf(cfg: RunConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """One check plus the size table row for the quantised GGUF and its f16 source."""
    gguf = output_gguf(cfg)
    f16 = f16_gguf(cfg)
    gguf_bytes = _file_size(gguf)
    f16_bytes = _file_size(f16)
    merged_bytes = _dir_size(cfg.merged_path) if cfg.merged_path.is_dir() else 0
    ratio = gguf_bytes / f16_bytes if f16_bytes else None
    merged_ratio = gguf_bytes / merged_bytes if merged_bytes else None

    problems: list[str] = []
    if not gguf.is_file():
        problems.append(f"{gguf} is missing; run `msft quantize --execute`")
    else:
        magic = _gguf_magic(gguf)
        if magic != GGUF_MAGIC:
            problems.append(f"{gguf} does not start with the GGUF magic ({magic!r})")
        if merged_ratio is not None and not MERGED_BAND_MIN <= merged_ratio <= MERGED_BAND_MAX:
            problems.append(
                f"{gguf.name} is {merged_ratio:.1%} of the merged weights "
                f"({_size_label(merged_bytes)}); a Q4 lands between "
                f"{MERGED_BAND_MIN:.0%} and {MERGED_BAND_MAX:.0%}"
            )
    if ratio is not None and not RATIO_MIN <= ratio <= RATIO_MAX:
        problems.append(
            f"{gguf.name} is {ratio:.1%} of {f16.name}; Q4_K_S lands between 25 % and 45 %"
        )
    if problems:
        detail = "; ".join(problems)
    elif merged_ratio is not None:
        detail = (
            f"{_size_label(gguf_bytes)}, {merged_ratio:.1%} of the merged weights"
            + (f", {ratio:.1%} of {f16.name}" if ratio is not None else "")
            + ("" if ratio is not None else "; f16 source absent, ratio not checked")
        )
    elif ratio is not None:
        detail = f"{_size_label(gguf_bytes)}, {ratio:.1%} of {f16.name}"
    else:
        detail = f"{_size_label(gguf_bytes)}; merged weights absent, size band not checked"

    sizes = {
        "gguf": {"path": str(gguf), "bytes": gguf_bytes, "label": _size_label(gguf_bytes)},
        "f16": {"path": str(f16), "bytes": f16_bytes, "label": _size_label(f16_bytes)},
        "ratio": round(ratio, 4) if ratio is not None else None,
        "merged_ratio": round(merged_ratio, 4) if merged_ratio is not None else None,
    }
    return {"name": "gguf", "passed": not problems, "detail": detail}, sizes


def _check_chat_template(cfg: RunConfig) -> dict[str, Any]:
    """The GGUF must carry the tokenizer's own chat template; check the merged files.

    Reading GGUF metadata is more code than this earns, and the merged directory is the
    source the converter copies from. An empty or absent template makes ollama fall back to
    a hand-written one — the failure rule 4 in AGENTS.md exists to prevent.
    """
    tokenizer_config = cfg.merged_path / "tokenizer_config.json"
    if tokenizer_config.is_file():
        try:
            data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "name": "chat_template",
                "passed": False,
                "detail": f"cannot read {tokenizer_config}: {exc}",
            }
        template = data.get("chat_template") if isinstance(data, dict) else None
        if isinstance(template, str) and template.strip():
            return {
                "name": "chat_template",
                "passed": True,
                "detail": "tokenizer_config.json carries a non-empty chat_template",
            }
        if isinstance(template, list) and template:
            return {
                "name": "chat_template",
                "passed": True,
                "detail": "tokenizer_config.json carries a chat_template list",
            }
    for name in ("chat_template.jinja", "chat_template.txt"):
        candidate = cfg.merged_path / name
        try:
            if candidate.is_file() and candidate.read_text(encoding="utf-8").strip():
                return {
                    "name": "chat_template",
                    "passed": True,
                    "detail": f"{name} is non-empty",
                }
        except OSError:
            continue
    return {
        "name": "chat_template",
        "passed": False,
        "detail": (
            f"{cfg.merged_path} has no non-empty chat_template; the GGUF would carry none "
            "and ollama would need a hand-written one"
        ),
    }


def _check_provenance(
    cfg: RunConfig, *, require: bool = False, deep: bool = False
) -> dict[str, Any]:
    """Check the sidecar next to the GGUF against the merged model and the GGUF itself."""
    gguf = output_gguf(cfg)
    sidecar = provenance.sidecar_path(gguf)
    record, reason = provenance.load(gguf)
    if record is None and reason == "absent":
        detail = (
            f"no provenance sidecar next to {gguf.name}; this GGUF predates sidecars or was "
            "not produced by msft quantize — provenance not checked"
        )
        if require:
            return {
                "name": "provenance",
                "passed": False,
                "warning": False,
                "detail": f"{detail} (--require-provenance)",
            }
        return {"name": "provenance", "passed": True, "warning": True, "detail": detail}
    if record is None:
        joiner = " is " if reason.startswith("corrupt") else ": "
        return {
            "name": "provenance",
            "passed": False,
            "warning": False,
            "detail": f"provenance sidecar {sidecar.name}{joiner}{reason}",
        }
    problems = provenance.compare(
        record,
        gguf=gguf,
        merged=cfg.merged_path,
        f16=f16_gguf(cfg),
        cfg_name=cfg.name,
        method=cfg.quantize.method.upper(),
        deep=deep,
    )
    if problems:
        return {
            "name": "provenance",
            "passed": False,
            "warning": False,
            "detail": "provenance: " + "; ".join(problems),
        }
    return {
        "name": "provenance",
        "passed": True,
        "warning": False,
        "detail": f"sidecar {sidecar.name} matches the artifact and the merged model",
    }


def run(
    cfg: RunConfig, *, require_provenance: bool = False, deep: bool = False
) -> dict[str, Any]:
    """Check merged config, tokenizer, chat template, GGUF and provenance; JSON verdict."""
    gguf_check, gguf_sizes = _check_gguf(cfg)
    checks = [
        _check_merged(cfg),
        _check_model_type(cfg),
        _check_tokenizer(cfg),
        _check_chat_template(cfg),
        gguf_check,
        _check_provenance(cfg, require=require_provenance, deep=deep),
    ]
    failures = [check["name"] for check in checks if not check["passed"]]
    warnings = [check["detail"] for check in checks if check.get("warning")]
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
        label = "WARN" if check.get("warning") else ("PASS" if check["passed"] else "FAIL")
        _log(f"{label} {check['name']}: {check['detail']}")
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
        "warnings": warnings,
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
        label = "WARN" if check.get("warning") else ("PASS" if check["passed"] else "FAIL")
        lines.append(f"  {label} {check['name']}: {check['detail']}")
    failed = len(result["failures"])
    lines.append(
        f"artifacts {'ok' if result['ok'] else 'FAILED'}"
        f" ({len(result['checks'])} checks, {failed} failed)"
    )
    return redact("\n".join(lines))
