"""Configuration objects, loading, validation and redaction.

Shared foundation for every other module: nothing here imports torch or the network.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class MsftError(RuntimeError):
    """Any user-facing failure; the CLI turns it into `error: <msg>` and exit code 2."""


@dataclass
class ModelCfg:
    id: str
    revision: str | None = None
    load_in_4bit: bool = True
    dtype: str = "bfloat16"


@dataclass
class LoraCfg:
    r: int = 32
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))
    modules_to_save: list[str] = field(default_factory=lambda: ["lm_head"])
    use_rslora: bool = False


@dataclass
class TrainCfg:
    max_seq_length: int = 4096
    per_device_batch: int = 1
    grad_accum: int = 32
    lr: float = 5e-5
    epochs: float = 2.0
    warmup_steps: int = 8
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    optim: str = "adamw_8bit"
    seed: int = 42
    packing: bool = False
    completion_only_loss: bool = True
    save_steps: int = 20
    eval_steps: int = 20


@dataclass
class DataCfg:
    input: str = "data/raw"
    val_ratio: float = 0.03
    dedupe: bool = True
    min_chars: int = 50
    max_chars: int = 8000
    max_tokens: int | None = None


@dataclass
class QuantCfg:
    method: str = "q4_k_m"
    imatrix: bool = False
    output: str = "out/model.gguf"


@dataclass
class RunConfig:
    name: str = "run"
    model: ModelCfg = field(default_factory=lambda: ModelCfg(id="Qwen/Qwen3.5-9B"))
    lora: LoraCfg = field(default_factory=LoraCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    quantize: QuantCfg = field(default_factory=QuantCfg)
    output_dir: str = "out"
    base_dir: str = "."

    # --- derived helpers -------------------------------------------------
    @property
    def output_path(self) -> Path:
        return Path(self.base_dir) / self.output_dir

    @property
    def adapter_path(self) -> Path:
        return self.output_path / "adapter"

    @property
    def merged_path(self) -> Path:
        return self.output_path / "merged"

    @property
    def dataset_path(self) -> Path:
        return self.output_path / "dataset"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(data: dict[str, Any], key: str, cls: type) -> Any:
    raw = data.get(key) or {}
    if not isinstance(raw, dict):
        raise MsftError(f"config section '{key}' must be a mapping")
    allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise MsftError(f"unknown key(s) in '{key}': {', '.join(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML (or JSON) run config."""
    p = Path(path)
    if not p.exists():
        raise MsftError(f"config not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:  # pragma: no cover - message shape
        raise MsftError(f"cannot parse {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise MsftError(f"{p} must contain a mapping at the top level")
    top = {"name", "output_dir", "base_dir"}
    unknown = sorted(set(data) - top - {"model", "lora", "train", "data", "quantize"})
    if unknown:
        raise MsftError(f"unknown top-level key(s): {', '.join(unknown)}")
    cfg = RunConfig(
        name=str(data.get("name") or p.stem),
        output_dir=str(data.get("output_dir") or "out"),
        base_dir=str(data.get("base_dir") or p.parent),
        model=_section(data, "model", ModelCfg),
        lora=_section(data, "lora", LoraCfg),
        train=_section(data, "train", TrainCfg),
        data=_section(data, "data", DataCfg),
        quantize=_section(data, "quantize", QuantCfg),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: RunConfig) -> list[str]:
    """Return warnings; raise MsftError on anything that cannot work."""
    warnings: list[str] = []
    if not cfg.model.id or "/" not in cfg.model.id:
        raise MsftError("model.id must look like 'org/name'")
    if cfg.lora.r <= 0:
        raise MsftError("lora.r must be positive")
    if cfg.lora.alpha <= 0:
        raise MsftError("lora.alpha must be positive")
    if cfg.train.max_seq_length < 256:
        raise MsftError("train.max_seq_length below 256 is not useful for chapters")
    if not 0.0 < cfg.data.val_ratio < 0.5:
        raise MsftError("data.val_ratio must be between 0 and 0.5")
    if cfg.train.per_device_batch * cfg.train.grad_accum < 8:
        warnings.append(
            "effective batch "
            f"{cfg.train.per_device_batch * cfg.train.grad_accum} is small; "
            "the published Russian runs use 16-128"
        )
    ratio = cfg.lora.alpha / cfg.lora.r
    if ratio >= 2.0 and cfg.train.lr > 1e-4:
        warnings.append(
            f"alpha/r={ratio:.2f} with lr={cfg.train.lr:g} is an aggressive adapter step; "
            "published runs pair alpha/r=2 with lr<=2e-4 and alpha/r=0.5 with lr~5e-5"
        )
    if cfg.train.epochs > 3:
        warnings.append("more than 3 epochs usually overfits a small literary set")
    if cfg.quantize.imatrix:
        warnings.append("imatrix calibration needs a corpus file and a llama.cpp build")
    return warnings


_SECRET = re.compile(r"(?i)(hf_[a-z0-9]{8,}|sk-[a-z0-9]{8,}|gh[pousr]_[a-z0-9]{8,}|[A-Za-z0-9_-]{32,})")
_HOST = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_HOME = re.compile(r"/(?:home|Users)/[^/\s]+")


def redact(value: str) -> str:
    """Mask secrets, IP addresses and home directories in anything user-visible."""
    out = _SECRET.sub("<redacted>", value)
    out = _HOST.sub("<host>", out)
    out = _HOME.sub("~", out)
    return out


def dump_config(cfg: RunConfig) -> str:
    """Human-readable config dump, redacted."""
    return redact(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
