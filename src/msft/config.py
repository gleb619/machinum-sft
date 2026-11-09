"""Configuration objects, loading, validation and redaction.

Shared foundation for every other module: nothing here imports torch or the network.
"""

from __future__ import annotations

import json
import re
from dataclasses import MISSING, asdict, dataclass, field, fields
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
        # Per-run, not a shared `out/merged`: two configs (0.8B and 4B) merged into one path
        # makes it possible to quantize the wrong weights under the right filename. That
        # happened on the first real run — out/model-4b.gguf was the 0.8B model.
        return self.output_path / f"merged-{self.name}"

    @property
    def dataset_path(self) -> Path:
        return self.output_path / "dataset"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_KNOWN_VALUES = {
    "dtype": {"bfloat16", "float16", "float32", "auto"},
    "scheduler": {
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "reduce_lr_on_plateau",
    },
    "optim": {
        "adamw_8bit",
        "paged_adamw_8bit",
        "adamw_torch",
        "adamw_torch_fused",
        "paged_adamw_32bit",
        "adafactor",
        "sgd",
    },
}
_QUANT_METHODS = {
    "q2_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "q4_0",
    "q4_1",
    "q4_k_s",
    "q4_k_m",
    "q5_0",
    "q5_1",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
    "f16",
    "bf16",
    "f32",
    "copy",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq2_m",
    "iq3_xxs",
    "iq3_xs",
    "iq3_s",
    "iq3_m",
    "iq4_nl",
    "iq4_xs",
}
_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def _coerce(section: str, key: str, annotation: str, value: Any) -> Any:
    """Check and coerce one YAML value to the dataclass field's annotated type."""
    where = f"config section '{section}' key '{key}'"
    if value is None:
        if annotation.endswith("| None"):
            return None
        raise MsftError(f"{where} must not be null")
    if annotation == "bool":
        # YAML `"false"` is a truthy string; an unquoted false is already a bool.
        if not isinstance(value, bool):
            raise MsftError(f"{where} must be true or false, not {value!r}")
        return value
    if annotation == "int":
        if isinstance(value, bool):
            raise MsftError(f"{where} must be an integer, not {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise MsftError(f"{where} must be an integer, not {value!r}")
    if annotation == "float":
        if isinstance(value, bool):
            raise MsftError(f"{where} must be a number, not {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise MsftError(f"{where} must be a number, not {value!r}")
    if annotation == "str":
        if not isinstance(value, str):
            raise MsftError(f"{where} must be a string, not {value!r}")
        return value
    if annotation == "list[str]":
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise MsftError(f"{where} must be a list of strings")
        items = list(value)
        if not all(isinstance(item, str) for item in items):
            raise MsftError(f"{where} must contain only strings")
        return items
    return value


def _section(data: dict[str, Any], key: str, cls: type) -> Any:
    raw = data.get(key)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise MsftError(f"config section '{key}' must be a mapping")
    allowed = {field.name: field for field in fields(cls)}
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise MsftError(f"unknown key(s) in '{key}': {', '.join(unknown)}")
    for spec in allowed.values():
        required = spec.default is MISSING and spec.default_factory is MISSING
        if required and spec.name not in raw:
            raise MsftError(f"config section '{key}' is missing required key '{spec.name}'")
    coerced = {
        name: _coerce(key, name, allowed[name].type, value) for name, value in raw.items()
    }
    try:
        return cls(**coerced)
    except TypeError as exc:
        raise MsftError(f"config section '{key}': {exc}") from exc


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
        base_dir=str(data.get("base_dir") or "."),
        model=_section(data, "model", ModelCfg),
        lora=_section(data, "lora", LoraCfg),
        train=_section(data, "train", TrainCfg),
        data=_section(data, "data", DataCfg),
        quantize=_section(data, "quantize", QuantCfg),
    )
    # A relative base_dir is resolved against the config file, so a config in
    # configs/ can say ".." and mean the repo root from any working directory.
    base = Path(cfg.base_dir)
    if not base.is_absolute():
        cfg.base_dir = str((p.parent / base).resolve())
    validate_config(cfg)
    return cfg


def _ignored_knob_reasons() -> dict[str, str]:
    """Fields the generated training script does not read, with the reason each is ignored."""
    from msft.train import IGNORED_KNOBS

    return IGNORED_KNOBS


def _knob_default(knob: str) -> Any:
    """Default value of a `section.field` knob, including default factories."""
    section, _, name = knob.partition(".")
    cls = {"model": ModelCfg, "lora": LoraCfg, "train": TrainCfg}.get(section)
    if cls is None:
        return None
    for spec in fields(cls):
        if spec.name == name:
            if spec.default is not MISSING:
                return spec.default
            if spec.default_factory is not MISSING:
                return spec.default_factory()
    return None


def _knob_is_changed(cfg: RunConfig, knob: str) -> bool:
    """True when a config sets `section.field` away from its dataclass default."""
    section, _, name = knob.partition(".")
    section_obj = getattr(cfg, section, None)
    value = getattr(section_obj, name, None)
    return value != _knob_default(knob)


def validate_config(cfg: RunConfig) -> list[str]:
    """Return warnings; raise MsftError on anything that cannot work."""
    warnings: list[str] = []
    if not _NAME_RE.fullmatch(cfg.name):
        raise MsftError(
            f"name '{cfg.name}' must match [A-Za-z0-9._-]+; it names directories and files"
        )
    if not cfg.model.id or "/" not in cfg.model.id:
        raise MsftError("model.id must look like 'org/name'")
    if cfg.model.dtype not in _KNOWN_VALUES["dtype"]:
        raise MsftError(
            f"model.dtype '{cfg.model.dtype}' is not one of "
            f"{', '.join(sorted(_KNOWN_VALUES['dtype']))}"
        )
    if cfg.lora.r <= 0:
        raise MsftError("lora.r must be positive")
    if cfg.lora.alpha <= 0:
        raise MsftError("lora.alpha must be positive")
    if cfg.train.max_seq_length < 256:
        raise MsftError("train.max_seq_length below 256 is not useful for chapters")
    for key, value in (
        ("train.epochs", cfg.train.epochs),
        ("train.grad_accum", cfg.train.grad_accum),
        ("train.per_device_batch", cfg.train.per_device_batch),
        ("train.lr", cfg.train.lr),
    ):
        if value <= 0:
            raise MsftError(f"{key} must be positive, not {value!r}")
    if cfg.train.scheduler not in _KNOWN_VALUES["scheduler"]:
        raise MsftError(
            f"train.scheduler '{cfg.train.scheduler}' is not one of "
            f"{', '.join(sorted(_KNOWN_VALUES['scheduler']))}"
        )
    if cfg.train.optim not in _KNOWN_VALUES["optim"]:
        raise MsftError(
            f"train.optim '{cfg.train.optim}' is not one of "
            f"{', '.join(sorted(_KNOWN_VALUES['optim']))}"
        )
    if cfg.quantize.method.lower() not in _QUANT_METHODS:
        raise MsftError(
            f"quantize.method '{cfg.quantize.method}' is not a known llama.cpp quantisation type"
        )
    if cfg.name not in Path(cfg.quantize.output).name:
        raise MsftError(
            f"quantize.output '{cfg.quantize.output}' does not contain the run name "
            f"'{cfg.name}'; two configs writing the same GGUF is how a 0.8B model once "
            "shipped as model-4b.gguf"
        )
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
    for knob, reason in _ignored_knob_reasons().items():
        if _knob_is_changed(cfg, knob):
            warnings.append(f"{knob} is ignored by the generated training script: {reason}")
    return warnings


# Explicit credential prefixes only: a generic 32+ character rule also masked checksums,
# so fetch/verify lost the sha256 they exist to report.
_SECRET = re.compile(
    r"(?i)(hf_[a-z0-9]{8,}|sk-[a-z0-9]{8,}|gh[pousr]_[a-z0-9]{8,}|xox[baprs]-[a-z0-9-]{8,})"
)
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
