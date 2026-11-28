"""LoRA delta merge: plan, generated standalone script, and execution.

Two arms train from the same base with the same ``r``/``alpha``/``target_modules``. peft's
delta for a linear layer is ``delta = s * B @ A`` with ``s = lora_alpha / r``; the union this
module builds is the *mean* of the two deltas, and a rank-``r`` factor pair can express that
mean exactly at rank ``2r``::

    A' = [A_a ; A_b]        (2r x d_in)
    B' = [B_a | B_b]        (d_out x 2r)
    r' = 2r, alpha' = alpha  =>  s' = alpha/(2r) = s/2
    delta' = s' B' A' = (delta_a + delta_b) / 2      exact, nothing averaged

So the merge is a concatenation plus an alpha rewrite: no matmul, no SVD, no TIES. The math
is a pure function over nested lists (``Factors``/``concatenate``) so CI checks it without
torch; the tensor step is a generated standalone script, the same convention as `merge.py`.

``--weights sum`` is the one ablation: same concatenation, ``alpha' = 2*alpha``, i.e.
``delta = delta_a + delta_b``. It answers "was the mean just too weak?".
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from msft.config import MsftError

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"
META_NAME = "merge_meta.json"
SCRIPT_NAME = "merge_script.py"
MODES = ("mean", "sum")

#: Keys carried into the merged `adapter_config.json`; peft-valid names only. A key peft
#: wrote but this list does not name is reported instead of copied blind.
CONFIG_KEYS = (
    "alpha_pattern",
    "base_model_name_or_path",
    "bias",
    "fan_in_fan_out",
    "inference_mode",
    "init_lora_weights",
    "lora_alpha",
    "lora_bias",
    "lora_dropout",
    "modules_to_save",
    "peft_type",
    "r",
    "rank_pattern",
    "revision",
    "target_modules",
    "task_type",
    "use_dora",
    "use_rslora",
)

#: Fallback probe when `--cases` is not given: English prose, so the reply is not empty.
DEFAULT_PROMPT = (
    "The innkeeper wiped the same tankard for the third time and pretended not to watch "
    "the door."
)

_A_MARKERS = (
    (".lora_A.weight", "A"),
    (".lora_A.default.weight", "A"),
    (".lora_B.weight", "B"),
    (".lora_B.default.weight", "B"),
)


# ------------------------------------------------------------------- pure delta math


@dataclass(frozen=True)
class Factors:
    """One LoRA factor pair: ``a`` is r x d_in, ``b`` is d_out x r, ``alpha`` the peft alpha."""

    a: list[list[float]]
    b: list[list[float]]
    r: int
    alpha: int

    @property
    def scale(self) -> float:
        """peft's LoRA scaling for this pair (``use_rslora`` is refused before we get here)."""
        return self.alpha / self.r

    def delta(self) -> list[list[float]]:
        """``s * B @ A`` as nested lists, in float64 arithmetic."""
        rows = len(self.b)
        inner = self.r
        cols = len(self.a[0]) if self.a else 0
        out = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                out[i][j] = self.scale * sum(
                    float(self.b[i][k]) * float(self.a[k][j]) for k in range(inner)
                )
        return out


def _check_dims(factors: Factors) -> None:
    """Refuse a factor pair whose matrices do not line up with its declared rank."""
    if factors.r <= 0:
        raise MsftError(f"LoRA rank must be positive, not {factors.r}")
    if len(factors.a) != factors.r:
        raise MsftError(f"lora_A has {len(factors.a)} row(s); r={factors.r} expects {factors.r}")
    if not factors.a or not factors.b:
        raise MsftError("a LoRA factor pair cannot be empty")
    d_in = len(factors.a[0])
    if any(len(row) != d_in for row in factors.a):
        raise MsftError("lora_A rows are not all d_in wide")
    if any(len(row) != factors.r for row in factors.b):
        raise MsftError(f"lora_B is not {factors.r} wide; it must be d_out x r")


def concatenate(first: Factors, second: Factors, *, mode: str = "mean") -> Factors:
    """Rank-2r concatenation: ``[A_a;A_b]`` and ``[B_a|B_b]`` with the alpha rewrite."""
    if mode not in MODES:
        raise MsftError(f"weights mode must be one of {', '.join(MODES)}, not {mode!r}")
    _check_dims(first)
    _check_dims(second)
    if first.r != second.r:
        raise MsftError(f"r differs: {first.r} vs {second.r}")
    if first.alpha != second.alpha:
        raise MsftError(f"lora_alpha differs: {first.alpha} vs {second.alpha}")
    if len(first.a[0]) != len(second.a[0]):
        raise MsftError("d_in differs between the adapters")
    if len(first.b) != len(second.b):
        raise MsftError("d_out differs between the adapters")
    a = [list(row) for row in first.a] + [list(row) for row in second.a]
    b = [list(left) + list(right) for left, right in zip(first.b, second.b, strict=True)]
    alpha = first.alpha if mode == "mean" else 2 * first.alpha
    return Factors(a=a, b=b, r=2 * first.r, alpha=alpha)


def merged_config(cfg: dict[str, Any], mode: str = "mean") -> dict[str, Any]:
    """peft-valid config for the concatenated adapter: ``r'=2r`` and the mode's alpha."""
    if mode not in MODES:
        raise MsftError(f"weights mode must be one of {', '.join(MODES)}, not {mode!r}")
    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    merged = {key: cfg[key] for key in CONFIG_KEYS if key in cfg}
    merged["r"] = 2 * r
    merged["lora_alpha"] = alpha if mode == "mean" else 2 * alpha
    return merged


# ------------------------------------------------------------------ adapter inspection


def read_header(path: Path) -> dict[str, dict[str, Any]]:
    """Tensor names, dtypes and shapes from a safetensors header, without importing torch.

    The format starts with a little-endian uint64 header length followed by a JSON object;
    every tensor entry carries ``dtype``, ``shape`` and ``data_offsets``. Reading it here is
    what lets the plan refuse a shape mismatch in CI, where safetensors is not installed.
    """
    with Path(path).open("rb") as handle:
        size = handle.read(8)
        if len(size) != 8:
            raise MsftError(f"{path}: not a safetensors file (short header length)")
        length = int.from_bytes(size, "little")
        if not 0 < length <= 1_000_000_000:
            raise MsftError(f"{path}: implausible safetensors header length {length}")
        raw = handle.read(length)
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MsftError(f"{path}: safetensors header is not JSON: {exc.msg}") from exc
    if not isinstance(header, dict):
        raise MsftError(f"{path}: safetensors header is not an object")
    header.pop("__metadata__", None)
    tensors: dict[str, dict[str, Any]] = {}
    for key, value in header.items():
        if not isinstance(value, dict):
            raise MsftError(f"{path}: malformed tensor entry {key!r}")
        shape = value.get("shape")
        if not isinstance(shape, list) or not all(isinstance(item, int) for item in shape):
            raise MsftError(f"{path}: tensor {key!r} has no integer shape")
        tensors[key] = {"dtype": str(value.get("dtype")), "shape": list(shape)}
    if not tensors:
        raise MsftError(f"{path}: safetensors file holds no tensors")
    return tensors


def split_key(key: str) -> tuple[str, str] | None:
    """Split a peft tensor name into ``(module, "A"|"B")``; None when it is neither."""
    for marker, role in _A_MARKERS:
        if key.endswith(marker):
            return key[: -len(marker)], role
    return None, None


def _load_config(directory: Path) -> dict[str, Any]:
    """Read an adapter's ``adapter_config.json`` or explain what is missing."""
    path = Path(directory) / CONFIG_NAME
    if not path.is_file():
        raise MsftError(f"{path} is missing; train the arm before merging it")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MsftError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(cfg, dict):
        raise MsftError(f"{path}: expected a JSON object")
    return cfg


def _ordered_set(values: Any) -> list[str]:
    """De-duplicate a module list while keeping peft's order."""
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(item) for item in values))


def _check_pair(a_dir: Path, b_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refuse two adapters that are not comparable; return both configs."""
    cfg_a = _load_config(a_dir)
    cfg_b = _load_config(b_dir)
    for key in ("base_model_name_or_path", "revision", "fan_in_fan_out", "task_type"):
        left, right = cfg_a.get(key), cfg_b.get(key)
        if left != right:
            raise MsftError(f"adapters differ in '{key}': {left!r} vs {right!r}")
    for key in ("r", "lora_alpha"):
        if cfg_a.get(key) != cfg_b.get(key):
            raise MsftError(f"adapters differ in '{key}': {cfg_a.get(key)!r} vs {cfg_b.get(key)!r}")
    if _ordered_set(cfg_a.get("target_modules")) != _ordered_set(cfg_b.get("target_modules")):
        raise MsftError("adapters differ in 'target_modules'")
    for label, cfg in (("a", cfg_a), ("b", cfg_b)):
        if cfg.get("use_rslora"):
            raise MsftError(f"adapter {label} uses rsLoRA; its scaling is not r-invariant")
        if cfg.get("use_dora"):
            raise MsftError(f"adapter {label} uses DoRA; magnitude vectors cannot be concatenated")
        if _ordered_set(cfg.get("modules_to_save")):
            raise MsftError(
                f"adapter {label} has a non-empty modules_to_save: "
                f"{cfg.get('modules_to_save')!r}"
            )
        if cfg.get("bias", "none") != "none":
            raise MsftError(
                f"adapter {label} trains bias={cfg.get('bias')!r}; only bias 'none' merges"
            )
        for key in ("rank_pattern", "alpha_pattern"):
            if cfg.get(key):
                raise MsftError(
                    f"adapter {label} has a non-empty {key}; per-layer ranks differ"
                )
    return cfg_a, cfg_b


def _shape_table(
    head_a: dict[str, dict[str, Any]], head_b: dict[str, dict[str, Any]], rank: int
) -> list[dict[str, Any]]:
    """Refuse mismatched tensor tables; return one row per module with its dims."""
    only_a = sorted(set(head_a) - set(head_b))
    only_b = sorted(set(head_b) - set(head_a))
    if only_a or only_b:
        raise MsftError(
            f"tensor key sets differ: only in a {only_a[:3]}, only in b {only_b[:3]}"
        )
    roles: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {}
    for key in sorted(head_a):
        module, role = split_key(key)
        if module is None or role is None:
            raise MsftError(f"{key!r} is not a lora_A/lora_B tensor; refusing to merge it")
        if head_a[key]["shape"] != head_b[key]["shape"]:
            raise MsftError(
                f"shape of {key!r} differs: {head_a[key]['shape']} vs "
                f"{head_b[key]['shape']}"
            )
        if head_a[key]["dtype"] != head_b[key]["dtype"]:
            raise MsftError(
                f"dtype of {key!r} differs: {head_a[key]['dtype']} vs "
                f"{head_b[key]['dtype']}"
            )
        roles.setdefault(module, {})[role] = (key, head_a[key])

    modules: list[dict[str, Any]] = []
    for module in sorted(roles):
        pair = roles[module]
        if set(pair) != {"A", "B"}:
            raise MsftError(
                f"module {module!r} has only {sorted(pair)}; "
                "both lora_A and lora_B must be present"
            )
        a_key, a_meta = pair["A"]
        b_key, b_meta = pair["B"]
        a_shape, b_shape = a_meta["shape"], b_meta["shape"]
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise MsftError(f"module {module!r} is not a 2-D LoRA pair: {a_shape}, {b_shape}")
        d_in, d_out = a_shape[1], b_shape[0]
        if a_shape[0] != rank or b_shape[1] != rank:
            raise MsftError(
                f"module {module!r} carries rank {a_shape[0]}/{b_shape[1]}, config says {rank}"
            )
        modules.append(
            {
                "module": module,
                "d_out": d_out,
                "d_in": d_in,
                "r": rank,
                "r_out": 2 * rank,
                "a_key": a_key,
                "b_key": b_key,
                "dtype": a_meta["dtype"],
                "a_shape": a_shape,
                "b_shape": b_shape,
            }
        )
    if not modules:
        raise MsftError("no LoRA modules found in the two adapters")
    return modules


def first_prompt(cases: str | Path | None) -> str:
    """The first usable prompt in a cases JSONL, or the built-in probe."""
    if cases is None:
        return DEFAULT_PROMPT
    path = Path(cases)
    if not path.is_file():
        raise MsftError(f"cases file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        value = record.get("prompt")
        if isinstance(value, str) and value.strip():
            return value
        value = record.get("user")
        if isinstance(value, str) and value.strip():
            return value
        messages = record.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
    raise MsftError(f"no prompt in {path}")


# ------------------------------------------------------------------------- planning


def plan(
    a: str | Path,
    b: str | Path,
    out: str | Path,
    *,
    weights: str = "mean",
    max_seq_length: int = 4096,
    cases: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the pair and describe the merge; raise `MsftError` on anything not comparable."""
    if weights not in MODES:
        raise MsftError(f"--weights must be one of {', '.join(MODES)}, not {weights!r}")
    a_dir, b_dir, out_dir = Path(a), Path(b), Path(out)
    cfg_a, cfg_b = _check_pair(a_dir, b_dir)
    rank = int(cfg_a["r"])
    alpha = int(cfg_a["lora_alpha"])

    weights_a = a_dir / WEIGHTS_NAME
    weights_b = b_dir / WEIGHTS_NAME
    checks = {
        "a_config": True,
        "b_config": True,
        "a_weights": weights_a.is_file(),
        "b_weights": weights_b.is_file(),
    }
    notes: list[str] = []
    modules: list[dict[str, Any]] = []
    if checks["a_weights"] and checks["b_weights"]:
        modules = _shape_table(read_header(weights_a), read_header(weights_b), rank)
    else:
        missing = [
            str(path)
            for path, ok in (
                (weights_a, checks["a_weights"]),
                (weights_b, checks["b_weights"]),
            )
            if not ok
        ]
        notes.append(f"weights missing ({', '.join(missing)}); run `msft train --execute` first")

    merged = merged_config(cfg_a, weights)
    dropped = sorted(set(cfg_a) - set(CONFIG_KEYS))
    if dropped:
        notes.append(f"adapter config key(s) not carried over: {', '.join(dropped)}")

    return {
        "stage": "adapters-merge",
        "a": str(a_dir),
        "b": str(b_dir),
        "out": str(out_dir),
        "adapter": str(out_dir / "adapter"),
        "meta": str(out_dir / META_NAME),
        "script": str(out_dir / SCRIPT_NAME),
        "weights": weights,
        "r": rank,
        "alpha": alpha,
        "r_out": merged["r"],
        "alpha_out": merged["lora_alpha"],
        "scale_in": rank and alpha / rank,
        "scale_out": merged["lora_alpha"] / merged["r"],
        "base_model": cfg_a.get("base_model_name_or_path"),
        "revision": cfg_a.get("revision"),
        "task_type": cfg_a.get("task_type"),
        "target_modules": _ordered_set(cfg_a.get("target_modules")),
        "probe": first_prompt(cases),
        "modules": modules,
        "checks": checks,
        "notes": notes,
        "executed": False,
        "command": [sys.executable, str(out_dir / SCRIPT_NAME)],
    }


_SCRIPT = Template(
    '''#!/usr/bin/env python
"""LoRA delta merge generated by `msft adapters merge` -- standalone, imports nothing from msft.

A:     $A_DIR
B:     $B_DIR
Out:   $OUT_DIR
Mode:  $MODE (mean = (delta_a + delta_b) / 2, sum = delta_a + delta_b)
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

A_DIR = Path($A_DIR)
B_DIR = Path($B_DIR)
OUT_DIR = Path($OUT_DIR)
MODE = $MODE
KEEP_KEYS = $KEEP_KEYS
PROBE = $PROMPT
BASE_FALLBACK = $BASE_FALLBACK
MAX_SEQ_LENGTH = $MAX_SEQ

WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
META_NAME = "merge_meta.json"
MARKERS = (
    (".lora_A.weight", "A"),
    (".lora_A.default.weight", "A"),
    (".lora_B.weight", "B"),
    (".lora_B.default.weight", "B"),
)


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    return 1


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_key(key):
    for marker, role in MARKERS:
        if key.endswith(marker):
            return key[: -len(marker)], role
    return None, None


def main():
    if not torch.cuda.is_available():
        return fail("no CUDA device: the merge self-check loads the base model")

    cfg_a = json.loads((A_DIR / CONFIG_NAME).read_text(encoding="utf-8"))
    cfg_b = json.loads((B_DIR / CONFIG_NAME).read_text(encoding="utf-8"))
    if cfg_a.get("r") != cfg_b.get("r") or cfg_a.get("lora_alpha") != cfg_b.get("lora_alpha"):
        return fail(
            "the two adapter configs disagree on r/lora_alpha; re-run `msft adapters merge`"
        )

    tensors_a = load_file(str(A_DIR / WEIGHTS_NAME))
    tensors_b = load_file(str(B_DIR / WEIGHTS_NAME))

    merged = {}
    pairs = {}
    for key in sorted(tensors_a):
        module, role = split_key(key)
        if module is None or role is None:
            return fail(f"unexpected tensor {key!r}: only lora_A/lora_B weights merge")
        merged[key] = torch.cat([tensors_a[key], tensors_b[key]], dim=0 if role == "A" else 1)
        pairs.setdefault(module, {})[role] = key

    out_adapter = OUT_DIR / "adapter"
    out_adapter.mkdir(parents=True, exist_ok=True)
    save_file({key: merged[key] for key in sorted(merged)}, str(out_adapter / WEIGHTS_NAME))

    r_in = int(cfg_a["r"])
    alpha_in = int(cfg_a["lora_alpha"])
    cfg = {key: cfg_a[key] for key in KEEP_KEYS if key in cfg_a}
    cfg["r"] = 2 * r_in
    cfg["lora_alpha"] = alpha_in if MODE == "mean" else 2 * alpha_in
    cfg["base_model_name_or_path"] = cfg_a.get("base_model_name_or_path") or BASE_FALLBACK
    (out_adapter / CONFIG_NAME).write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )

    base = cfg["base_model_name_or_path"]
    if not base:
        return fail("the adapters do not name a base model, and none was passed in")

    from peft import PeftModel
    from peft import __version__ as peft_version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map={"": "cuda"}
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map={"": "cuda"}
        )
    model = PeftModel.from_pretrained(model, str(out_adapter))
    model.eval()
    print(f"self-check 1: adapter reloads on {base} as {type(model).__name__}")

    scale = alpha_in / r_in
    scale_out = cfg["lora_alpha"] / cfg["r"]
    module = sorted(pairs)[0]
    a_key, b_key = pairs[module]["A"], pairs[module]["B"]
    delta_a = scale * (tensors_a[b_key].double() @ tensors_a[a_key].double())
    delta_b = scale * (tensors_b[b_key].double() @ tensors_b[a_key].double())
    delta_out = scale_out * (merged[b_key].double() @ merged[a_key].double())
    target = (delta_a + delta_b) / 2 if MODE == "mean" else delta_a + delta_b
    error = (delta_out - target).abs().max().item()
    limit = 1e-6 * max(delta_a.abs().max().item(), delta_b.abs().max().item(), 1e-12)
    if not error <= limit:
        return fail(f"delta merge is not exact on {module}: {error:.3e} > {limit:.3e}")
    print(f"self-check 2: max|delta_out - {MODE}| = {error:.3e} <= {limit:.3e} on {module}")

    messages = [{"role": "user", "content": PROBE}]
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
    except TypeError:
        print("note: this tokenizer has no enable_thinking flag", file=sys.stderr)
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    if not all(bool(torch.isfinite(score).all()) for score in output.scores):
        return fail("generation produced non-finite logits")
    reply = tokenizer.decode(
        output.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    if not reply:
        return fail("the merged adapter produced an empty reply")
    print(f"self-check 3: {len(reply)}-char reply, think: false, finite logits")

    meta = {
        "a": str(A_DIR),
        "b": str(B_DIR),
        "a_sha256": sha256(A_DIR / WEIGHTS_NAME),
        "b_sha256": sha256(B_DIR / WEIGHTS_NAME),
        "weights": MODE,
        "r": r_in,
        "alpha": alpha_in,
        "r_out": cfg["r"],
        "alpha_out": cfg["lora_alpha"],
        "base_model": base,
        "modules": {
            name: {
                "d_in": int(merged[keys["A"]].shape[1]),
                "d_out": int(merged[keys["B"]].shape[0]),
                "r": int(merged[keys["A"]].shape[0]),
                "a_shape": list(merged[keys["A"]].shape),
                "b_shape": list(merged[keys["B"]].shape),
                "dtype": str(merged[keys["A"]].dtype),
            }
            for name, keys in sorted(pairs.items())
        },
        "peft": peft_version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT_DIR / META_NAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
    print(f"merged adapter written to {out_adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
)


def build_script(
    a: str | Path,
    b: str | Path,
    out: str | Path,
    *,
    weights: str = "mean",
    max_seq_length: int = 4096,
    base_model: str | None = None,
) -> str:
    """The standalone merge script, ready to save and run with any python that has torch+peft."""
    if weights not in MODES:
        raise MsftError(f"--weights must be one of {', '.join(MODES)}, not {weights!r}")
    return _SCRIPT.substitute(
        A_DIR=json.dumps(str(Path(a))),
        B_DIR=json.dumps(str(Path(b))),
        OUT_DIR=json.dumps(str(Path(out))),
        MODE=json.dumps(weights),
        KEEP_KEYS=json.dumps(list(CONFIG_KEYS)),
        PROMPT=json.dumps(DEFAULT_PROMPT),
        BASE_FALLBACK=json.dumps(base_model),
        MAX_SEQ=max_seq_length,
    )


def _log(message: str) -> None:
    """Progress line on stderr, so a `--json` run keeps stdout to one object."""
    print(message, file=sys.stderr)


def _stream(command: list[str]) -> tuple[int, list[str]]:
    """Run the script, echo every line and return (rc, lines)."""
    lines: list[str] = []
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        _log(line.rstrip("\n"))
    return process.wait(), lines


def run(
    a: str | Path,
    b: str | Path,
    out: str | Path,
    *,
    weights: str = "mean",
    cases: str | Path | None = None,
    max_seq_length: int = 4096,
    execute: bool = False,
) -> dict[str, Any]:
    """Plan the merge and, with `execute`, write and run the script.

    `rc=0` is not success: the script must print all three self-check lines and the three
    artifacts must exist afterwards.
    """
    result = plan(a, b, out, weights=weights, max_seq_length=max_seq_length, cases=cases)
    if not execute:
        return result

    checks = result["checks"]
    if not checks["a_weights"] or not checks["b_weights"]:
        raise MsftError(
            f"no adapter weights to merge: {result['a']} / {result['b']}; "
            "run `msft train --execute` for both arms first"
        )

    out_dir = Path(out)
    script_path = out_dir / SCRIPT_NAME
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        build_script(
            a,
            b,
            out,
            weights=weights,
            max_seq_length=max_seq_length,
            base_model=result["base_model"],
        ),
        encoding="utf-8",
    )
    _log(f"wrote {script_path}")
    _log(f"$ {sys.executable} {script_path}")

    returncode, lines = _stream([sys.executable, str(script_path)])
    if returncode != 0:
        raise MsftError(
            f"adapter merge script exited {returncode}; read the log above, fix the cause "
            f"and re-run: {script_path}"
        )
    missing = [
        marker
        for marker in ("self-check 1:", "self-check 2:", "self-check 3:")
        if not any(line.startswith(marker) for line in lines)
    ]
    if missing:
        raise MsftError(
            f"merge exited 0 without {', '.join(missing)}; rc=0 is not proof of a merge"
        )
    artifacts = [
        out_dir / "adapter" / CONFIG_NAME,
        out_dir / "adapter" / WEIGHTS_NAME,
        out_dir / META_NAME,
    ]
    absent = [str(path) for path in artifacts if not path.is_file()]
    if absent:
        raise MsftError(f"merge exited 0 but wrote no {', '.join(absent)}")

    written = json.loads(artifacts[0].read_text(encoding="utf-8"))
    if written.get("r") != 2 * int(result["r"]):
        raise MsftError(
            f"merged adapter has r={written.get('r')}; expected {2 * int(result['r'])}"
        )

    result["executed"] = True
    result["returncode"] = returncode
    result["self_checks"] = [line for line in lines if line.startswith("self-check")]
    result["written"] = [str(path) for path in artifacts]
    return result
