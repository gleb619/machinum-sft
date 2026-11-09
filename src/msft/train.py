"""Training: JSON plan, the generated Unsloth/TRL script, and its execution.

The generated script is standalone: it imports nothing from ``msft`` so it can be copied to a
remote GPU box as-is. No heavy dependency is imported at module import time.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import Any

from msft.config import MsftError, RunConfig

TRAIN_SCRIPT_NAME = "train_script.py"
RUN_META_NAME = "run_meta.json"
CHECKPOINT_DIR_NAME = "checkpoints"
MIN_TRL = (0, 16)

IGNORED_KNOBS: dict[str, str] = {
    "model.dtype": "the script passes dtype=None so Unsloth chooses the device dtype",
}

MASK_REPORT_SOURCE = '''\
# --- loss-mask report: shared verbatim with src/msft/train.py ---
def mask_report(input_ids, labels, attention_mask, *, prompt_len, tokenizer=None) -> dict:
    """Count labeled vs real tokens in one collated batch and name the loss mask."""
    real = [i for i, flag in enumerate(attention_mask) if flag]
    labeled = [i for i in real if labels[i] != -100]
    real_tokens = len(real)
    labeled_tokens = len(labeled)
    first = labeled[0] if labeled else None
    last_real = real[-1] if real else None
    last_labeled = last_real is not None and labels[last_real] != -100
    masked_leading = first if first is not None else real_tokens
    if first == 0 and real_tokens and labeled_tokens >= 0.98 * real_tokens:
        verdict = "FULL-SEQUENCE"
    elif first is not None and abs(first - prompt_len) <= 2 and last_labeled:
        verdict = "COMPLETION-ONLY"
    else:
        verdict = "UNEXPECTED"
    report = {
        "real_tokens": real_tokens,
        "labeled": labeled_tokens,
        "masked_leading": masked_leading,
        "first_labeled": first,
        "last_real_labeled": last_labeled,
        "verdict": verdict,
    }
    if tokenizer is not None and first is not None:
        snippet = tokenizer.decode(list(input_ids)[first : first + 16])
        report["first_labeled_text"] = snippet.replace("\\n", "\\\\n")[:40]
    return report
# --- end loss-mask report ---
'''

_MASK_NAMESPACE: dict[str, Any] = {}
exec(MASK_REPORT_SOURCE, _MASK_NAMESPACE)
mask_report = _MASK_NAMESPACE["mask_report"]


def dataset_paths(cfg: RunConfig) -> dict[str, Path]:
    """The train/validation JSONL files produced by `msft data build`."""
    return {
        "train": cfg.dataset_path / "train.jsonl",
        "val": cfg.dataset_path / "val.jsonl",
    }


def _count_jsonl(path: Path) -> int | None:
    """Number of non-empty JSON lines in `path`, or None when the file does not exist."""
    if not path.is_file():
        return None
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _params_b(model_id: str) -> float | None:
    """Parameter count in billions parsed from the model name, e.g. `Qwen3.5-9B` -> 9.0."""
    tail = model_id.rsplit("/", 1)[-1]
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z0-9])", tail)
    return float(match.group(1)) if match else None


def vram_class(cfg: RunConfig, *, target_gb: float = 16.0) -> dict[str, Any]:
    """Coarse QLoRA footprint class for a 16 GB card; a planning hint, not a measurement."""
    params = _params_b(cfg.model.id)
    if params is None:
        return {
            "target_gb": target_gb,
            "params_b": None,
            "estimated_gb": None,
            "class": "unknown",
            "note": "model id carries no '<N>B' size; `msft doctor` measures the real figure",
        }
    weights = 0.56 * params  # 4-bit NF4 weights + quantisation metadata
    overhead = 4.2  # CUDA context, dequant buffers, logits, LoRA and optimizer state
    seq_delta = (cfg.train.max_seq_length / 4096.0 - 1.0) * 0.35 * math.sqrt(params)
    estimate = round(weights + overhead + seq_delta, 1)
    if estimate <= 0.72 * target_gb:
        klass = "fits"
    elif estimate <= 0.95 * target_gb:
        klass = "tight"
    else:
        klass = "exceeds"
    note = (
        f"{'4-bit QLoRA' if cfg.model.load_in_4bit else 'bf16 LoRA'}, r={cfg.lora.r}, "
        f"seq={cfg.train.max_seq_length}, "
        f"batch={cfg.train.per_device_batch}x{cfg.train.grad_accum}"
    )
    if cfg.model.load_in_4bit and cfg.model.id.lower().startswith("qwen/qwen3.5"):
        note += "; Unsloth discourages QLoRA on Qwen3.5 — prefer bf16 LoRA (4B) or Gemma 4"
    if klass != "fits":
        note += "; drop max_seq_length or r if the run OOMs"
    return {
        "target_gb": target_gb,
        "params_b": params,
        "estimated_gb": estimate,
        "class": klass,
        "note": note,
    }


def step_estimate(cfg: RunConfig, rows: dict[str, int | None]) -> dict[str, Any]:
    """Optimizer-step estimate, or `None` counts when `train.jsonl` is not built yet."""
    train_rows = rows.get("train")
    micro_batch = cfg.train.per_device_batch * cfg.train.grad_accum
    if train_rows is None:
        return {
            "per_epoch": None,
            "total": None,
            "warmup": cfg.train.warmup_steps,
            "note": "no train.jsonl yet; run `msft data build` to get a step count",
        }
    per_epoch = max(1, math.ceil(train_rows / micro_batch))
    total = max(1, math.ceil(per_epoch * cfg.train.epochs))
    note = f"{train_rows} rows / effective batch {micro_batch}"
    if cfg.train.packing:
        note += "; packing is on, so step counts assume packed sequences"
    return {
        "per_epoch": per_epoch,
        "total": total,
        "warmup": cfg.train.warmup_steps,
        "evals": max(1, total // max(1, cfg.train.eval_steps)),
        "note": note,
    }


def plan(cfg: RunConfig) -> dict[str, Any]:
    """Everything the training run will touch: paths, batch shape, steps, size class."""
    paths = dataset_paths(cfg)
    rows = {split: _count_jsonl(path) for split, path in paths.items()}
    script_path = cfg.output_path / TRAIN_SCRIPT_NAME
    return {
        "stage": "train",
        "name": cfg.name,
        "model": cfg.model.id,
        "revision": cfg.model.revision,
        "adapter": str(cfg.adapter_path),
        "run_meta": str(cfg.output_path / RUN_META_NAME),
        "checkpoints": str(cfg.output_path / CHECKPOINT_DIR_NAME),
        "script": str(script_path),
        "command": [sys.executable, str(script_path)],
        "install": "uv pip install -e '.[train]'",
        "dataset": {split: str(path) for split, path in paths.items()},
        "rows": rows,
        "batch": {
            "per_device": cfg.train.per_device_batch,
            "grad_accum": cfg.train.grad_accum,
            "effective": cfg.train.per_device_batch * cfg.train.grad_accum,
        },
        "steps": step_estimate(cfg, rows),
        "lora": {
            "r": cfg.lora.r,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": list(cfg.lora.target_modules),
            "modules_to_save": list(cfg.lora.modules_to_save),
            "use_rslora": cfg.lora.use_rslora,
        },
        "optim": {
            "lr": cfg.train.lr,
            "epochs": cfg.train.epochs,
            "optim": cfg.train.optim,
            "scheduler": cfg.train.scheduler,
            "max_seq_length": cfg.train.max_seq_length,
            "packing": cfg.train.packing,
            "completion_only_loss": cfg.train.completion_only_loss,
            "save_steps": cfg.train.save_steps,
            "eval_steps": cfg.train.eval_steps,
            "ignored_knobs": dict(IGNORED_KNOBS),
        },
        "vram_16gb": vram_class(cfg),
    }


_SCRIPT = Template(
    '''#!/usr/bin/env python
"""QLoRA/SFT run generated by `msft train` -- standalone, imports nothing from msft.

Model:   $MODEL_ID
Adapter: $ADAPTER_DIR
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Unsloth has to be imported before transformers/trl so its patches land first.
from unsloth import FastLanguageModel

import torch
import transformers
import trl
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL_ID = $MODEL_ID
MODEL_REVISION = $MODEL_REVISION
NAME = $NAME
ADAPTER_DIR = Path($ADAPTER_DIR)
RUN_META = Path($RUN_META)
CHECKPOINT_DIR = Path($CHECKPOINT_DIR)
TRAIN_FILE = $TRAIN_FILE
VAL_FILE = $VAL_FILE
PACKING = $PACKING
COMPLETION_ONLY = $COMPLETION_ONLY
MIN_TRL = $MIN_TRL


def fail(message: str) -> int:
    """Report a fatal problem without a traceback."""
    print(f"error: {message}", file=sys.stderr)
    return 1


def trl_version() -> tuple[int, ...]:
    """Major/minor tuple of the installed TRL, or () when it cannot be parsed."""
    parts: list[int] = []
    for chunk in trl.__version__.split(".")[:2]:
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def versions() -> dict[str, str]:
    """Library versions recorded in run_meta.json."""
    import datasets
    import unsloth

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "datasets": datasets.__version__,
        "unsloth": getattr(unsloth, "__version__", "unknown"),
    }


def gpu_name() -> str | None:
    """Name of the first visible GPU, or None on a CPU box."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


$MASK_REPORT_SOURCE


def text_tokenizer(tokenizer):
    """The real text tokenizer behind a processor.

    A Processor wraps one and treats a positional string as *image* input, which fails with
    "Incorrect image source ... Got <|im_start|>system" for a plain chat row.
    """
    return getattr(tokenizer, "tokenizer", tokenizer)


def render_row(tokenizer, messages) -> str:
    """Render one conversation to text, unwrapping a Processor if the model ships one."""
    out = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if isinstance(out, str):
        return out
    return out["text"][0] if isinstance(out.get("text"), list) else out["text"]


def render_prompt(tokenizer, messages) -> str:
    """Render a conversation through the assistant's generation header (the masked region)."""
    out = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(out, str):
        return out
    return out["text"][0] if isinstance(out.get("text"), list) else out["text"]


def completion_split(tokenizer, messages):
    """Token ids of the full row and the length of its prompt, through the assistant header.

    The assistant header must be masked too, otherwise the model is trained to emit
    `<|im_start|>assistant`. The split is only sound when the rendered prompt is a literal
    prefix of the rendered row; a template that reorders or rewrites the prompt cannot be
    masked by token count, so refuse rather than guess.
    """
    full = render_row(tokenizer, messages)
    prompt = render_prompt(tokenizer, messages[:-1])
    if not full.startswith(prompt):
        fail(
            "cannot build a completion-only example: the chat template's prompt is not a "
            "prefix of the full row, so the completion boundary cannot be located. "
            f"prompt={prompt[:80]!r} full={full[:80]!r}"
        )
        raise SystemExit(1)
    text_tok = text_tokenizer(tokenizer)
    input_ids = list(text_tok(full, add_special_tokens=False)["input_ids"])
    prompt_len = len(text_tok(prompt, add_special_tokens=False)["input_ids"])
    if prompt_len > len(input_ids):
        fail(
            "cannot build a completion-only example: the prompt tokenises to more tokens "
            f"({prompt_len}) than the full row ({len(input_ids)})"
        )
        raise SystemExit(1)
    return input_ids, prompt_len


def expected_prompt_len(tokenizer, messages) -> int:
    """Token count of the prompt the collator masks, including the assistant header."""
    return completion_split(tokenizer, messages)[1]


def build_example(tokenizer, messages, *, completion_only_loss: bool) -> dict:
    """Tokenise one chat row; mask the prompt when completion-only loss is requested."""
    input_ids, prompt_len = completion_split(tokenizer, messages)
    if not input_ids:
        fail("the chat template rendered an empty row")
        raise SystemExit(1)
    if not completion_only_loss:
        return {"input_ids": input_ids, "labels": list(input_ids)}
    if prompt_len >= len(input_ids):
        fail(
            "completion_only_loss would mask every token: the row has no assistant reply "
            f"({len(input_ids)} tokens, prompt {prompt_len})"
        )
        raise SystemExit(1)
    return {"input_ids": input_ids, "labels": [-100] * prompt_len + input_ids[prompt_len:]}


class MsftCollator:
    """Right-pad `input_ids` with the pad token, `labels` with -100, and mask the padding.

    Examples must already carry `input_ids` and `labels`; padding is only ever created here,
    so an attention mask of 0 always means a pad slot and never a real token.
    """

    def __init__(self, tokenizer) -> None:
        text_tok = text_tokenizer(tokenizer)
        pad_id = getattr(text_tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(text_tok, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        self.pad_id = int(pad_id)

    def __call__(self, examples):
        import torch

        input_ids = [list(example["input_ids"]) for example in examples]
        labels = [list(example["labels"]) for example in examples]
        width = max(len(row) for row in input_ids)
        attention_mask = []
        for row, label_row in zip(input_ids, labels):
            pad = width - len(row)
            attention_mask.append([1] * len(row) + [0] * pad)
            row.extend([self.pad_id] * pad)
            label_row.extend([-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def main() -> int:
    """Load the base model, train the adapter, save it next to run_meta.json."""
    started = time.time()

    # --- device guard -------------------------------------------------------------
    # A CPU run is never what anyone wants here: the same 0.8B job takes minutes on a GPU
    # and days on a CPU. Fail loudly instead of looking "slow but fine".
    if not torch.cuda.is_available():
        for line in (
            "!" * 78,
            "!! msft: NO CUDA DEVICE VISIBLE - refusing to train on the CPU.",
            "!! A 4B LoRA that takes ~3 min on a GPU takes days on a CPU; a silent fallback",
            "!! is almost always a broken container or driver, not a small job.",
            "!! Check: nvidia-smi, that the container runs with --gpus all,",
            "!! and that CUDA_VISIBLE_DEVICES is not empty.",
            "!! Override only if you truly mean it: MSFT_ALLOW_CPU=1",
            "!" * 78,
        ):
            print(line, file=sys.stderr, flush=True)
        if os.environ.get("MSFT_ALLOW_CPU") != "1":
            return fail("no CUDA device; set MSFT_ALLOW_CPU=1 to force a CPU run")
        print("!! msft: MSFT_ALLOW_CPU=1 - continuing on the CPU, expect hours.", flush=True)

    if trl_version() and trl_version() < MIN_TRL:
        return fail(f"trl {trl.__version__} is too old; completion_only_loss needs >= 0.16")

    if PACKING:
        return fail("train.packing=true is not supported with the msft collator")

    for path, split in ((TRAIN_FILE, "train"), (VAL_FILE, "validation")):
        if not Path(path).is_file():
            return fail(f"{split} dataset missing: {path}; run `msft data build` first")

    raw = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    if len(raw["validation"]) == 0:
        return fail("validation split is empty; rebuild the dataset with more input")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=$MAX_SEQ,
        dtype=None,
        load_in_4bit=$LOAD_IN_4BIT,
    )
    if getattr(tokenizer, "chat_template", None) is None:
        return fail(f"tokenizer for {MODEL_ID} has no chat template; pick a chat model")

    EXPECTED_PROMPT_LEN = expected_prompt_len(tokenizer, raw["train"][0]["messages"])

    def render(example: dict) -> dict:
        """Tokenise one conversation, masking the prompt for completion-only loss."""
        return build_example(
            tokenizer, example["messages"], completion_only_loss=COMPLETION_ONLY
        )

    train_ds = raw["train"].map(
        render, remove_columns=raw["train"].column_names, desc="chat template"
    )
    val_ds = raw["validation"].map(
        render, remove_columns=raw["validation"].column_names or [], desc="chat template"
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=$LORA_R,
        lora_alpha=$LORA_ALPHA,
        lora_dropout=$LORA_DROPOUT,
        target_modules=$TARGET_MODULES,
        modules_to_save=$MODULES_TO_SAVE,
        use_rslora=$USE_RSLORA,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=$SEED,
    )

    # The model can still land on the CPU even with CUDA available (device_map misread,
    # a container that exposes the driver but not the device). Say where it actually is.
    _device = next(model.parameters()).device
    if _device.type != "cuda":
        for line in (
            "!" * 78,
            f"!! msft: the model is on {_device}, not a GPU - aborting, not burning CPU time.",
            "!! This is a configuration bug, not a slow machine: check nvidia-smi,",
            "!! the --gpus all flag, and CUDA_VISIBLE_DEVICES.",
            "!" * 78,
        ):
            print(line, file=sys.stderr, flush=True)
        return fail(f"model placed on {_device}; expected cuda")
    _props = torch.cuda.get_device_properties(_device)
    print(
        f"[msft] training on {_device} - {torch.cuda.get_device_name(_device)}"
        f" ({_props.total_memory / 2**30:.1f} GiB)",
        flush=True,
    )

    args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        dataset_text_field=None,
        max_length=$MAX_SEQ,
        packing=False,
        per_device_train_batch_size=$PER_DEVICE_BATCH,
        per_device_eval_batch_size=$PER_DEVICE_BATCH,
        gradient_accumulation_steps=$GRAD_ACCUM,
        learning_rate=$LR,
        num_train_epochs=$EPOCHS,
        warmup_steps=$WARMUP,
        weight_decay=$WEIGHT_DECAY,
        max_grad_norm=$MAX_GRAD_NORM,
        lr_scheduler_type=$SCHEDULER,
        optim=$OPTIM,
        bf16=True,
        fp16=False,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=$EVAL_STEPS,
        save_strategy="steps",
        save_steps=$SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=$SEED,
        data_seed=$SEED,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=MsftCollator(tokenizer),
        args=args,
    )

    # Measure the loss mask before training on the first training row, through the same
    # collator the trainer uses. The training dataloader shuffles, so its first row is not
    # necessarily row 0 and its prompt length would not match EXPECTED_PROMPT_LEN.
    batch = MsftCollator(tokenizer)([train_ds[0]])
    ids = batch["input_ids"][0].tolist()
    # MsftCollator always builds an attention mask; a hand-edited script might not, and a
    # batch without one is unpadded, so every position is real.
    attn = batch["attention_mask"][0].tolist() if "attention_mask" in batch else [1] * len(ids)
    report = mask_report(
        ids,
        batch["labels"][0].tolist(),
        attn,
        prompt_len=EXPECTED_PROMPT_LEN,
        tokenizer=tokenizer,
    )
    print(
        f"[msft] loss-mask: real_tokens={report['real_tokens']} "
        f"labeled={report['labeled']} masked_leading={report['masked_leading']} "
        f"(expected prompt len {EXPECTED_PROMPT_LEN})",
        flush=True,
    )
    if report["first_labeled"] is not None:
        snippet = report.get("first_labeled_text", "")
        print(f"[msft] loss-mask: first labeled: '{snippet}…'", flush=True)
    notes = {
        "FULL-SEQUENCE": "system and user tokens are trained",
        "COMPLETION-ONLY": "only the assistant reply is trained",
        "UNEXPECTED": "neither full-sequence nor completion-only; check the data collator",
    }
    print(
        f"[msft] loss-mask: VERDICT {report['verdict']} — {notes[report['verdict']]}",
        flush=True,
    )
    expected = "COMPLETION-ONLY" if COMPLETION_ONLY else "FULL-SEQUENCE"
    if report["verdict"] != expected:
        for line in (
            "!" * 78,
            f"!! msft: completion_only_loss={COMPLETION_ONLY} but this batch is "
            f"{report['verdict']}.",
            f"!! {report['labeled']} of {report['real_tokens']} real tokens are labeled;",
            "!! check the collator before spending GPU time on a mask that is not the one",
            "!! the config asks for.",
            "!" * 78,
        ):
            print(line, file=sys.stderr, flush=True)

    if os.environ.get("MSFT_INSPECT_MASK") == "1":
        print("[msft] MSFT_INSPECT_MASK=1: stopping before trainer.train()", flush=True)
        return 0

    result = trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    meta = {
        "name": NAME,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "adapter": str(ADAPTER_DIR),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"train": len(train_ds), "validation": len(val_ds)},
        "hyperparameters": {
            "r": $LORA_R,
            "alpha": $LORA_ALPHA,
            "dropout": $LORA_DROPOUT,
            "learning_rate": $LR,
            "epochs": $EPOCHS,
            "max_seq_length": $MAX_SEQ,
            "per_device_batch": $PER_DEVICE_BATCH,
            "grad_accum": $GRAD_ACCUM,
            "optim": $OPTIM,
            "completion_only_loss": COMPLETION_ONLY,
            "packing": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        },
        "metrics": result.metrics,
        "best_eval_loss": trainer.state.best_metric,
        "train_seconds": round(time.time() - started, 1),
        "versions": versions(),
        "gpu": gpu_name(),
    }
    RUN_META.parent.mkdir(parents=True, exist_ok=True)
    RUN_META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(f"adapter written to {ADAPTER_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
)


def _literal(value: object) -> str:
    """Python literal for a config value: `None` stays `None`, everything else is JSON."""
    return "None" if value is None else json.dumps(value)


def _list_literal(items: list[str], *, indent: int = 12) -> str:
    """Multi-line Python list literal, one quoted item per line at `indent` spaces."""
    pad = " " * indent
    body = (",\n" + pad).join(json.dumps(item) for item in items)
    return "[\n" + pad + body + ",\n" + " " * (indent - 4) + "]"


def build_script(cfg: RunConfig) -> str:
    """The standalone training script for `cfg`, ready to save and run with any python."""
    paths = dataset_paths(cfg)
    return _SCRIPT.substitute(
        MODEL_ID=json.dumps(cfg.model.id),
        MODEL_REVISION=_literal(cfg.model.revision),
        NAME=json.dumps(cfg.name),
        ADAPTER_DIR=json.dumps(str(cfg.adapter_path)),
        RUN_META=json.dumps(str(cfg.output_path / RUN_META_NAME)),
        CHECKPOINT_DIR=json.dumps(str(cfg.output_path / CHECKPOINT_DIR_NAME)),
        TRAIN_FILE=json.dumps(str(paths["train"])),
        VAL_FILE=json.dumps(str(paths["val"])),
        PACKING=cfg.train.packing,
        COMPLETION_ONLY=cfg.train.completion_only_loss,
        MAX_SEQ=cfg.train.max_seq_length,
        PER_DEVICE_BATCH=cfg.train.per_device_batch,
        GRAD_ACCUM=cfg.train.grad_accum,
        LR=repr(cfg.train.lr),
        EPOCHS=repr(cfg.train.epochs),
        WARMUP=cfg.train.warmup_steps,
        WEIGHT_DECAY=repr(cfg.train.weight_decay),
        MAX_GRAD_NORM=repr(cfg.train.max_grad_norm),
        SCHEDULER=json.dumps(cfg.train.scheduler),
        OPTIM=json.dumps(cfg.train.optim),
        SEED=cfg.train.seed,
        EVAL_STEPS=cfg.train.eval_steps,
        SAVE_STEPS=cfg.train.save_steps,
        LORA_R=cfg.lora.r,
        LORA_ALPHA=cfg.lora.alpha,
        LORA_DROPOUT=repr(cfg.lora.dropout),
        TARGET_MODULES=_list_literal(list(cfg.lora.target_modules)),
        MODULES_TO_SAVE=repr(list(cfg.lora.modules_to_save)),
        USE_RSLORA=cfg.lora.use_rslora,
        LOAD_IN_4BIT=cfg.model.load_in_4bit,
        MIN_TRL=repr(MIN_TRL),
        MASK_REPORT_SOURCE=MASK_REPORT_SOURCE,
    )


def _log(message: str) -> None:
    """Progress line on stderr, so a `--json` run keeps stdout to one object."""
    print(message, file=sys.stderr)


def run(
    cfg: RunConfig, *, execute: bool = False, inspect_mask: bool = False
) -> dict[str, Any]:
    """Return the plan; with `execute=True` write output/train_script.py and run it.

    Nothing is written when `execute` is false: a dry run must not create artifacts.
    """
    result = plan(cfg)
    result["executed"] = False
    if not execute:
        return result

    script_path = Path(result["script"])
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(build_script(cfg), encoding="utf-8")
    _log(f"wrote {script_path}")
    _log(f"$ {sys.executable} {script_path}")

    env = dict(os.environ)
    if inspect_mask:
        env["MSFT_INSPECT_MASK"] = "1"
    completed = subprocess.run(
        [sys.executable, str(script_path)], stdout=sys.stderr, check=False, env=env
    )
    if completed.returncode != 0:
        raise MsftError(
            f"training script exited {completed.returncode}; read the log above, fix the "
            f"cause and re-run: {script_path}"
        )
    result["executed"] = True
    result["returncode"] = completed.returncode
    if inspect_mask:
        result["inspect_mask"] = True
        return result
    adapter_config = cfg.adapter_path / "adapter_config.json"
    if not adapter_config.is_file():
        raise MsftError(
            f"training exited 0 but {adapter_config} is missing; the adapter was not saved"
        )
    return result
