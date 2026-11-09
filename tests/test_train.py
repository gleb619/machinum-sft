"""Tests for the training plan, the generated standalone script and its execution."""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from msft import merge, train
from msft.config import (
    DataCfg,
    LoraCfg,
    ModelCfg,
    MsftError,
    QuantCfg,
    RunConfig,
    TrainCfg,
    validate_config,
)


@pytest.fixture()
def cfg(tmp_path: Path) -> RunConfig:
    """Published Russian-recipe defaults rooted in `tmp_path`."""
    return RunConfig(
        name="ru-test",
        base_dir=str(tmp_path),
        output_dir="out",
        model=ModelCfg(id="Qwen/Qwen3.5-9B"),
        lora=LoraCfg(),
        train=TrainCfg(),
        data=DataCfg(),
        quantize=QuantCfg(),
    )


def _write_jsonl(path: Path, rows: int) -> None:
    """Write `rows` minimal chat records, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"messages": [{"role": "user", "content": f"q{i}"}]}) for i in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_plan_reports_paths_batch_and_vram(cfg: RunConfig) -> None:
    """The plan names the model, adapter, dataset files, batch shape and VRAM class."""
    plan = train.plan(cfg)
    assert plan["stage"] == "train"
    assert plan["model"] == "Qwen/Qwen3.5-9B"
    assert plan["adapter"] == str(Path(cfg.base_dir) / "out" / "adapter")
    assert plan["dataset"]["train"].endswith("out/dataset/train.jsonl")
    assert plan["dataset"]["val"].endswith("out/dataset/val.jsonl")
    assert plan["batch"] == {"per_device": 1, "grad_accum": 32, "effective": 32}
    assert plan["vram_16gb"]["class"] == "fits"
    assert plan["vram_16gb"]["params_b"] == 9.0
    assert plan["vram_16gb"]["target_gb"] == 16.0
    assert json.loads(json.dumps(plan)) == plan


def test_plan_has_no_step_count_without_dataset(cfg: RunConfig) -> None:
    """With no train.jsonl the plan still works and says how to get a step count."""
    steps = train.plan(cfg)["steps"]
    assert steps["per_epoch"] is None
    assert steps["total"] is None
    assert "data build" in steps["note"]


def test_plan_estimates_steps_from_dataset_rows(cfg: RunConfig) -> None:
    """70 train rows at effective batch 32 over 2 epochs -> 3 update steps per epoch."""
    _write_jsonl(Path(cfg.base_dir) / "out" / "dataset" / "train.jsonl", 70)
    _write_jsonl(Path(cfg.base_dir) / "out" / "dataset" / "val.jsonl", 3)
    plan = train.plan(cfg)
    assert plan["rows"] == {"train": 70, "val": 3}
    assert plan["steps"]["per_epoch"] == 3
    assert plan["steps"]["total"] == 6
    assert plan["steps"]["warmup"] == 8


def test_script_imports_unsloth_before_trl_and_transformers(cfg: RunConfig) -> None:
    """Unsloth's patches only land when it is imported before transformers/trl."""
    script = train.build_script(cfg)
    unsloth_at = script.index("from unsloth import FastLanguageModel")
    trl_at = script.index("from trl import SFTConfig, SFTTrainer")
    transformers_at = script.index("import transformers")
    assert unsloth_at < transformers_at < trl_at


def test_script_carries_the_recipe_arguments(cfg: RunConfig) -> None:
    """Every published default and every current TRL/Unsloth argument is in the script."""
    script = train.build_script(cfg)
    required = [
        "max_seq_length=4096",
        "dtype=None",
        "load_in_4bit=True",
        "r=32",
        "lora_alpha=16",
        "modules_to_save=['lm_head']",
        '"down_proj",',
        "random_state=42",
        'use_gradient_checkpointing="unsloth"',
        'load_dataset("json"',
        "apply_chat_template",
        "dataset_text_field=None",
        "max_length=4096",
        "build_example(",
        "data_collator=MsftCollator(tokenizer)",
        "MsftCollator",
        "packing=False",
        "per_device_train_batch_size=1",
        "gradient_accumulation_steps=32",
        "learning_rate=5e-05",
        "num_train_epochs=2.0",
        "warmup_steps=8",
        'optim="adamw_8bit"',
        "bf16=True",
        'eval_strategy="steps"',
        'save_strategy="steps"',
        "eval_steps=20",
        "save_steps=20",
        "load_best_model_at_end=True",
        'metric_for_best_model="eval_loss"',
        "model.save_pretrained(str(ADAPTER_DIR))",
        "tokenizer.save_pretrained(str(ADAPTER_DIR))",
        "run_meta.json",
    ]
    missing = [needle for needle in required if needle not in script]
    assert missing == []


def test_script_is_standalone_python(cfg: RunConfig) -> None:
    """The script compiles, holds no JSON literals and imports nothing from this package."""
    script = train.build_script(cfg)
    compile(script, "train_script.py", "exec")
    assert "null" not in script  # JSON null is not Python
    imports = [line.strip() for line in script.splitlines()]
    assert not [line for line in imports if line.startswith(("from msft", "import msft"))]


def test_merge_script_merges_the_adapter_to_16_bit(cfg: RunConfig) -> None:
    """The merge script loads the adapter with Unsloth and saves merged_16bit weights."""
    script = merge.build_script(cfg)
    compile(script, "merge_script.py", "exec")
    assert "null" not in script
    assert script.index("from unsloth import FastLanguageModel") < script.index("def main")
    assert "model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method=SAVE_METHOD)" in (
        script
    )
    assert 'SAVE_METHOD = "merged_16bit"' in script
    assert f'MERGED_DIR = Path("{cfg.merged_path}")' in script
    assert f'ADAPTER_DIR = Path("{cfg.adapter_path}")' in script


def test_larger_model_is_reported_as_too_big(tmp_path: Path) -> None:
    """A 32B model cannot fit a 16 GB card in 4-bit at 4k context; an unparsable id is unknown."""
    big = RunConfig(name="big", base_dir=str(tmp_path), model=ModelCfg(id="Qwen/Qwen2.5-32B"))
    assert train.vram_class(big)["class"] == "exceeds"
    vague = RunConfig(name="vague", base_dir=str(tmp_path), model=ModelCfg(id="acme/mystery"))
    assert train.vram_class(vague)["class"] == "unknown"


def test_run_without_execute_writes_nothing(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run returns the plan and never creates the output tree or spawns anything."""

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess must not run without execute=True")

    monkeypatch.setattr(train.subprocess, "run", boom)
    result = train.run(cfg)
    assert result["executed"] is False
    assert result["adapter"] == str(cfg.adapter_path)
    assert not cfg.output_path.exists()


def test_run_execute_writes_script_and_runs_it_under_this_python(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execution writes output/train_script.py and runs it with sys.executable."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        cfg.adapter_path.mkdir(parents=True, exist_ok=True)
        (cfg.adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    result = train.run(cfg, execute=True)

    script = cfg.output_path / "train_script.py"
    assert script.is_file()
    assert script.read_text(encoding="utf-8") == train.build_script(cfg)
    assert calls == [[sys.executable, str(script)]]
    assert result["executed"] is True


def test_run_execute_raises_on_non_zero_exit(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing child becomes MsftError, not a silent success."""

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    with pytest.raises(MsftError, match="exited 3"):
        train.run(cfg, execute=True)


def test_run_execute_rejects_a_silent_no_op(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0 without an adapter on disk is still a failure."""

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    with pytest.raises(MsftError, match="adapter_config.json"):
        train.run(cfg, execute=True)


def test_execute_sends_the_child_output_to_stderr(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A script printing to stdout would corrupt a `--json` result; it inherits our stderr."""
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], stdout=None, check: bool = False, **kwargs: object):
        captured["stdout"] = stdout
        cfg.adapter_path.mkdir(parents=True, exist_ok=True)
        (cfg.adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    train.run(cfg, execute=True)

    assert captured["stdout"] is sys.stderr
    assert capsys.readouterr().out == ""


# --- loss-mask measurement ------------------------------------------------


def _extract_mask_report(script: str):
    """Exec the report helper exactly as it is embedded in the generated script."""
    start = script.index("# --- loss-mask report")
    end = script.index("# --- end loss-mask report ---")
    end += len("# --- end loss-mask report ---")
    namespace: dict[str, object] = {}
    exec(script[start:end], namespace)
    return namespace["mask_report"]


def test_mask_report_full_sequence() -> None:
    report = train.mask_report([0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1], prompt_len=2)
    assert report["verdict"] == "FULL-SEQUENCE"
    assert report["real_tokens"] == 4
    assert report["labeled"] == 4
    assert report["masked_leading"] == 0


def test_mask_report_completion_only() -> None:
    report = train.mask_report(
        [0, 1, 2, 3, 4], [-100, -100, 2, 3, 4], [1, 1, 1, 1, 1], prompt_len=2
    )
    assert report["verdict"] == "COMPLETION-ONLY"
    assert report["first_labeled"] == 2
    assert report["masked_leading"] == 2
    assert report["last_real_labeled"] is True


def test_mask_report_ignores_padding_tokens() -> None:
    """Truncation/padding leaves attention_mask==0 slots that must not count as real tokens."""
    report = train.mask_report(
        [0, 1, 2, 3, 0, 0], [-100, -100, 2, 3, -100, -100], [1, 1, 1, 1, 0, 0], prompt_len=2
    )
    assert report["real_tokens"] == 4
    assert report["labeled"] == 2
    assert report["verdict"] == "COMPLETION-ONLY"


def test_mask_report_flags_an_unlabeled_final_token() -> None:
    report = train.mask_report(
        [0, 1, 2, 3], [-100, -100, 2, -100], [1, 1, 1, 1], prompt_len=2
    )
    assert report["verdict"] == "UNEXPECTED"
    assert report["last_real_labeled"] is False


def test_mask_report_counts_every_leading_masked_token() -> None:
    report = train.mask_report(
        [0, 1, 2, 3, 4], [-100, -100, -100, 3, 4], [1, 1, 1, 1, 1], prompt_len=3
    )
    assert report["masked_leading"] == 3
    assert report["first_labeled"] == 3
    assert report["verdict"] == "COMPLETION-ONLY"


def test_the_embedded_mask_helper_matches_the_imported_one(cfg: RunConfig) -> None:
    embedded = _extract_mask_report(train.build_script(cfg))
    args = ([0, 1, 2, 3], [-100, -100, 2, 3], [1, 1, 1, 1])
    assert embedded(*args, prompt_len=2) == train.mask_report(*args, prompt_len=2)
    assert embedded(*args, prompt_len=2)["verdict"] == "COMPLETION-ONLY"


def test_the_script_measures_the_mask_before_training(cfg: RunConfig) -> None:
    script = train.build_script(cfg)
    ast.parse(script)
    assert "mask_report(" in script
    assert "MsftCollator(tokenizer)([train_ds[0]])" in script
    assert "MSFT_INSPECT_MASK" in script
    assert "loss-mask: VERDICT" in script
    assert script.index("SFTTrainer(") < script.index("= mask_report(")
    assert script.index("= mask_report(") < script.index("trainer.train()")


# --- our own collator: what completion_only_loss actually masks ----------


class WordTokenizer:
    """Whitespace tokenizer with a fake Qwen-style template; each word is one token."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._ids: dict[str, int] = {}

    def token_id(self, word: str) -> int:
        """Stable id for one word, assigning new ids in first-seen order."""
        if word not in self._ids:
            self._ids[word] = len(self._ids) + 10
        return self._ids[word]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self.token_id(word) for word in text.split()]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        text = " ".join(f"<|{m['role']}|> {m['content']} <|end|>" for m in messages)
        return f"{text} <|assistant|>" if add_generation_prompt else text


class ReorderedTokenizer(WordTokenizer):
    """A template that rewrites the prompt, so it is not a prefix of the full row."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        text = super().apply_chat_template(messages, tokenize, False)
        return f"PREFIX> {text}" if add_generation_prompt else text


MESSAGES = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "hello there"},
    {"role": "assistant", "content": "bonjour"},
]


def _extract_script_helpers(cfg: RunConfig) -> dict[str, object]:
    """Exec the generated helper functions without importing unsloth or torch."""
    script = train.build_script(cfg)
    start = script.index("def fail(")
    end = script.index("def main(")
    namespace: dict[str, object] = {"sys": sys}
    exec(script[start:end], namespace)
    return namespace


class _FakeTensor(list):
    """Just enough of a tensor for the collator's assertions."""

    def tolist(self) -> list[object]:
        return list(self)


def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a bare `import torch` succeed without the real 2 GB dependency."""
    module = types.ModuleType("torch")
    module.long = "long"
    module.tensor = lambda data, dtype=None: _FakeTensor(data)
    monkeypatch.setitem(sys.modules, "torch", module)


def test_build_example_masks_the_prompt_for_completion_only(cfg: RunConfig) -> None:
    helpers = _extract_script_helpers(cfg)
    tokenizer = WordTokenizer()
    example = helpers["build_example"](tokenizer, MESSAGES, completion_only_loss=True)

    prompt_len = len(helpers["render_prompt"](tokenizer, MESSAGES[:-1]).split())
    assert prompt_len > 0
    assert example["labels"][:prompt_len] == [-100] * prompt_len
    assert all(label != -100 for label in example["labels"][prompt_len:])
    # The response and its end marker are trained; the assistant header is not.
    assert example["labels"][prompt_len] == tokenizer.token_id("bonjour")
    assert example["labels"][-1] == tokenizer.token_id("<|end|>")


def test_build_example_labels_every_token_without_completion_only(cfg: RunConfig) -> None:
    helpers = _extract_script_helpers(cfg)
    example = helpers["build_example"](
        WordTokenizer(), MESSAGES, completion_only_loss=False
    )
    assert example["labels"] == example["input_ids"]
    assert -100 not in example["labels"]


def test_build_example_fails_when_the_prompt_is_not_a_prefix(
    cfg: RunConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    helpers = _extract_script_helpers(cfg)
    with pytest.raises(SystemExit):
        helpers["build_example"](
            ReorderedTokenizer(), MESSAGES, completion_only_loss=True
        )
    assert "not a prefix" in capsys.readouterr().err


def test_collator_pads_a_ragged_batch(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_torch(monkeypatch)
    helpers = _extract_script_helpers(cfg)
    collator = helpers["MsftCollator"](WordTokenizer())

    batch = collator(
        [
            {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
            {"input_ids": [8, 9], "labels": [8, 9]},
        ]
    )

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, 9, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_script_never_passes_completion_only_loss_to_trl(cfg: RunConfig) -> None:
    script = train.build_script(cfg)
    block = script[script.index("args = SFTConfig(") : script.index("trainer = SFTTrainer(")]
    assert "completion_only_loss" not in block
    assert "dataset_text_field=None" in block
    assert "MsftCollator" in script


def test_the_collator_output_reports_completion_only(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated script's own report must call this batch COMPLETION-ONLY."""
    _fake_torch(monkeypatch)
    helpers = _extract_script_helpers(cfg)
    tokenizer = WordTokenizer()
    example = helpers["build_example"](tokenizer, MESSAGES, completion_only_loss=True)
    batch = helpers["MsftCollator"](tokenizer)([example])
    prompt_len = len(helpers["render_prompt"](tokenizer, MESSAGES[:-1]).split())

    report = helpers["mask_report"](
        list(batch["input_ids"][0]),
        list(batch["labels"][0]),
        list(batch["attention_mask"][0]),
        prompt_len=prompt_len,
    )

    assert report["verdict"] == "COMPLETION-ONLY"
    assert report["masked_leading"] == prompt_len
    assert report["last_real_labeled"] is True


def test_inspect_mask_sets_the_env_and_skips_the_adapter_assertion(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    result = train.run(cfg, execute=True, inspect_mask=True)

    env = seen["env"]
    assert isinstance(env, dict)
    assert env["MSFT_INSPECT_MASK"] == "1"
    assert result["inspect_mask"] is True
    assert result["executed"] is True
    assert not cfg.adapter_path.exists()


def test_plan_lists_the_ignored_knobs(cfg: RunConfig) -> None:
    assert train.plan(cfg)["optim"]["ignored_knobs"] == train.IGNORED_KNOBS


def test_validate_config_rejects_packing_with_the_msft_collator(tmp_path: Path) -> None:
    cfg = RunConfig(
        name="model",
        base_dir=str(tmp_path),
        train=TrainCfg(packing=True),
        quantize=QuantCfg(output="out/model.gguf"),
    )
    with pytest.raises(MsftError, match="msft collator"):
        validate_config(cfg)


def _sentinel_for(value: object) -> object:
    """A value of the same shape but certainly different from `value`."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 7
    if isinstance(value, float):
        return value + 0.123
    if isinstance(value, list):
        return [*value, "sentinel_x"]
    return "sentinel_x"


def test_every_config_knob_reaches_the_script_or_is_listed_ignored(
    tmp_path: Path,
) -> None:
    baseline = train.build_script(RunConfig(name="knob", base_dir=str(tmp_path)))
    sections = {"model": ModelCfg, "lora": LoraCfg, "train": TrainCfg}

    for section, cls in sections.items():
        for spec in dataclasses.fields(cls):
            cfg = RunConfig(name="knob", base_dir=str(tmp_path))
            target = getattr(cfg, section)
            if spec.default is not dataclasses.MISSING:
                default = spec.default
            elif spec.default_factory is not dataclasses.MISSING:
                default = spec.default_factory()
            else:
                default = getattr(target, spec.name)
            setattr(target, spec.name, _sentinel_for(default))
            knob = f"{section}.{spec.name}"
            changed = train.build_script(cfg) != baseline
            if knob in train.IGNORED_KNOBS:
                assert not changed, (
                    f"stale IGNORED_KNOBS entry {knob}: it now changes the script; "
                    "remove it"
                )
            else:
                assert changed, (
                    f"knob {knob} is read from the config but never reaches the generated "
                    "script; use it or add it to IGNORED_KNOBS with a reason"
                )


def test_merge_sends_the_child_output_to_stderr(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg.adapter_path.mkdir(parents=True, exist_ok=True)
    (cfg.adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], stdout=None, check: bool = False, **kwargs: object):
        captured["stdout"] = stdout
        cfg.merged_path.mkdir(parents=True, exist_ok=True)
        (cfg.merged_path / "config.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(merge.subprocess, "run", fake_run)
    merge.run(cfg, execute=True)

    assert captured["stdout"] is sys.stderr
    assert capsys.readouterr().out == ""
