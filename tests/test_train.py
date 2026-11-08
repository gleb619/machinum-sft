"""Tests for the training plan, the generated standalone script and its execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from msft import merge, train
from msft.config import DataCfg, LoraCfg, ModelCfg, MsftError, QuantCfg, RunConfig, TrainCfg


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
        'dataset_text_field="text"',
        "max_length=4096",
        "completion_only_loss=True",
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

    def fake_run(argv: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
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

    def fake_run(argv: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    with pytest.raises(MsftError, match="exited 3"):
        train.run(cfg, execute=True)


def test_run_execute_rejects_a_silent_no_op(
    cfg: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0 without an adapter on disk is still a failure."""

    def fake_run(argv: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(train.subprocess, "run", fake_run)
    with pytest.raises(MsftError, match="adapter_config.json"):
        train.run(cfg, execute=True)
