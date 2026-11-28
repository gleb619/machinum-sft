"""Tests for `msft merge --adapter`: the union adapter changes the source, not the target."""

from __future__ import annotations

from pathlib import Path

import pytest

from msft import merge
from msft.config import MsftError, load_config

CONFIG = """\
name: w4-union-0.8b
model:
  id: Qwen/Qwen3.5-0.8B
  dtype: bfloat16
lora:
  r: 16
  alpha: 32
train:
  max_seq_length: 4096
data:
  val_ratio: 0.05
quantize:
  output: out/union-run/model-w4-union-0.8b.gguf
output_dir: out/union-run
"""


def config_for(tmp_path: Path):
    """A valid union config whose paths live under `tmp_path`."""
    path = tmp_path / "union.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return load_config(path)


def test_plan_without_override_uses_the_config_adapter(tmp_path: Path) -> None:
    cfg = config_for(tmp_path)
    result = merge.plan(cfg)
    assert result["adapter"] == str(cfg.adapter_path)
    assert result["merged"] == str(cfg.merged_path)


def test_adapter_override_changes_the_source_not_the_merged_dir(tmp_path: Path) -> None:
    cfg = config_for(tmp_path)
    union = tmp_path / "union-adapter"
    result = merge.plan(cfg, adapter=union)

    assert result["adapter"] == str(union)
    assert result["merged"] == str(cfg.merged_path)
    assert Path(result["adapter"]) != cfg.adapter_path
    # The merged dir is keyed on the run name, so an arm's merged weights cannot be hit.
    assert cfg.name in Path(result["merged"]).name


def test_build_script_embeds_the_override(tmp_path: Path) -> None:
    cfg = config_for(tmp_path)
    union = tmp_path / "union-adapter"
    script = merge.build_script(cfg, adapter=union)

    assert f'ADAPTER_DIR = Path("{union}")' in script
    assert f'MERGED_DIR = Path("{cfg.merged_path}")' in script
    assert str(cfg.adapter_path) not in script
    compile(script, "merge_script.py", "exec")


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    cfg = config_for(tmp_path)
    result = merge.run(cfg, adapter=tmp_path / "union-adapter")

    assert result["executed"] is False
    assert not cfg.output_path.exists()


def test_plan_notes_a_missing_override(tmp_path: Path) -> None:
    cfg = config_for(tmp_path)
    union = tmp_path / "union-adapter"
    result = merge.plan(cfg, adapter=union)
    assert any(str(union) in note for note in result["notes"])
    assert result["checks"]["adapter_exists"] is False


def test_execute_refuses_without_the_adapter(tmp_path: Path) -> None:
    cfg = config_for(tmp_path)
    with pytest.raises(MsftError) as caught:
        merge.run(cfg, adapter=tmp_path / "union-adapter", execute=True)
    assert "no adapter at" in str(caught.value)
