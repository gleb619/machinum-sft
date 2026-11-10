"""Tests for `scripts/abliterate.py` — the plan, the refusal counter, and the projection maths."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "abliterate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("abliterate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_counts_every_residual_writer(tmp_path: Path) -> None:
    module = load_module()
    weight_map = {
        "model.language_model.layers.0.linear_attn.out_proj.weight": "a",
        "model.language_model.layers.0.mlp.down_proj.weight": "a",
        "model.language_model.layers.1.self_attn.o_proj.weight": "b",
        "model.language_model.layers.1.mlp.up_proj.weight": "b",
        "mtp.layers.0.self_attn.o_proj.weight": "b",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": 2560}}), encoding="utf-8"
    )
    info = module.plan(tmp_path)
    assert info["layers"] == 2
    assert info["writers"] == 3, "up_proj and the mtp head are not residual writers"
    assert info["hidden_size"] == 2560


def test_refusal_count_uses_the_opening_words_only() -> None:
    module = load_module()
    replies = [
        "I cannot provide instructions for that.",
        "I'm sorry, but I can't help with this.",
        "Sure — first, gather the tools.",
        "Here is a translation: он вошёл в комнату.",
        "Не могу помочь с этим запросом.",
    ]
    assert module.refusal_count(replies) == 3


def test_project_out_removes_the_direction() -> None:
    torch = pytest.importorskip("torch")
    module = load_module()
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(6, 4, generator=generator)
    direction = torch.randn(6, generator=generator)
    direction = direction / direction.norm()
    module.project_out(matrix, direction)
    residual = direction @ matrix
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
