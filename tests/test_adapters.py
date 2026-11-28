"""Tests for the LoRA delta merge: validation, the concat math and the generated script.

No torch, no peft, no safetensors: the header reader and the math are pure Python, which is
exactly what has to hold before a 0.8B model is loaded on the box.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from msft.adapters import (
    CONFIG_KEYS,
    DEFAULT_PROMPT,
    Factors,
    build_script,
    concatenate,
    first_prompt,
    merged_config,
    plan,
    read_header,
    run,
)
from msft.config import MsftError

BASE = "Qwen/Qwen3.5-0.8B"


def write_safetensors(
    path: Path, tensors: dict[str, list[list[float]]], dtype: str = "F32"
) -> Path:
    """Write a minimal valid safetensors file: 8-byte header length, JSON header, float data."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for key in sorted(tensors):
        matrix = tensors[key]
        values = [float(item) for row in matrix for item in row]
        payload = struct.pack(f"<{len(values)}f", *values) if values else b""
        header[key] = {
            "dtype": dtype,
            "shape": [len(matrix), len(matrix[0]) if matrix else 0],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        blobs.append(payload)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
    return path


def adapter_config(**overrides: object) -> dict:
    """A peft-shaped LoRA config, as peft writes it for this base."""
    config: dict[str, object] = {
        "base_model_name_or_path": BASE,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "modules_to_save": [],
        "peft_type": "LORA",
        "r": 2,
        "revision": None,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config.update(overrides)
    return config


def module_tensors(
    modules: tuple[str, ...] = ("q_proj", "v_proj"),
    *,
    r: int = 2,
    d_in: int = 3,
    d_out: int = 4,
    seed: float = 0.25,
) -> dict[str, list[list[float]]]:
    """One lora_A/lora_B pair per module, deterministic and distinct per adapter."""
    tensors: dict[str, list[list[float]]] = {}
    for position, module in enumerate(modules):
        key = f"base_model.model.model.layers.0.self_attn.{module}"
        tensors[f"{key}.lora_A.weight"] = [
            [seed * (position + 1) + row for row in range(d_in)] for _ in range(r)
        ]
        tensors[f"{key}.lora_B.weight"] = [
            [seed * (column + 1) for column in range(r)] for _ in range(d_out)
        ]
    return tensors


def write_adapter(
    root: Path,
    *,
    config: dict | None = None,
    tensors: dict[str, list[list[float]]] | None = None,
    dtype: str = "F32",
) -> Path:
    """Write an adapter dir with a config and safetensors weights."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(
        json.dumps(config or adapter_config(), indent=2) + "\n", encoding="utf-8"
    )
    write_safetensors(root / "adapter_model.safetensors", tensors or module_tensors(), dtype)
    return root


def pair(tmp_path: Path, **b_overrides: object) -> tuple[Path, Path]:
    """A comparable adapter pair, with `b` optionally made incomparable."""
    first = write_adapter(tmp_path / "a")
    second_config = adapter_config(**b_overrides) if b_overrides else None
    second = write_adapter(
        tmp_path / "b",
        config=second_config,
        tensors=module_tensors(seed=0.5),
        dtype=str(b_overrides.get("dtype", "F32")),
    )
    return first, second


# ----------------------------------------------------------------------- validation


def test_plan_reports_the_shape_table(tmp_path: Path) -> None:
    first, second = pair(tmp_path)
    result = plan(first, second, tmp_path / "union")

    assert result["r"] == 2
    assert result["alpha"] == 4
    assert result["r_out"] == 4
    assert result["alpha_out"] == 4
    assert result["scale_in"] == 2.0
    assert result["scale_out"] == 1.0
    assert sorted(module["module"].rsplit(".", 1)[-1] for module in result["modules"]) == [
        "q_proj",
        "v_proj",
    ]
    for module in result["modules"]:
        assert (module["d_in"], module["d_out"], module["r_out"]) == (3, 4, 4)
        assert module["a_shape"] == [2, 3]
        assert module["b_shape"] == [4, 2]
    assert result["base_model"] == BASE
    assert result["executed"] is False
    assert not (tmp_path / "union").exists()


def test_sum_mode_doubles_the_alpha(tmp_path: Path) -> None:
    first, second = pair(tmp_path)
    result = plan(first, second, tmp_path / "union", weights="sum")
    assert result["alpha_out"] == 8
    assert result["r_out"] == 4
    assert result["scale_out"] == 2.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_model_name_or_path": "Other/Model"}, "base_model_name_or_path"),
        ({"revision": "deadbeef"}, "revision"),
        ({"r": 4}, "'r'"),
        ({"lora_alpha": 8}, "lora_alpha"),
        ({"target_modules": ["q_proj"]}, "target_modules"),
        ({"use_dora": True}, "DoRA"),
        ({"use_rslora": True}, "rsLoRA"),
        ({"bias": "all"}, "bias"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"rank_pattern": {"q_proj": 4}}, "rank_pattern"),
    ],
)
def test_incomparable_adapters_are_refused(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    first = write_adapter(tmp_path / "a")
    second = write_adapter(tmp_path / "b", config=adapter_config(**overrides))
    with pytest.raises(MsftError) as caught:
        plan(first, second, tmp_path / "union")
    assert message in str(caught.value)


def test_key_set_mismatch_is_refused(tmp_path: Path) -> None:
    first = write_adapter(tmp_path / "a")
    second = write_adapter(
        tmp_path / "b",
        tensors=module_tensors(("q_proj", "v_proj", "o_proj"), seed=0.5),
    )
    with pytest.raises(MsftError) as caught:
        plan(first, second, tmp_path / "union")
    assert "tensor key sets differ" in str(caught.value)


def test_shape_mismatch_is_refused(tmp_path: Path) -> None:
    first = write_adapter(tmp_path / "a")
    second = write_adapter(tmp_path / "b", tensors=module_tensors(d_in=5, seed=0.5))
    with pytest.raises(MsftError) as caught:
        plan(first, second, tmp_path / "union")
    assert "shape of" in str(caught.value)


def test_dtype_mismatch_is_refused(tmp_path: Path) -> None:
    first = write_adapter(tmp_path / "a")
    second = write_adapter(tmp_path / "b", tensors=module_tensors(seed=0.5), dtype="F16")
    with pytest.raises(MsftError) as caught:
        plan(first, second, tmp_path / "union")
    assert "dtype of" in str(caught.value)


def test_non_lora_tensors_are_refused(tmp_path: Path) -> None:
    extra = {"base_model.model.lm_head.weight": [[1.0, 2.0], [3.0, 4.0]]}
    first = write_adapter(tmp_path / "a", tensors=dict(module_tensors(), **extra))
    second = write_adapter(tmp_path / "b", tensors=dict(module_tensors(seed=0.5), **extra))
    with pytest.raises(MsftError) as caught:
        plan(first, second, tmp_path / "union")
    assert "not a lora_A/lora_B tensor" in str(caught.value)


def test_missing_adapter_config_is_loud(tmp_path: Path) -> None:
    first = write_adapter(tmp_path / "a")
    (tmp_path / "b").mkdir()
    with pytest.raises(MsftError) as caught:
        plan(first, tmp_path / "b", tmp_path / "union")
    assert "adapter_config.json is missing" in str(caught.value)


def test_missing_weights_are_a_note_not_a_shape_table(tmp_path: Path) -> None:
    first = write_adapter(tmp_path / "a")
    second = tmp_path / "b"
    second.mkdir()
    (second / "adapter_config.json").write_text(json.dumps(adapter_config()) + "\n")
    result = plan(first, second, tmp_path / "union")
    assert result["modules"] == []
    assert any("weights missing" in note for note in result["notes"])


def test_read_header_needs_no_torch(tmp_path: Path) -> None:
    path = write_safetensors(tmp_path / "x.safetensors", {"a": [[1.0, 2.0], [3.0, 4.0]]})
    assert read_header(path) == {"a": {"dtype": "F32", "shape": [2, 2]}}


# ------------------------------------------------------------------------- the math


def _max_abs_diff(got: list[list[float]], want: list[list[float]]) -> float:
    """Largest elementwise difference between two dense matrices."""
    return max(
        abs(got[i][j] - want[i][j]) for i in range(len(got)) for j in range(len(got[0]))
    )


def naive_delta(
    a: list[list[float]], b: list[list[float]], alpha: int, r: int
) -> list[list[float]]:
    """Reference s * B @ A in plain Python, independent of `Factors.delta`."""
    scale = alpha / r
    return [
        [scale * sum(b[i][k] * a[k][j] for k in range(r)) for j in range(len(a[0]))]
        for i in range(len(b))
    ]


@pytest.mark.parametrize(
    ("a_a", "b_a", "a_b", "b_b", "r", "alpha"),
    [
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.0, 0.5], [0.25, -1.0], [2.0, 0.0]],
            [[0.5, -1.0, 2.5], [1.5, 0.0, -2.0]],
            [[0.0, 1.0], [-2.0, 0.5], [1.0, 1.0]],
            2,
            4,
        ),
        (
            [[1.0, -2.0], [0.5, 3.0]],
            [[2.0, 1.0], [-1.0, 0.5], [0.0, 2.0], [1.5, -0.5]],
            [[-0.5, 2.0], [3.0, 1.0]],
            [[1.0, 0.0], [0.5, -2.0], [2.0, 1.0], [-1.0, 0.5]],
            2,
            3,
        ),
    ],
)
def test_concatenation_mean_is_exact(
    a_a: list, b_a: list, a_b: list, b_b: list, r: int, alpha: int
) -> None:
    first = Factors(a=a_a, b=b_a, r=r, alpha=alpha)
    second = Factors(a=a_b, b=b_b, r=r, alpha=alpha)
    union = concatenate(first, second, mode="mean")

    assert (union.r, union.alpha) == (2 * r, alpha)
    got = union.delta()
    want = naive_delta(first.a + second.a, union.b, union.alpha, union.r)
    expected = [
        [(x + y) / 2 for x, y in zip(row_a, row_b, strict=True)]
        for row_a, row_b in zip(first.delta(), second.delta(), strict=True)
    ]
    assert _max_abs_diff(got, expected) < 1e-12
    assert _max_abs_diff(got, want) < 1e-12


def test_concatenation_sum_is_the_full_delta_sum() -> None:
    first = Factors(a=[[1.0, 2.0], [0.0, 1.0]], b=[[1.0, 0.5], [2.0, -1.0]], r=2, alpha=4)
    second = Factors(a=[[0.5, -1.0], [2.0, 0.0]], b=[[-1.0, 1.0], [0.5, 2.0]], r=2, alpha=4)
    union = concatenate(first, second, mode="sum")

    got = union.delta()
    expected = [
        [x + y for x, y in zip(row_a, row_b, strict=True)]
        for row_a, row_b in zip(first.delta(), second.delta(), strict=True)
    ]
    assert _max_abs_diff(got, expected) < 1e-12


def test_concatenate_refuses_what_it_cannot_express() -> None:
    base = Factors(a=[[1.0, 2.0], [3.0, 4.0]], b=[[1.0, 0.0], [0.0, 1.0]], r=2, alpha=4)
    with pytest.raises(MsftError):
        concatenate(base, Factors(a=[[1.0, 2.0]], b=[[1.0, 0.0]], r=1, alpha=2))
    with pytest.raises(MsftError):
        concatenate(base, Factors(a=[[1.0, 2.0], [3.0, 4.0]], b=[[1.0, 0.0]], r=2, alpha=4))
    with pytest.raises(MsftError):
        concatenate(base, base, mode="ties")


# --------------------------------------------------------------- the written config


def test_merged_config_rewrites_rank_and_alpha() -> None:
    config = adapter_config()
    mean = merged_config(config, "mean")
    assert (mean["r"], mean["lora_alpha"]) == (4, 4)
    assert set(mean) <= set(CONFIG_KEYS)
    assert list(mean) == sorted(mean)

    summed = merged_config(config, "sum")
    assert (summed["r"], summed["lora_alpha"]) == (4, 8)


def test_merged_config_drops_unknown_keys_and_keeps_peft_ones() -> None:
    config = dict(adapter_config(), some_future_key=1)
    merged = merged_config(config, "mean")
    assert "some_future_key" not in merged
    assert merged["peft_type"] == "LORA"
    assert merged["target_modules"] == ["q_proj", "v_proj"]
    assert merged["modules_to_save"] == []


def test_plan_notes_dropped_config_keys(tmp_path: Path) -> None:
    first = write_adapter(tmp_path / "a", config=dict(adapter_config(), future_key=1))
    second = write_adapter(tmp_path / "b", config=dict(adapter_config(), future_key=1))
    result = plan(first, second, tmp_path / "union")
    assert any("future_key" in note for note in result["notes"])


# ---------------------------------------------------------------------- the script


def test_build_script_embeds_the_paths_and_the_self_checks(tmp_path: Path) -> None:
    script = build_script(tmp_path / "a", tmp_path / "b", tmp_path / "union")
    assert str(tmp_path / "a") in script
    assert str(tmp_path / "union") in script
    assert "self-check 1:" in script
    assert "self-check 2:" in script
    assert "self-check 3:" in script
    assert "enable_thinking=False" in script
    compile(script, "merge_script.py", "exec")


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    first, second = pair(tmp_path)
    result = run(first, second, tmp_path / "union")
    assert result["executed"] is False
    assert not (tmp_path / "union").exists()


def test_execute_without_weights_refuses(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    for folder in (first, second):
        folder.mkdir()
        (folder / "adapter_config.json").write_text(json.dumps(adapter_config()) + "\n")
    with pytest.raises(MsftError) as caught:
        run(first, second, tmp_path / "union", execute=True)
    assert "no adapter weights" in str(caught.value)


def test_first_prompt_reads_the_cases_file(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"case": "x", "prompt": "Kael waited."}) + "\n"
        + json.dumps({"case": "y", "prompt": "Later."}) + "\n",
        encoding="utf-8",
    )
    assert first_prompt(cases) == "Kael waited."
    assert first_prompt(None) == DEFAULT_PROMPT
    cases.write_text(
        json.dumps(
            {"case": "m", "messages": [{"role": "user", "content": "From messages."}]}
        )
        + "\n",
        encoding="utf-8",
    )
    assert first_prompt(cases) == "From messages."


def test_first_prompt_without_a_prompt_is_loud(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"case": "x", "task": "ner"}) + "\n", encoding="utf-8")
    with pytest.raises(MsftError):
        first_prompt(cases)
    with pytest.raises(MsftError):
        first_prompt(tmp_path / "absent.jsonl")
