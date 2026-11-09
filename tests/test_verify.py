"""Tests for `msft verify`: merged model_type, tokenizer, GGUF size and f16 ratio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from msft import verify
from msft.cli import main
from msft.config import ModelCfg, QuantCfg, RunConfig


@pytest.fixture()
def cfg(tmp_path: Path) -> RunConfig:
    """A run rooted in `tmp_path` with an `out/` tree of its own."""
    return RunConfig(
        name="ru",
        base_dir=str(tmp_path),
        output_dir="out",
        model=ModelCfg(id="org/tiny"),
        quantize=QuantCfg(output="out/model.gguf"),
    )


@pytest.fixture(autouse=True)
def small_gguf_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the 500 MB floor out of a unit test; the production constant is unchanged."""
    monkeypatch.setattr(verify, "MIN_GGUF_BYTES", 10)


def base_cache(tmp_path: Path, model_type: str = "qwen3") -> Path:
    """Fake HF cache with a base model config and return the cache root."""
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--tiny" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return hub


def merged(cfg: RunConfig, model_type: str = "qwen3") -> None:
    """Create the merged directory with a model config."""
    cfg.merged_path.mkdir(parents=True, exist_ok=True)
    (cfg.merged_path / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )


def ggufs(cfg: RunConfig, *, gguf_bytes: int = 30, f16_bytes: int | None = 100) -> None:
    """Write the quantised GGUF and, optionally, its f16 source."""
    out = Path(cfg.base_dir) / cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.gguf").write_bytes(b"g" * gguf_bytes)
    if f16_bytes is not None:
        (out / "model-f16.gguf").write_bytes(b"f" * f16_bytes)


def by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index checks by name."""
    return {check["name"]: check for check in result["checks"]}


def test_all_good_artifacts_pass(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "tokenizer loads"))

    result = verify.run(cfg)

    assert result["ok"] is True
    assert result["failures"] == []
    assert [check["name"] for check in result["checks"]] == [
        "merged_dir",
        "model_type",
        "tokenizer",
        "gguf",
    ]
    assert result["sizes"]["ratio"] == 0.3
    assert by_name(result)["model_type"]["passed"] is True
    assert "qwen3" in by_name(result)["model_type"]["detail"]
    report = verify.format_report(result)
    assert "PASS model_type" in report
    assert "gguf = 30.0% of f16" in report
    assert report.rstrip().endswith("artifacts ok (4 checks, 0 failed)")


def test_model_type_mismatch_names_the_previous_model(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path, "qwen3")))
    merged(cfg, "llama")
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    result = verify.run(cfg)

    check = by_name(result)["model_type"]
    assert result["ok"] is False
    assert check["passed"] is False
    assert "llama" in check["detail"] and "qwen3" in check["detail"]
    assert "another model" in check["detail"]
    assert result["failures"] == ["model_type"]


def test_missing_merged_directory_fails_named_checks(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    ggufs(cfg)

    result = verify.run(cfg)

    named = by_name(result)
    assert named["merged_dir"]["passed"] is False
    assert "msft merge --execute" in named["merged_dir"]["detail"]
    assert named["model_type"]["passed"] is False
    assert named["tokenizer"]["passed"] is False
    assert "merged_dir" in result["failures"]
    assert "model_type" in result["failures"]
    assert "tokenizer" in result["failures"]


def test_gguf_below_the_floor_fails(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg, gguf_bytes=5, f16_bytes=100)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    check = by_name(verify.run(cfg))["gguf"]

    assert check["passed"] is False
    assert "at least 500 MB" in check["detail"]


def test_gguf_ratio_outside_the_band_fails(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg, gguf_bytes=50, f16_bytes=100)  # 50 %, a Q8, not a Q4_K_S
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    check = by_name(verify.run(cfg))["gguf"]

    assert check["passed"] is False
    assert "50.0%" in check["detail"]
    assert "25 % and 45 %" in check["detail"]


def test_ratio_is_skipped_without_the_f16_file(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg, gguf_bytes=30, f16_bytes=None)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    result = verify.run(cfg)

    assert result["ok"] is True
    assert result["sizes"]["ratio"] is None
    assert "f16 source absent" in by_name(result)["gguf"]["detail"]


def test_tokenizer_failure_is_named(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg)
    monkeypatch.setattr(
        verify, "tokenizer_loads", lambda path: (False, "Tokenizer class does not exist")
    )

    result = verify.run(cfg)

    assert result["ok"] is False
    assert result["failures"] == ["tokenizer"]
    assert "Tokenizer class does not exist" in by_name(result)["tokenizer"]["detail"]


def test_unknown_base_model_type_fails_with_the_source(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged(cfg)
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))
    monkeypatch.setattr(
        verify, "base_model_type", lambda config: (None, "no cached config.json for org/tiny")
    )

    check = by_name(verify.run(cfg))["model_type"]

    assert check["passed"] is False
    assert "no cached config.json" in check["detail"]


def test_base_model_type_is_read_from_the_hub_cache(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path, "qwen3_5")))
    model_type, source = verify.base_model_type(cfg)
    assert model_type == "qwen3_5"
    assert "HF cache" in source


def cli_config(tmp_path: Path) -> Path:
    """A config the CLI can load, rooted at `tmp_path`."""
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: ru\n"
        'base_dir: "."\n'
        "model:\n"
        "  id: org/tiny\n"
        "quantize:\n"
        "  output: out/model.gguf\n",
        encoding="utf-8",
    )
    return path


def test_cli_verify_passes_and_prints_one_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = cli_config(tmp_path)
    cfg = RunConfig(name="ru", base_dir=str(tmp_path), model=ModelCfg(id="org/tiny"))
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    code = main(["--json", "--config", str(config_path), "verify"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["failures"] == []


def test_cli_verify_exits_one_on_a_named_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = cli_config(tmp_path)
    cfg = RunConfig(name="ru", base_dir=str(tmp_path), model=ModelCfg(id="org/tiny"))
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path, "qwen3")))
    merged(cfg, "llama")
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    code = main(["--config", str(config_path), "verify"])

    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL model_type" in captured.out
    assert "artifacts FAILED" in captured.out
