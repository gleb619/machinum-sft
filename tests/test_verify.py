"""Tests for `msft verify`: merged config identity, chat template, GGUF magic and size bands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from msft import provenance, verify
from msft.cli import main
from msft.config import ModelCfg, QuantCfg, RunConfig

BASE = {
    "model_type": "qwen3",
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "vocab_size": 1000,
}


@pytest.fixture()
def cfg(tmp_path: Path) -> RunConfig:
    """A run rooted in `tmp_path` with an `out/` tree of its own."""
    return RunConfig(
        name="ru",
        base_dir=str(tmp_path),
        output_dir="out",
        model=ModelCfg(id="org/tiny"),
        quantize=QuantCfg(output="out/model-ru.gguf"),
    )


def base_cache(tmp_path: Path, **overrides: Any) -> Path:
    """Fake HF cache with a base model config and return the cache root."""
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--tiny" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    data = {**BASE, **overrides}
    (snapshot / "config.json").write_text(json.dumps(data), encoding="utf-8")
    return hub


def merged(cfg: RunConfig, *, weights_bytes: int = 200, **overrides: Any) -> None:
    """Create the merged directory: config, a weight blob and the tokenizer's template."""
    cfg.merged_path.mkdir(parents=True, exist_ok=True)
    data = {**BASE, **overrides}
    (cfg.merged_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    (cfg.merged_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}),
        encoding="utf-8",
    )
    (cfg.merged_path / "model.safetensors").write_bytes(b"w" * weights_bytes)


def ggufs(
    cfg: RunConfig,
    *,
    gguf_payload: int = 56,
    f16_payload: int | None = 146,
    magic: bytes = b"GGUF",
) -> None:
    """Write the quantised GGUF and, optionally, its f16 source."""
    out = verify.output_gguf(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(magic + b"g" * gguf_payload)
    if f16_payload is not None:
        verify.f16_gguf(cfg).write_bytes(magic + b"f" * f16_payload)


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
        "chat_template",
        "gguf",
        "provenance",
    ]
    assert result["sizes"]["ratio"] == 0.4
    assert result["sizes"]["merged_ratio"] is not None
    assert by_name(result)["model_type"]["passed"] is True
    assert "merged matches base for model_type" in by_name(result)["model_type"]["detail"]
    report = verify.format_report(result)
    assert "PASS model_type" in report
    assert "PASS chat_template" in report
    assert "gguf = 40.0% of f16" in report
    assert "WARN provenance: no provenance sidecar" in report
    assert report.rstrip().endswith("artifacts ok (6 checks, 0 failed)")


def test_model_type_mismatch_names_the_previous_model(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path, model_type="qwen3")))
    merged(cfg, model_type="llama")
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    result = verify.run(cfg)

    check = by_name(result)["model_type"]
    assert result["ok"] is False
    assert check["passed"] is False
    assert "llama" in check["detail"] and "qwen3" in check["detail"]
    assert "another model" in check["detail"]
    assert result["failures"] == ["model_type"]


def test_same_model_type_but_different_dimensions_fails(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """model_type is identical across a family; hidden_size is what caught the 0.8B-as-4B run."""
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg, hidden_size=256)
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    check = by_name(verify.run(cfg))["model_type"]

    assert check["passed"] is False
    assert "hidden_size 256 != base 512" in check["detail"]
    assert "another model" in check["detail"]


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
    assert named["chat_template"]["passed"] is False
    for name in ("merged_dir", "model_type", "tokenizer", "chat_template"):
        assert name in result["failures"]


def test_gguf_outside_the_merged_weight_band_fails(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quant 3x the merged weights is not a Q4 of those weights, whatever its f16 ratio."""
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg, gguf_payload=1000, f16_payload=3000)  # 33 % of f16, ~270 % of merged
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    check = by_name(verify.run(cfg))["gguf"]

    assert check["passed"] is False
    assert "merged weights" in check["detail"]
    assert "15% and 65%" in check["detail"]


def test_gguf_ratio_outside_the_band_fails(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg, gguf_payload=146, f16_payload=146)  # 100 %, a copy, not a Q4_K_S
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    check = by_name(verify.run(cfg))["gguf"]

    assert check["passed"] is False
    assert "100.0%" in check["detail"]
    assert "25 % and 45 %" in check["detail"]


def test_gguf_must_start_with_the_gguf_magic(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated or wrongly-named file is not a GGUF even when its size looks plausible."""
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg, magic=b"NOTG")
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    check = by_name(verify.run(cfg))["gguf"]

    assert check["passed"] is False
    assert "GGUF magic" in check["detail"]


def test_ratio_is_skipped_without_the_f16_file(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg, f16_payload=None)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    result = verify.run(cfg)

    assert result["ok"] is True
    assert result["sizes"]["ratio"] is None
    assert "f16 source absent" in by_name(result)["gguf"]["detail"]


def test_empty_chat_template_fails(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty template is the same failure as none: ollama falls back to a hand-written one."""
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    (cfg.merged_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "   "}), encoding="utf-8"
    )
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    result = verify.run(cfg)

    assert result["failures"] == ["chat_template"]
    assert "no non-empty chat_template" in by_name(result)["chat_template"]["detail"]


def test_chat_template_may_live_in_a_jinja_file(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    (cfg.merged_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (cfg.merged_path / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    assert by_name(verify.run(cfg))["chat_template"]["passed"] is True


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


def test_unknown_base_config_fails_with_the_source(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged(cfg)
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))
    monkeypatch.setattr(
        verify, "base_config", lambda config: (None, "no cached config.json for org/tiny")
    )

    check = by_name(verify.run(cfg))["model_type"]

    assert check["passed"] is False
    assert "no cached config.json" in check["detail"]


def test_base_model_type_is_read_from_the_hub_cache(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path, model_type="qwen3_5")))
    model_type, source = verify.base_model_type(cfg)
    assert model_type == "qwen3_5"
    assert "HF cache" in source


def write_sidecar(cfg: RunConfig) -> Path:
    """Write a provenance record that describes the current merged model and GGUF."""
    gguf = verify.output_gguf(cfg)
    record = provenance.build_record(
        config_name=cfg.name,
        model_id=cfg.model.id,
        revision=cfg.model.revision,
        merged=provenance.merged_snapshot(cfg.merged_path),
        f16=verify.f16_gguf(cfg),
        gguf=gguf,
        method=cfg.quantize.method.upper(),
        imatrix=False,
        llama_cpp={"commit": None, "source": None},
    )
    return provenance.write(gguf, record)


def ready(cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Base cache + merged dir + GGUF, with the tokenizer check stubbed out."""
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path)))
    merged(cfg)
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))


def test_a_matching_sidecar_passes_without_a_warning(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(cfg, tmp_path, monkeypatch)
    write_sidecar(cfg)

    result = verify.run(cfg)

    check = by_name(result)["provenance"]
    assert result["ok"] is True
    assert result["warnings"] == []
    assert check["passed"] is True
    assert check.get("warning") is not True
    assert "matches the artifact and the merged model" in check["detail"]


def test_an_absent_sidecar_is_a_warning_not_a_failure(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(cfg, tmp_path, monkeypatch)

    result = verify.run(cfg)

    check = by_name(result)["provenance"]
    assert result["ok"] is True
    assert check["passed"] is True
    assert check["warning"] is True
    assert result["warnings"] == [check["detail"]]
    assert "provenance not checked" in check["detail"]


def test_a_mutated_config_json_fails_and_names_both_digests(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(cfg, tmp_path, monkeypatch)
    sidecar = write_sidecar(cfg)
    recorded = json.loads(sidecar.read_text(encoding="utf-8"))["merged"]["config_sha256"]
    config = cfg.merged_path / "config.json"
    config.write_text(json.dumps({**BASE}) + "\n", encoding="utf-8")
    actual = provenance.sha256_file(config)

    result = verify.run(cfg)

    check = by_name(result)["provenance"]
    assert result["ok"] is False
    assert check["passed"] is False
    assert "provenance: merged config.json sha256" in check["detail"]
    assert f"{recorded[:8]}…" in check["detail"]
    assert f"{actual[:8]}…" in check["detail"]
    assert "re-run quantize" in check["detail"]


def test_changed_weight_files_fail(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(cfg, tmp_path, monkeypatch)
    write_sidecar(cfg)
    (cfg.merged_path / "model-00002-of-00002.safetensors").write_bytes(b"extra shard")

    check = by_name(verify.run(cfg))["provenance"]

    assert check["passed"] is False
    assert "merged weight files differ" in check["detail"]


def test_a_corrupt_sidecar_fails(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(cfg, tmp_path, monkeypatch)
    provenance.sidecar_path(verify.output_gguf(cfg)).write_text("{oops", encoding="utf-8")

    result = verify.run(cfg)

    check = by_name(result)["provenance"]
    assert result["ok"] is False
    assert check["passed"] is False
    assert "corrupt" in check["detail"]


def test_require_provenance_turns_absence_into_a_failure(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(cfg, tmp_path, monkeypatch)

    result = verify.run(cfg, require_provenance=True)

    assert result["ok"] is False
    assert result["failures"] == ["provenance"]
    assert "--require-provenance" in by_name(result)["provenance"]["detail"]


def test_deep_catches_a_flipped_byte_in_the_gguf(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(cfg, tmp_path, monkeypatch)
    write_sidecar(cfg)
    gguf = verify.output_gguf(cfg)
    payload = bytearray(gguf.read_bytes())
    payload[-1] ^= 0xFF  # same size, different content
    gguf.write_bytes(payload)

    shallow = verify.run(cfg)
    deep = verify.run(cfg, deep=True)

    assert shallow["ok"] is True
    assert deep["ok"] is False
    assert "sha256" in by_name(deep)["provenance"]["detail"]


def cli_config(tmp_path: Path) -> Path:
    """A config the CLI can load, rooted at `tmp_path`."""
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: ru\n"
        'base_dir: "."\n'
        "model:\n"
        "  id: org/tiny\n"
        "quantize:\n"
        "  output: out/model-ru.gguf\n",
        encoding="utf-8",
    )
    return path


def test_cli_verify_passes_and_prints_one_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = cli_config(tmp_path)
    cfg = RunConfig(
        name="ru",
        base_dir=str(tmp_path),
        model=ModelCfg(id="org/tiny"),
        quantize=QuantCfg(output="out/model-ru.gguf"),
    )
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
    cfg = RunConfig(
        name="ru",
        base_dir=str(tmp_path),
        model=ModelCfg(id="org/tiny"),
        quantize=QuantCfg(output="out/model-ru.gguf"),
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(base_cache(tmp_path, model_type="qwen3")))
    merged(cfg, model_type="llama")
    ggufs(cfg)
    monkeypatch.setattr(verify, "tokenizer_loads", lambda path: (True, "ok"))

    code = main(["--config", str(config_path), "verify"])

    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL model_type" in captured.out
    assert "artifacts FAILED" in captured.out
