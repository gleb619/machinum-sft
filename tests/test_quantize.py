"""Tests for llama.cpp discovery, the GGUF argv plan and its execution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from msft import provenance, quantize
from msft.config import ModelCfg, MsftError, QuantCfg, RunConfig


@pytest.fixture()
def cfg(tmp_path: Path) -> RunConfig:
    """A config whose merged model, f16 and quantised GGUF all live under `tmp_path`."""
    return RunConfig(
        name="ru-quant",
        base_dir=str(tmp_path),
        output_dir="out",
        model=ModelCfg(id="Qwen/Qwen3.5-9B"),
        quantize=QuantCfg(method="q4_k_m", imatrix=False, output="out/model.gguf"),
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the developer's own llama.cpp checkout leak into the tests."""
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.delenv("MSFT_IMATRIX_CORPUS", raising=False)
    monkeypatch.setattr(quantize, "SEARCH_ROOTS", ())
    monkeypatch.setattr(quantize.shutil, "which", lambda name: None)


def _install_tools(
    monkeypatch: pytest.MonkeyPatch, root: Path, *, convert: bool = True, binaries: bool = True
) -> Path:
    """Fake a llama.cpp checkout plus PATH-installed binaries; return the checkout root."""
    root.mkdir(parents=True, exist_ok=True)
    if convert:
        (root / "convert_hf_to_gguf.py").write_text("print('convert')\n", encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(root))
    on_path = (
        {
            "llama-quantize": "/fake/bin/llama-quantize",
            "llama-imatrix": "/fake/bin/llama-imatrix",
        }
        if binaries
        else {}
    )
    monkeypatch.setattr(quantize.shutil, "which", lambda name: on_path.get(name))
    return root


def _merged(cfg: RunConfig) -> None:
    """Create the merged-model directory the quantizer expects as input."""
    cfg.merged_path.mkdir(parents=True, exist_ok=True)
    (cfg.merged_path / "config.json").write_text("{}", encoding="utf-8")


def test_plan_builds_exact_convert_and_quantize_argv(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without imatrix the plan is convert -> quantize, with the method upper-cased."""
    root = _install_tools(monkeypatch, tmp_path / "llama.cpp")
    plan = quantize.plan(cfg)

    f16 = tmp_path / "out" / "model-f16.gguf"
    assert plan["ready"] is True
    assert plan["missing"] == []
    assert plan["method"] == "Q4_K_M"
    assert len(plan["commands"]) == 2
    assert plan["commands"][0] == [
        sys.executable,
        str(root / "convert_hf_to_gguf.py"),
        str(cfg.merged_path),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
    ]
    partial = tmp_path / "out" / "model.gguf.partial"
    assert plan["commands"][1] == ["/fake/bin/llama-quantize", str(f16), str(partial), "Q4_K_M"]


def test_plan_orders_convert_then_imatrix_then_quantize(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Imatrix calibration sits between conversion and quantisation and feeds --imatrix."""
    cfg.quantize.imatrix = True
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("chapter text\n", encoding="utf-8")
    monkeypatch.setenv("MSFT_IMATRIX_CORPUS", str(corpus))

    plan = quantize.plan(cfg)
    f16 = tmp_path / "out" / "model-f16.gguf"
    imatrix = tmp_path / "out" / "model.imatrix.gguf"
    convert_cmd, imatrix_cmd, quantize_cmd = plan["commands"]
    assert convert_cmd[0] == sys.executable
    assert imatrix_cmd == [
        "/fake/bin/llama-imatrix",
        "-m",
        str(f16),
        "-f",
        str(corpus),
        "-o",
        str(imatrix),
    ]
    assert quantize_cmd == [
        "/fake/bin/llama-quantize",
        "--imatrix",
        str(imatrix),
        str(f16),
        str(tmp_path / "out" / "model.gguf.partial"),
        "Q4_K_M",
    ]
    assert plan["imatrix"] == {
        "enabled": True,
        "corpus": str(corpus),
        "output": str(imatrix),
    }


def test_quantize_method_is_upper_cased(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """llama-quantize wants Q5_K_M, not q5_k_m."""
    cfg.quantize.method = "q5_k_m"
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    plan = quantize.plan(cfg)
    assert plan["method"] == "Q5_K_M"
    assert plan["commands"][-1][-1] == "Q5_K_M"


def test_imatrix_without_corpus_raises(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for calibration without MSFT_IMATRIX_CORPUS is a config error, not a silent skip."""
    cfg.quantize.imatrix = True
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    with pytest.raises(MsftError, match="MSFT_IMATRIX_CORPUS"):
        quantize.plan(cfg)


def test_missing_llama_cpp_raises_with_install_hint(cfg: RunConfig) -> None:
    """No checkout and no binaries: MsftError that says how to get llama.cpp."""
    with pytest.raises(MsftError) as excinfo:
        quantize.plan(cfg)
    message = str(excinfo.value)
    assert "LLAMA_CPP_DIR" in message
    assert "convert_hf_to_gguf.py" in message
    assert "llama-quantize" in message


def test_run_without_execute_returns_plan_and_writes_nothing(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run prints the plan, returns it, and creates no files."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess must not run without execute=True")

    monkeypatch.setattr(quantize.subprocess, "run", boom)
    result = quantize.run(cfg)
    assert result["executed"] is False
    assert len(result["commands"]) == 2
    assert result["output"] == str(tmp_path / "out" / "model.gguf")
    assert not cfg.output_path.exists()


def test_run_execute_runs_commands_in_order(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --execute every planned argv runs in order and the GGUF must appear."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)
    planned = quantize.plan(cfg)["commands"]
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        _write_tool_output(argv, b"GGUF" + b"\0" * 16)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(quantize.subprocess, "run", fake_run)
    result = quantize.run(cfg, execute=True)
    # The planned tool commands run first; the provenance git probe follows them.
    assert calls[: len(planned)] == planned
    assert result["executed"] is True
    assert (tmp_path / "out" / "model.gguf").is_file()


def test_run_execute_refuses_when_tools_are_missing(cfg: RunConfig) -> None:
    """Execution without llama.cpp fails before anything is written or spawned."""
    with pytest.raises(MsftError, match="LLAMA_CPP_DIR"):
        quantize.run(cfg, execute=True)


def test_run_execute_requires_a_merged_model(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quantising a directory with no merged weights points back at `msft merge`."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    with pytest.raises(MsftError, match="msft merge"):
        quantize.run(cfg, execute=True)
    assert not cfg.output_path.exists()


def test_run_execute_raises_on_non_zero_exit(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing llama.cpp tool surfaces as MsftError with the command index."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(quantize.subprocess, "run", fake_run)
    with pytest.raises(MsftError, match="exited 1"):
        quantize.run(cfg, execute=True)


def _write_tool_output(argv: list[str], payload: bytes) -> None:
    """Write `payload` where a convert/quantize command was told to write.

    The provenance git probe is also a subprocess call; it must not be mistaken for a tool.
    """
    if "--outfile" in argv:
        target = Path(argv[argv.index("--outfile") + 1])
    elif argv and "quantize" in Path(argv[0]).name:
        target = Path(argv[-2])
    else:
        return
    target.write_bytes(payload)


def _writer(payload: bytes = b"GGUF" + b"\0" * 16):
    """A fake subprocess.run that writes `payload` where the tool was told to write."""

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _write_tool_output(argv, payload)
        return subprocess.CompletedProcess(argv, 0)

    return fake_run


def test_plan_writes_the_quantizer_output_to_a_partial(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final name must never exist until the quant is complete and checked."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    plan = quantize.plan(cfg)
    assert plan["output"] == str(tmp_path / "out" / "model.gguf")
    assert plan["partial"] == str(tmp_path / "out" / "model.gguf.partial")
    assert plan["commands"][-1][-2] == plan["partial"]


def test_run_execute_removes_a_stale_output_before_running(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A previous run's GGUF must not survive a failed quant and look current."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)
    out = tmp_path / "out" / "model.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"GGUF old model")
    partial = tmp_path / "out" / "model.gguf.partial"
    partial.write_bytes(b"GGUF stale partial")
    observed: dict[str, bool] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--outfile" in argv or (argv and "quantize" in Path(argv[0]).name):
            observed["out_existed"] = out.exists()
        _write_tool_output(argv, b"GGUF" + b"n" * 16)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(quantize.subprocess, "run", fake_run)
    result = quantize.run(cfg, execute=True)

    assert observed["out_existed"] is False
    assert result["executed"] is True
    assert out.read_bytes().startswith(b"GGUF")
    assert not partial.exists()


def test_run_execute_fails_when_the_partial_is_missing(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rc=0 without an artifact is not success; the final file must not appear."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)
    monkeypatch.setattr(
        quantize.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0)
    )
    with pytest.raises(MsftError, match="missing"):
        quantize.run(cfg, execute=True)
    assert not (tmp_path / "out" / "model.gguf").exists()


def test_run_execute_fails_when_the_partial_is_not_a_gguf(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)
    monkeypatch.setattr(quantize.subprocess, "run", _writer(b"not a gguf at all"))
    with pytest.raises(MsftError, match="not a valid GGUF"):
        quantize.run(cfg, execute=True)
    assert not (tmp_path / "out" / "model.gguf").exists()


# --- provenance sidecar ---------------------------------------------------


def test_plan_lists_the_sidecar_path(cfg: RunConfig, tmp_path: Path, monkeypatch) -> None:
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    plan = quantize.plan(cfg)
    assert plan["sidecar"] == str(tmp_path / "out" / "model.gguf.provenance.json")


def test_run_execute_writes_the_sidecar_after_the_rename(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)
    out = tmp_path / "out" / "model.gguf"
    observed: dict[str, bool] = {}
    real_replace = os.replace

    def spy(source, destination, *args, **kwargs):
        target = Path(destination)
        if target == out:
            observed["sidecar_at_rename"] = provenance.sidecar_path(target).exists()
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(quantize.os, "replace", spy)
    monkeypatch.setattr(quantize.subprocess, "run", _writer())

    result = quantize.run(cfg, execute=True)

    sidecar = provenance.sidecar_path(out)
    record, reason = provenance.load(out)
    assert result["executed"] is True
    assert observed["sidecar_at_rename"] is False
    assert sidecar.is_file() and reason == ""
    assert record["gguf"] == {
        "name": "model.gguf",
        "bytes": out.stat().st_size,
        "sha256": provenance.sha256_file(out),
    }
    assert record["config_name"] == "ru-quant"
    assert record["quantize"] == {"method": "Q4_K_M", "imatrix": False}


def test_a_failed_command_leaves_no_sidecar(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)
    monkeypatch.setattr(
        quantize.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1)
    )

    with pytest.raises(MsftError, match="exited 1"):
        quantize.run(cfg, execute=True)

    assert not (tmp_path / "out" / "model.gguf").exists()
    assert not provenance.sidecar_path(tmp_path / "out" / "model.gguf").exists()


def test_a_stale_sidecar_is_deleted_with_its_output(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sidecar must never outlive the file it describes."""
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)
    out = tmp_path / "out" / "model.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"GGUF old")
    sidecar = provenance.sidecar_path(out)
    sidecar.write_text('{"schema": 1, "stale": true}', encoding="utf-8")
    monkeypatch.setattr(quantize.subprocess, "run", _writer())

    quantize.run(cfg, execute=True)

    record, reason = provenance.load(out)
    assert reason == ""
    assert record["gguf"]["bytes"] == out.stat().st_size
    assert "stale" not in sidecar.read_text(encoding="utf-8")


def test_a_merged_directory_mutated_mid_run_raises(
    cfg: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_tools(monkeypatch, tmp_path / "llama.cpp")
    _merged(cfg)

    def mutate(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        (cfg.merged_path / "model.safetensors").write_bytes(b"new weights")
        _write_tool_output(argv, b"GGUF" + b"n" * 16)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(quantize.subprocess, "run", mutate)

    with pytest.raises(MsftError, match="merged directory changed while quantizing"):
        quantize.run(cfg, execute=True)

    assert not provenance.sidecar_path(tmp_path / "out" / "model.gguf").exists()
