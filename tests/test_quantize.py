"""Tests for llama.cpp discovery, the GGUF argv plan and its execution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from msft import quantize
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
        # convert_hf_to_gguf.py names its output; llama-quantize takes it as the last arg.
        target = Path(
            argv[argv.index("--outfile") + 1] if "--outfile" in argv else argv[-2]
        )
        target.write_bytes(b"GGUF" + b"\0" * 16)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(quantize.subprocess, "run", fake_run)
    result = quantize.run(cfg, execute=True)
    assert calls == planned
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


def _writer(payload: bytes = b"GGUF" + b"\0" * 16):
    """A fake subprocess.run that writes `payload` where the tool was told to write."""

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        target = Path(
            argv[argv.index("--outfile") + 1] if "--outfile" in argv else argv[-2]
        )
        target.write_bytes(payload)
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
        observed["out_existed"] = out.exists()
        target = Path(
            argv[argv.index("--outfile") + 1] if "--outfile" in argv else argv[-2]
        )
        target.write_bytes(b"GGUF" + b"n" * 16)
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
