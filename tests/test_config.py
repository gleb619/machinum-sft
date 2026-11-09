"""Config loading: YAML type coercion, required keys, enumerations and redaction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from msft.config import MsftError, load_config, redact

MINIMAL = """
name: tiny
model:
  id: org/tiny
quantize:
  output: out/model-tiny.gguf
"""


def write(tmp_path: Path, text: str = MINIMAL) -> Path:
    """Write a config and return its path."""
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- required keys and types ---------------------------------------------


def test_missing_model_section_is_an_error(tmp_path: Path) -> None:
    text = MINIMAL.replace("model:\n  id: org/tiny\n", "")
    with pytest.raises(MsftError, match="missing required key 'id'"):
        load_config(write(tmp_path, text))


def test_scientific_notation_without_a_dot_is_coerced_to_float(tmp_path: Path) -> None:
    """PyYAML 1.1 reads `5e-5` as a string; the trainer would then get `lr` as str."""
    path = write(tmp_path, MINIMAL + "train:\n  lr: 5e-5\n")
    cfg = load_config(path)
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.lr == 5e-5


@pytest.mark.parametrize("value", ['"false"', "0", '""'])
def test_string_bool_is_rejected(tmp_path: Path, value: str) -> None:
    path = write(tmp_path, MINIMAL + f"data:\n  dedupe: {value}\n")
    with pytest.raises(MsftError, match="dedupe"):
        load_config(path)


def test_non_numeric_number_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL + "train:\n  lr: fast\n")
    with pytest.raises(MsftError, match="lr"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL + "train:\n  learning_rate: 0.1\n")
    with pytest.raises(MsftError, match="unknown key"):
        load_config(path)


# --- ranges and enumerations ---------------------------------------------


@pytest.mark.parametrize(
    "snippet, name",
    [
        ("train:\n  epochs: 0\n", "epochs"),
        ("train:\n  grad_accum: 0\n", "grad_accum"),
        ("train:\n  per_device_batch: 0\n", "per_device_batch"),
        ("train:\n  lr: 0\n", "lr"),
        ("data:\n  val_ratio: 1.5\n", "val_ratio"),
    ],
)
def test_non_positive_or_out_of_range_numbers_are_rejected(
    tmp_path: Path, snippet: str, name: str
) -> None:
    with pytest.raises(MsftError, match=name):
        load_config(write(tmp_path, MINIMAL + snippet))


def test_unknown_dtype_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL.replace("  id: org/tiny\n", "  id: org/tiny\n  dtype: int8\n"))
    with pytest.raises(MsftError, match="dtype"):
        load_config(path)


def test_unknown_scheduler_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL + "train:\n  scheduler: fastish\n")
    with pytest.raises(MsftError, match="scheduler"):
        load_config(path)


def test_unknown_optimizer_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL + "train:\n  optim: magic\n")
    with pytest.raises(MsftError, match="optim"):
        load_config(path)


def test_unknown_quantize_method_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "name: tiny\nmodel:\n  id: org/tiny\nquantize:\n  method: q9\n")
    with pytest.raises(MsftError, match="method"):
        load_config(path)


def test_a_name_that_is_a_path_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, MINIMAL.replace("name: tiny", "name: ../x"))
    with pytest.raises(MsftError, match="name"):
        load_config(path)


def test_quantize_output_must_name_the_run(tmp_path: Path) -> None:
    """Two configs writing one GGUF is how a 0.8B model shipped as model-4b.gguf."""
    text = MINIMAL.replace("out/model-tiny.gguf", "out/model.gguf")
    with pytest.raises(MsftError, match="does not contain the run name"):
        load_config(write(tmp_path, text))


# --- redaction ------------------------------------------------------------


def test_a_sha256_survives_redaction() -> None:
    digest = hashlib.sha256(b"chapter text").hexdigest()
    assert redact(f"sha256 {digest}") == f"sha256 {digest}"


def test_a_hf_token_is_masked() -> None:
    token = "hf_abcdefghijklmnopqrstuvwxyz"
    text = redact(f"Authorization: Bearer {token}")
    assert token not in text
    assert "<redacted>" in text
