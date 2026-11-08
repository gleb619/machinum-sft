"""Tests for msft.data: reading, normalisation, filtering, dedupe and deterministic splits."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msft.config import DataCfg, MsftError, RunConfig  # noqa: E402
from msft.data import (  # noqa: E402
    build_dataset,
    iter_records,
    normalize_record,
    render_chat,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MESSAGES = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]


def cfg_for(tmp_path: Path, input_path: Path | str = FIXTURES, **data: Any) -> RunConfig:
    """Build a RunConfig that reads input_path and writes under tmp_path."""
    return RunConfig(
        name="test",
        data=DataCfg(input=str(input_path), **data),
        base_dir=str(tmp_path),
        output_dir="out",
    )


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as JSONL, UTF-8, one object per line."""
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return path


def digest_of(record: dict) -> str:
    """Recompute the documented dedupe/split digest independently of the module."""
    payload = json.dumps(
        record["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file back into dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- reading


def test_iter_records_reads_jsonl_file() -> None:
    records = list(iter_records(FIXTURES / "book_pairs.jsonl"))
    assert len(records) == 4
    assert records[0]["source"] == "translate"
    assert "reply" in records[0]


def test_iter_records_reads_json_array() -> None:
    records = list(iter_records(FIXTURES / "array.json"))
    assert len(records) == 3
    assert records[0]["source"] == "array/translate"


def test_iter_records_walks_directory_recursively() -> None:
    records = list(iter_records(FIXTURES))
    assert len(records) == 17
    assert all(isinstance(r, dict) for r in records)


def test_iter_records_skips_other_suffixes(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not a record", encoding="utf-8")
    write_jsonl(tmp_path / "one.jsonl", [{"user": "u", "reply": "a"}])
    assert len(list(iter_records(tmp_path))) == 1


def test_iter_records_rejects_unknown_file_type(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(MsftError, match=r"must be \.jsonl or \.json"):
        list(iter_records(path))


def test_iter_records_reports_bad_json_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"user": "u", "reply": "a"}\n{broken\n', encoding="utf-8")
    with pytest.raises(MsftError, match=r"bad\.jsonl:2: invalid JSON"):
        list(iter_records(path))


def test_iter_records_missing_path(tmp_path: Path) -> None:
    with pytest.raises(MsftError, match="data input not found"):
        list(iter_records(tmp_path / "nope"))


# ----------------------------------------------------------------------- normalising


def test_normalize_fields_builds_messages() -> None:
    record = normalize_record(
        {"system": "s", "user": "u", "reply": "a", "source": "book", "prompt_hash": "deadbeef"}
    )
    assert record == {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        "source": "book",
    }


def test_normalize_without_system_and_without_source() -> None:
    assert normalize_record({"user": "u", "reply": "a"}) == {
        "messages": MESSAGES,
        "source": "",
    }


def test_normalize_messages_drops_empty_and_unknown_turns() -> None:
    raw = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "   "},
            {"role": "tool", "content": "call"},
            {"role": " ASSISTANT ", "content": "a"},
        ]
    }
    assert normalize_record(raw) == {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        "source": "",
    }


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"system": "s"},
        {"user": "u"},
        {"reply": "a"},
        {"messages": []},
        {"messages": "not a list"},
        {"user": "u", "reply": "  \n\t "},
        {"user": "u", "reply": "```python\n```"},
        {"notes": "unknown shape"},
    ],
)
def test_normalize_rejects_unusable_records(raw: dict) -> None:
    assert normalize_record(raw) is None


def test_normalize_rejects_non_dict() -> None:
    assert normalize_record(["user", "assistant"]) is None  # type: ignore[arg-type]


def test_normalize_keeps_fence_with_code_body() -> None:
    raw = {"user": "u", "reply": "```python\nprint('hi')\n```"}
    assert normalize_record(raw) is not None


def test_normalize_falls_back_to_fields_when_messages_are_empty() -> None:
    raw = {"messages": [], "user": "u", "reply": "a"}
    assert normalize_record(raw) == {"messages": MESSAGES, "source": ""}


# -------------------------------------------------------------------------- stats


def test_build_dataset_dry_run_stats(tmp_path: Path) -> None:
    cfg = cfg_for(tmp_path, val_ratio=0.34)
    stats = build_dataset(cfg)
    assert stats.records_in == 17
    assert stats.records_out == 8
    assert stats.dropped == {
        "code_fence": 1,
        "duplicate": 1,
        "no_assistant": 4,
        "no_user": 2,
        "too_short": 1,
    }
    assert stats.train + stats.val == stats.records_out
    assert stats.train > 0
    assert stats.val > 0
    assert stats.approx_tokens == math.ceil(stats.chars / 3.2)
    assert stats.approx_tokens_method == "ceil(chars/3.2)"
    assert sum(stats.sources.values()) == stats.records_out
    assert not cfg.dataset_path.exists()


def test_build_dataset_execute_writes_same_stats(tmp_path: Path) -> None:
    cfg = cfg_for(tmp_path, val_ratio=0.34)
    dry = build_dataset(cfg)
    stats = build_dataset(cfg, execute=True)
    assert stats.to_dict() == dry.to_dict()

    out = cfg.dataset_path
    train = read_jsonl(out / "train.jsonl")
    val = read_jsonl(out / "val.jsonl")
    assert len(train) == stats.train
    assert len(val) == stats.val
    assert len(train) + len(val) == stats.records_out == 8

    on_disk = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert on_disk == stats.to_dict()


def test_split_is_deterministic_by_hash(tmp_path: Path) -> None:
    cfg = cfg_for(tmp_path, val_ratio=0.34)
    build_dataset(cfg, execute=True)
    train = read_jsonl(cfg.dataset_path / "train.jsonl")
    val = read_jsonl(cfg.dataset_path / "val.jsonl")

    for bucket, in_val in ((train, False), (val, True)):
        digests = [digest_of(record) for record in bucket]
        assert digests == sorted(digests)
        for digest in digests:
            assert (int(digest[:8], 16) / float(1 << 32) < 0.34) is in_val


def test_build_is_byte_identical_across_runs(tmp_path: Path) -> None:
    first = cfg_for(tmp_path, val_ratio=0.34)
    first.output_dir = "run_a"
    second = cfg_for(tmp_path, val_ratio=0.34)
    second.output_dir = "run_b"
    build_dataset(first, execute=True)
    build_dataset(second, execute=True)

    for name in ("train.jsonl", "val.jsonl", "stats.json"):
        left = (first.dataset_path / name).read_bytes()
        right = (second.dataset_path / name).read_bytes()
        assert left == right


def test_val_ratio_extremes(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    records = [{"user": f"u{i}", "reply": "a" * 80} for i in range(6)]
    write_jsonl(data_dir / "records.jsonl", records)

    all_train = build_dataset(cfg_for(tmp_path, data_dir, val_ratio=0.0))
    assert (all_train.train, all_train.val) == (6, 0)

    all_val = build_dataset(cfg_for(tmp_path, data_dir, val_ratio=1.0))
    assert (all_val.train, all_val.val) == (0, 6)


def test_dedupe_uses_normalised_messages(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    good = {"user": "u", "reply": "a" * 80, "source": "first"}
    write_jsonl(
        data_dir / "records.jsonl",
        [good, {**good, "source": "second"}, {"user": "v", "reply": "b" * 80}],
    )

    deduped = build_dataset(cfg_for(tmp_path, data_dir, val_ratio=0.0))
    assert deduped.records_out == 2
    assert deduped.dropped == {"duplicate": 1}

    kept = build_dataset(cfg_for(tmp_path, data_dir, val_ratio=0.0, dedupe=False))
    assert kept.records_out == 3
    assert "duplicate" not in kept.dropped


def test_char_bounds_are_inclusive_and_reported(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    write_jsonl(
        data_dir / "records.jsonl",
        [
            {"user": "u", "reply": "x" * 10},
            {"user": "u", "reply": "y" * 60},
            {"user": "u", "reply": "z" * 30},
        ],
    )

    cfg = cfg_for(tmp_path, data_dir, val_ratio=0.0, min_chars=10, max_chars=60)
    assert build_dataset(cfg).records_out == 3

    too_short = build_dataset(cfg_for(tmp_path, data_dir, val_ratio=0.0, min_chars=11))
    assert too_short.records_out == 2
    assert too_short.dropped == {"too_short": 1}

    too_long = build_dataset(
        cfg_for(tmp_path, data_dir, val_ratio=0.0, min_chars=1, max_chars=59)
    )
    assert too_long.records_out == 2
    assert too_long.dropped == {"too_long": 1}


def test_max_tokens_filter(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    write_jsonl(data_dir / "records.jsonl", [{"user": "u" * 10, "reply": "a" * 100}])

    assert build_dataset(cfg_for(tmp_path, data_dir, max_tokens=35)).records_out == 1
    dropped = build_dataset(cfg_for(tmp_path, data_dir, max_tokens=34))
    assert dropped.records_out == 0
    assert dropped.dropped == {"too_many_tokens": 1}


def test_empty_input_is_an_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    with pytest.raises(MsftError, match="no records found"):
        build_dataset(cfg_for(tmp_path, data_dir))


def test_relative_input_is_resolved_against_base_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    write_jsonl(data_dir / "records.jsonl", [{"user": "u", "reply": "a" * 80}])
    cfg = RunConfig(name="test", data=DataCfg(input="raw"), base_dir=str(tmp_path))
    assert build_dataset(cfg).records_out == 1


# ---------------------------------------------------------------------- rendering


def test_render_chat_uses_plain_role_markers() -> None:
    rendered = render_chat(
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
        }
    )
    assert rendered == "<|system|>\ns\n<|user|>\nu\n<|assistant|>\na"


def test_render_chat_accepts_raw_fields_and_empty_records() -> None:
    assert render_chat({"user": "u", "reply": "a"}) == "<|user|>\nu\n<|assistant|>\na"
    assert render_chat({}) == ""
