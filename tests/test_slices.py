"""Tests for the protocol slice builder: determinism, accounting, gates and case picking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msft import slices
from msft.config import DataCfg, MsftError, RunConfig
from msft.data import build_dataset
from msft.evaluate import run_gates
from msft.slices import GateError

TRANSLATE_SYSTEM = (
    "You are a professional literary translator. Translate the user text from English to "
    "Russian."
)
PROOFREAD_SYSTEM = "You are an AI assistant that proofreads web novel chapters from English."
TYPOFIX_SYSTEM = "Fix only obvious typos and punctuation mistakes. Keep the meaning."
SUMMARIZE_SYSTEM = "You are a precise AI assistant producing plain text summary notes."
NER_SYSTEM = "You are a Web Novel Glossary Extractor. Identify notable terms."

SHARED = "Kael found the artifact under the old oak."
ARMS = ["a=summarize,ner", "b=proofread,typofix"]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as JSONL, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


def field_record(system: str, user: str, reply: str, source: str) -> dict:
    """A pipeline record in the `system`/`user`/`reply` shape."""
    return {"system": system, "user": user, "reply": reply, "source": source}


def message_record(system: str, user: str, reply: str) -> dict:
    """A pipeline record in the `messages` shape (how the W4a exports arrive)."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
        ]
    }


def build_tree(root: Path) -> list[str]:
    """Three field-format files plus two message-format ones under `root`."""
    write_jsonl(
        root / "tr" / "translate.jsonl",
        [
            field_record(
                TRANSLATE_SYSTEM,
                f"Kael walked the {index} road to the village.",
                f"Кэл шёл дорогой номер {index}.\nОн устал.",
                f"book1/chapter-1.en.md#p{index}",
            )
            for index in range(6)
        ],
    )
    write_jsonl(
        root / "pf" / "proofread.jsonl",
        [
            field_record(
                PROOFREAD_SYSTEM,
                f"<result>\n<id>1</id>\n<text>The guard spoke to stranger {index}.</text>\n"
                "</result>",
                f"<result>\n<id>1</id>\n<text>The guard spoke to stranger {index}.</text>\n"
                "</result>",
                f"book1/proofread#g{index}",
            )
            for index in range(6)
        ],
    )
    write_jsonl(
        root / "tf" / "typofix.jsonl",
        [
            field_record(
                TYPOFIX_SYSTEM,
                f"Teh dog barked at teh moon nnumber {index}.",
                f"The dog barked at the moon number {index}.",
                f"book1/typofix#b{index}",
            )
            for index in range(6)
        ],
    )
    write_jsonl(
        root / "sm" / "summarize.jsonl",
        [
            message_record(
                SUMMARIZE_SYSTEM,
                f"Chapter {index} tells how Kael finds the artifact.",
                f"- Kael finds the artifact, mark {index}.\n- The guard watches.",
            )
            for index in range(8)
        ],
    )
    write_jsonl(
        root / "ne" / "ner.jsonl",
        [
            message_record(
                NER_SYSTEM,
                f"Chapter {index} names Kael and the old oak.",
                f'[{{"name": "Kael", "category": "Character", "mark": {index}}}]',
            )
            for index in range(8)
        ],
    )
    return ["tr", "pf", "tf", "sm", "ne"]


def run_slice(root: Path, out: Path, **overrides: object) -> dict:
    """Slice the fixture tree with the W4 arm shape and small gates."""
    kwargs = {
        "min_per_protocol": 2,
        "min_per_arm": 3,
        "execute": True,
    }
    kwargs.update(overrides)
    return slices.run(
        [str(root / name) for name in build_tree(root)],
        ARMS,
        out,
        **kwargs,
    )


def test_accounting_identity_and_counts(tmp_path: Path) -> None:
    result = run_slice(tmp_path / "in", tmp_path / "out")

    assert result["records_in"] == 34
    assert result["per_protocol"] == {
        "ner": 8,
        "proofread": 6,
        "summarize": 8,
        "translate": 6,
        "typofix": 6,
    }
    assert result["unclassified"] == 0
    assert result["duplicates"] == 0
    assert result["cross_arm_dropped"] == 0
    # translate is classified but no arm claims it: counted, never silently dropped.
    assert result["outside_arms"] == 6
    assert result["outside_arms_per_protocol"] == {"translate": 6}
    assert result["per_arm"] == {"a": 16, "b": 12}
    assert result["records_in"] == (
        sum(result["per_arm"].values())
        + result["unclassified"]
        + result["outside_arms"]
        + result["cross_arm_dropped"]
        + result["duplicates"]
    )


def test_slice_is_byte_identical_on_rerun(tmp_path: Path) -> None:
    first = run_slice(tmp_path / "in", tmp_path / "out1")
    second = run_slice(tmp_path / "in", tmp_path / "out2")

    assert first["files"] == second["files"]
    for row in first["files"]["a"] + first["files"]["union"]:
        left = (tmp_path / "out1" / row["path"]).read_bytes()
        right = (tmp_path / "out2" / row["path"]).read_bytes()
        assert left == right


def test_input_order_does_not_change_the_slice(tmp_path: Path) -> None:
    names = build_tree(tmp_path / "in")
    forward = slices.run(
        [str(tmp_path / "in" / name) for name in names],
        ARMS,
        tmp_path / "forward",
        min_per_protocol=2,
        min_per_arm=3,
        execute=True,
    )
    reverse = slices.run(
        [str(tmp_path / "in" / name) for name in reversed(names)],
        ARMS,
        tmp_path / "reverse",
        min_per_protocol=2,
        min_per_arm=3,
        execute=True,
    )
    assert forward["files"] == reverse["files"]


def test_cross_arm_passage_is_dropped_and_reported(tmp_path: Path) -> None:
    root = tmp_path / "in"
    write_jsonl(
        root / "tr" / "translate.jsonl",
        [field_record(TRANSLATE_SYSTEM, SHARED, "Кэл нашёл артефакт.\nПод дубом.", "book1#p0")],
    )
    write_jsonl(
        root / "tf" / "typofix.jsonl",
        [
            # Same normalised user text as the translation above: that is the leakage the
            # builder has to notice, because `source` alone would not show it.
            field_record(TYPOFIX_SYSTEM, SHARED, "Kael found the artifact.", "book1/typofix#b0"),
            field_record(
                TYPOFIX_SYSTEM, "The guard slept.", "The guard slept.", "book1/typofix#b1"
            ),
        ],
    )
    result = slices.run(
        [str(root)],
        ["a=translate", "b=typofix"],
        tmp_path / "out",
        min_per_protocol=1,
        min_per_arm=1,
        execute=True,
    )

    assert result["cross_arm_dropped"] == 1
    assert result["per_arm"] == {"a": 1, "b": 1}
    assert len(result["cross_arm_passages"]) == 1
    collision = result["cross_arm_passages"][0]
    assert collision["kept"] == "a"
    assert collision["arms"] == ["a", "b"]
    kept = (tmp_path / "out" / "a" / "translate.jsonl").read_text(encoding="utf-8")
    assert SHARED in kept


def test_gate_refusal_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "in"
    write_jsonl(
        root / "sm" / "summarize.jsonl",
        [
            message_record(SUMMARIZE_SYSTEM, f"text {index}", f"- note {index}")
            for index in range(3)
        ],
    )
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    with pytest.raises(GateError) as caught:
        slices.run(
            [str(root)],
            ["a=summarize"],
            out,
            report=report,
            min_per_protocol=20,
            min_per_arm=60,
            execute=True,
        )
    assert "summarize" in str(caught.value)
    assert not out.exists()
    assert not report.exists()


def test_duplicate_records_are_counted(tmp_path: Path) -> None:
    root = tmp_path / "in"
    duplicate = field_record(TYPOFIX_SYSTEM, "Teh cat sat.", "The cat sat.", "book1/typofix#b0")
    write_jsonl(root / "one" / "a.jsonl", [duplicate])
    write_jsonl(root / "two" / "b.jsonl", [duplicate])
    result = slices.run(
        [str(root)],
        ["a=typofix"],
        tmp_path / "out",
        min_per_protocol=1,
        min_per_arm=1,
        execute=True,
    )
    assert result["records_in"] == 2
    assert result["duplicates"] == 1
    assert result["per_arm"] == {"a": 1}


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into dicts, skipping blank lines."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_records_are_copied_verbatim(tmp_path: Path) -> None:
    inputs = {name: (tmp_path / "in" / name) for name in build_tree(tmp_path / "in")}
    result = run_slice(tmp_path / "in", tmp_path / "out")
    assert result["executed"] is True

    for filename, arm in (("typofix.jsonl", "b"), ("summarize.jsonl", "a")):
        folder = "tf" if arm == "b" else "sm"
        source = read_jsonl(inputs[folder] / filename)
        written = read_jsonl(tmp_path / "out" / arm / filename)
        assert sorted(
            written, key=lambda item: json.dumps(item, sort_keys=True)
        ) == sorted(source, key=lambda item: json.dumps(item, sort_keys=True))


def test_slice_lays_out_what_data_build_reads(tmp_path: Path) -> None:
    result = run_slice(tmp_path / "in", tmp_path / "out")
    assert result["files"]["a"]
    config = RunConfig(
        name="slice-a",
        data=DataCfg(input=str(tmp_path / "out" / "a"), val_ratio=0.0, min_chars=1),
        output_dir=str(tmp_path / "built"),
    )
    stats = build_dataset(config, execute=True)
    assert stats.records_in == 16
    assert stats.records_out == 16
    assert stats.sources and all(Path(source).suffix == "" for source in stats.sources)


def test_cases_have_the_required_fields(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    result = run_slice(
        tmp_path / "in",
        tmp_path / "out",
        cases_dir=cases_dir,
        cases_per_protocol=2,
    )
    assert result["cases"]["typofix"]["count"] == 2
    assert set(result["cases"]) == {"summarize", "ner", "proofread", "typofix"}
    rows = [
        json.loads(line)
        for line in (cases_dir / "typofix.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["case"] for row in rows] == ["w4-typofix-val-1", "w4-typofix-val-2"]
    for row in rows:
        assert row["task"] == "typofix"
        assert row["prompt"]
        assert row["reference"]
        assert row["expected_lines"] == len(
            [line for line in row["reference"].splitlines() if line.strip()]
        )
        assert row["split"] in ("val", "train")
    system = json.loads((cases_dir / "typofix.system.json").read_text(encoding="utf-8"))
    assert system["protocol"] == "typofix"
    assert system["system"] == TYPOFIX_SYSTEM


def test_case_picker_prefers_the_val_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_tree(tmp_path / "in")
    # Deterministic stand-in for data.py's val split: every second digest is "val".
    seen: dict[str, bool] = {}

    def fake_in_val(digest: str, ratio: float) -> bool:
        seen.setdefault(digest, len(seen) % 2 == 0)
        return seen[digest]

    monkeypatch.setattr(slices, "_in_val", fake_in_val)
    result = slices.run(
        [str(tmp_path / "in" / name) for name in ("tf",)],
        ["b=typofix"],
        tmp_path / "out",
        cases_dir=tmp_path / "cases",
        cases_per_protocol=3,
        min_per_protocol=2,
        min_per_arm=3,
        execute=True,
    )
    assert result["cases"]["typofix"]["count"] == 3
    assert result["cases"]["typofix"]["from_val"] >= 1
    assert result["cases"]["typofix"]["from_val"] + result["cases"]["typofix"]["from_train"] == 3


def test_cases_shortfall_is_loud(tmp_path: Path) -> None:
    build_tree(tmp_path / "in")
    with pytest.raises(GateError) as caught:
        slices.run(
            [str(tmp_path / "in" / "tf")],
            ["b=typofix"],
            tmp_path / "out",
            cases_dir=tmp_path / "cases",
            cases_per_protocol=50,
            min_per_protocol=2,
            min_per_arm=3,
            execute=True,
        )
    assert "typofix" in str(caught.value)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    build_tree(tmp_path / "in")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    cases_dir = tmp_path / "cases"
    result = slices.run(
        [str(tmp_path / "in")],
        ARMS,
        out,
        report=report,
        cases_dir=cases_dir,
        cases_per_protocol=2,
        min_per_protocol=2,
        min_per_arm=3,
        execute=False,
    )
    assert result["executed"] is False
    assert result["files"]["a"]
    assert not out.exists()
    assert not report.exists()
    assert not cases_dir.exists()


def test_arms_must_be_disjoint() -> None:
    with pytest.raises(MsftError) as caught:
        slices.parse_arms(["a=summarize", "b=summarize,ner"])
    assert "summarize" in str(caught.value)
    with pytest.raises(MsftError):
        slices.parse_arms(["a=not-a-protocol"])
    with pytest.raises(MsftError):
        slices.parse_arms(["a=translate", "a=ner", "b=typofix"])


FIXTURE_CASES = Path(__file__).resolve().parent / "fixtures" / "w4-cases" / "cases.jsonl"


def test_fixture_cases_parse_through_the_gates(tmp_path: Path) -> None:
    """The committed 2-case smoke subset must load and be scorable by `msft eval`."""
    rows = read_jsonl(FIXTURE_CASES)
    assert [row["case"] for row in rows] == ["w4-proofread-val-1", "w4-typofix-val-1"]
    assert {row["task"] for row in rows} == {"proofread", "typofix"}
    assert all(row["expected_lines"] > 0 and row["prompt"] for row in rows)

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "case": row["case"],
                    "task": row["task"],
                    "reply": row["reference"],
                    "expected_lines": row["expected_lines"],
                    "prompt_logged": row["prompt"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    gates = {gate.gate: gate for gate in run_gates(predictions)}
    assert set(gates) == {
        "no_chatter",
        "xml_shape",
        "properties_shape",
        "line_parity",
        "terms_block",
        "schema_validity",
    }
    assert gates["no_chatter"].passed
    assert gates["line_parity"].passed
    # Both cases are proofread/typofix, so the translation contract is not scored at all
    # (W4d, kanban T9): the gold reference has no <terms> block and never was asked for one.
    assert gates["terms_block"].passed
    assert gates["terms_block"].value is None
    assert gates["schema_validity"].passed
    assert gates["schema_validity"].value == 1.0
