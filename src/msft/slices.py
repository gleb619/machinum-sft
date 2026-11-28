"""Deterministic protocol slices: split artifact trees into disjoint training arms.

`msft data slice` answers one question: given the artifact trees the pipeline produced, which
records train protocol A and which train protocol B, with nothing shared between the arms.
Everything here is a pure function of the input bytes: records are classified by their system
turn (`msft.protocols`), ordered by the same SHA-256 digest `data.py` uses for dedupe and
split, and written per input file so the result is inspectable and `msft data build` can eat
it unchanged.

The builder is deliberately loud. Every record that is read ends up in exactly one bucket,
the accounting identity is asserted before anything is written, and a run that cannot satisfy
`--min-per-protocol`/`--min-per-arm` exits without touching the output tree: a half-built
slice would be trained on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msft.config import MsftError
from msft.data import _digest, _in_val, _input_files, iter_records, normalize_record
from msft.protocols import PROTOCOLS, classify_record, system_text

#: Split ratio the case picker uses; the same 0.05 the configs carry for `data build`.
CASE_VAL_RATIO = 0.05
UNION = "union"
_TAGS = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_ARM_NAME = re.compile(r"[A-Za-z0-9._-]+")


class GateError(MsftError):
    """A slice gate refused the inputs: a red result (exit 1), not a usage error (exit 2)."""


@dataclass
class Entry:
    """One classified record with everything the builder needs to place it."""

    uid: int
    stem: str
    raw: dict[str, Any]
    messages: list[dict[str, Any]]
    digest: str
    protocol: str | None
    passage: str
    source: str = ""


def parse_arm(value: str) -> tuple[str, list[str]]:
    """Parse ``NAME=PROTO[,PROTO]`` into an arm name and its canonical protocol list."""
    name, separator, raw = str(value).partition("=")
    name = name.strip()
    if not separator or not name:
        raise MsftError(f"--arm must be NAME=PROTO[,PROTO]: {value!r}")
    if not _ARM_NAME.fullmatch(name):
        raise MsftError(f"arm name '{name}' must match [A-Za-z0-9._-]+")
    if name == UNION:
        raise MsftError(f"arm name '{UNION}' is reserved for the combined slice")
    protocols = [item.strip() for item in raw.split(",") if item.strip()]
    if not protocols:
        raise MsftError(f"arm '{name}' names no protocol: {value!r}")
    unknown = [item for item in protocols if item not in PROTOCOLS]
    if unknown:
        raise MsftError(
            f"arm '{name}' names unknown protocol(s): {', '.join(unknown)}; "
            f"known: {', '.join(PROTOCOLS)}"
        )
    if len(set(protocols)) != len(protocols):
        raise MsftError(f"arm '{name}' repeats a protocol: {value!r}")
    return name, protocols


def parse_arms(values: list[str]) -> dict[str, list[str]]:
    """Parse every ``--arm`` and refuse an arm order that gives one protocol two owners."""
    arms: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for value in values:
        name, protocols = parse_arm(value)
        if name in arms:
            raise MsftError(f"arm '{name}' is declared twice")
        for protocol in protocols:
            if protocol in owner:
                raise MsftError(
                    f"protocol '{protocol}' is in arm '{owner[protocol]}' and arm '{name}'; "
                    "the arms must be disjoint"
                )
            owner[protocol] = name
        arms[name] = protocols
    if not arms:
        raise MsftError("at least one --arm NAME=PROTO[,PROTO] is required")
    return arms


def passage_key(user: str) -> str:
    """Protocol-less passage key: tags stripped, whitespace collapsed, lower-cased, 16 hex."""
    text = _TAGS.sub(" ", user)
    text = _WHITESPACE.sub(" ", text).strip().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _user_text(messages: list[dict[str, Any]]) -> str:
    """Every user turn of a record, joined in order."""
    return "\n".join(
        str(message["content"]) for message in messages if message.get("role") == "user"
    )


def _assistant_text(messages: list[dict[str, Any]]) -> str:
    """The last assistant turn, or the empty string when the record has none."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message["content"])
    return ""


def _read_entries(inputs: list[str]) -> list[Entry]:
    """Read every input file once, tagging each record with its classifier verdict."""
    stems: dict[str, Path] = {}
    entries: list[Entry] = []
    uid = 0
    for value in inputs:
        for path in _input_files(Path(value)):
            stem = path.stem
            if stem in stems and stems[stem] != path:
                raise MsftError(
                    f"two input files share the stem '{stem}': {stems[stem]} and {path}; "
                    "the per-file slice output would collide"
                )
            stems[stem] = path
            for raw in iter_records(path):
                record = normalize_record(raw)
                if record is None:
                    entries.append(
                        Entry(uid, stem, raw, [], "", None, passage_key(""))
                    )
                    uid += 1
                    continue
                messages = record["messages"]
                entries.append(
                    Entry(
                        uid,
                        stem,
                        raw,
                        messages,
                        _digest(messages),
                        classify_record(raw),
                        passage_key(_user_text(messages)),
                        record.get("source", ""),
                    )
                )
                uid += 1
    if not entries:
        raise MsftError("no records found in " + ", ".join(str(item) for item in inputs))
    return entries


def _ordered(entries: list[Entry]) -> tuple[list[Entry], int]:
    """Sort by digest then file stem and drop duplicate digests; return (kept, duplicates)."""
    ordered = sorted(entries, key=lambda entry: (entry.digest, entry.stem, entry.uid))
    kept: list[Entry] = []
    seen: set[str] = set()
    duplicates = 0
    for entry in ordered:
        if entry.digest and entry.digest in seen:
            duplicates += 1
            continue
        if entry.digest:
            seen.add(entry.digest)
        kept.append(entry)
    return kept, duplicates


def _drop_cross_arm(
    entries: list[Entry], owner: dict[str, str]
) -> tuple[set[int], list[dict[str, Any]]]:
    """Keep one copy per passage across arms; return (dropped uids, collision report)."""
    by_passage: dict[str, list[Entry]] = {}
    for entry in entries:
        if entry.protocol is None or entry.protocol not in owner:
            continue
        by_passage.setdefault(entry.passage, []).append(entry)

    dropped: set[int] = set()
    report: list[dict[str, Any]] = []
    for key in sorted(by_passage):
        group = by_passage[key]
        arms = sorted({owner[entry.protocol] for entry in group if entry.protocol})
        if len(arms) < 2:
            continue
        keep = arms[0]
        gone = [entry for entry in group if owner.get(entry.protocol or "") != keep]
        if not gone:
            continue
        dropped.update(entry.uid for entry in gone)
        report.append(
            {
                "passage": key,
                "arms": arms,
                "kept": keep,
                "dropped": len(gone),
                "dropped_protocols": sorted({str(entry.protocol) for entry in gone}),
            }
        )
    return dropped, report


def _jsonl_bytes(raws: list[dict[str, Any]]) -> bytes:
    """Render records as JSONL exactly as `data.py` writes them, keeping every string byte."""
    return "".join(
        json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n" for raw in raws
    ).encode("utf-8")


def _case_protocols(arms: dict[str, list[str]]) -> list[str]:
    """Protocols the arms claim, in canonical order."""
    claimed = {protocol for protocols in arms.values() for protocol in protocols}
    return [protocol for protocol in PROTOCOLS if protocol in claimed]


def _cases(
    entries: list[Entry], arms: dict[str, list[str]], per_protocol: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, dict[str, int]]]:
    """Pick eval cases per protocol; val side first, then the lowest train-side digests."""
    cases: dict[str, list[dict[str, Any]]] = {}
    systems: dict[str, str] = {}
    stats: dict[str, dict[str, int]] = {}
    for protocol in _case_protocols(arms):
        pool = sorted(
            (entry for entry in entries if entry.protocol == protocol),
            key=lambda entry: entry.digest,
        )
        if len(pool) < per_protocol:
            raise GateError(
                f"protocol '{protocol}' has {len(pool)} record(s); "
                f"--cases-per-protocol {per_protocol} cannot be met"
            )
        val = [entry for entry in pool if _in_val(entry.digest, CASE_VAL_RATIO)]
        train = [entry for entry in pool if not _in_val(entry.digest, CASE_VAL_RATIO)]
        chosen = [(entry, "val") for entry in val[:per_protocol]]
        if len(chosen) < per_protocol:
            short = per_protocol - len(chosen)
            chosen += [(entry, "train") for entry in train[:short]]
        rows = []
        for index, (entry, split) in enumerate(chosen, 1):
            reference = _assistant_text(entry.messages)
            rows.append(
                {
                    "case": f"w4-{protocol}-val-{index}",
                    "task": protocol,
                    "prompt": _user_text(entry.messages),
                    "reference": reference,
                    "expected_lines": len(
                        [line for line in reference.splitlines() if line.strip()]
                    ),
                    "split": split,
                    "source": entry.source,
                }
            )
        cases[protocol] = rows
        texts = {system_text(entry.raw) for entry, _ in chosen}
        systems[protocol] = sorted(texts, key=lambda text: (-len(text), text))[0]
        stats[protocol] = {
            "count": len(rows),
            "from_val": sum(1 for _, split in chosen if split == "val"),
            "from_train": sum(1 for _, split in chosen if split == "train"),
            "system_variants": len(texts),
        }
    return cases, systems, stats


def _outside_counts(entries: list[Entry], owner: dict[str, str]) -> dict[str, int]:
    """Classified records whose protocol no arm claims, per protocol."""
    counts: dict[str, int] = {}
    for entry in entries:
        if entry.protocol is not None and entry.protocol not in owner:
            counts[entry.protocol] = counts.get(entry.protocol, 0) + 1
    return dict(sorted(counts.items()))


def _write_tree(root: Path, outputs: dict[str, bytes]) -> None:
    """Write every relative path under `root` (already a temporary directory)."""
    for relative in sorted(outputs):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(outputs[relative])


def _replace_tree(tmp: Path, out: Path) -> None:
    """Swap the finished temporary tree in, removing the previous slice only now."""
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)


def run(
    inputs: list[str],
    arms_values: list[str],
    out: str | Path,
    *,
    report: str | Path | None = None,
    cases_dir: str | Path | None = None,
    cases_per_protocol: int = 0,
    min_per_protocol: int = 1,
    min_per_arm: int = 1,
    execute: bool = False,
) -> dict[str, Any]:
    """Classify, dedupe, split into arms and (with `execute`) write the slice tree.

    Nothing is written when `execute` is false, and nothing is written when a gate fails.
    """
    arms = parse_arms(arms_values)
    if cases_per_protocol < 0:
        raise MsftError("--cases-per-protocol must not be negative")
    if cases_per_protocol and cases_dir is None:
        raise MsftError("--cases-per-protocol needs --cases-dir to write the cases into")
    if min_per_protocol < 0 or min_per_arm < 0:
        raise MsftError("--min-per-protocol and --min-per-arm must not be negative")

    entries = _read_entries([str(item) for item in inputs])
    records_in = len(entries)
    kept, duplicates = _ordered(entries)

    per_protocol: dict[str, int] = {}
    unclassified = 0
    invalid = 0
    for entry in kept:
        if not entry.messages:
            invalid += 1
        if entry.protocol is None:
            unclassified += 1
            continue
        per_protocol[entry.protocol] = per_protocol.get(entry.protocol, 0) + 1

    owner = {protocol: name for name, protocols in arms.items() for protocol in protocols}
    dropped_uids, cross_arm_passages = _drop_cross_arm(kept, owner)
    cross_arm_dropped = len(dropped_uids)

    placed = [entry for entry in kept if entry.uid not in dropped_uids]
    outside_arms = sum(
        1 for entry in placed if entry.protocol is not None and entry.protocol not in owner
    )
    per_arm = {name: 0 for name in arms}
    for entry in placed:
        if entry.protocol in owner:
            per_arm[owner[entry.protocol]] += 1

    per_arm_protocol: dict[str, dict[str, int]] = {name: {} for name in arms}
    for entry in placed:
        if entry.protocol in owner:
            bucket = per_arm_protocol[owner[entry.protocol]]
            bucket[entry.protocol] = bucket.get(entry.protocol, 0) + 1

    for name, protocols in arms.items():
        for protocol in protocols:
            found = per_arm_protocol[name].get(protocol, 0)
            if found < min_per_protocol:
                raise GateError(
                    f"arm '{name}' has {found} '{protocol}' record(s), "
                    f"below --min-per-protocol {min_per_protocol}; nothing was written"
                )
        if per_arm[name] < min_per_arm:
            raise GateError(
                f"arm '{name}' has {per_arm[name]} record(s), "
                f"below --min-per-arm {min_per_arm}; nothing was written"
            )

    identity = sum(per_arm.values()) + unclassified + outside_arms + cross_arm_dropped
    if identity + duplicates != records_in:
        raise MsftError(
            f"accounting identity broken: records_in={records_in} but "
            f"arms={sum(per_arm.values())} + unclassified={unclassified} + "
            f"outside_arms={outside_arms} + cross_arm_dropped={cross_arm_dropped} + "
            f"duplicates={duplicates} = {identity + duplicates}"
        )

    outputs: dict[str, bytes] = {}
    files: dict[str, list[dict[str, Any]]] = {}
    for name in list(arms) + [UNION]:
        if name == UNION:
            chosen = [entry for entry in placed if entry.protocol in owner]
        else:
            chosen = [entry for entry in placed if owner.get(entry.protocol or "") == name]
        by_stem: dict[str, list[dict[str, Any]]] = {}
        for entry in sorted(chosen, key=lambda item: (item.stem, item.digest)):
            by_stem.setdefault(entry.stem, []).append(entry.raw)
        files[name] = []
        for stem in sorted(by_stem):
            payload = _jsonl_bytes(by_stem[stem])
            relative = f"{name}/{stem}.jsonl"
            outputs[relative] = payload
            files[name].append(
                {
                    "path": relative,
                    "records": len(by_stem[stem]),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

    cases: dict[str, list[dict[str, Any]]] = {}
    systems: dict[str, str] = {}
    case_stats: dict[str, dict[str, int]] = {}
    if cases_per_protocol:
        cases, systems, case_stats = _cases(placed, arms, cases_per_protocol)
        for protocol, rows in cases.items():
            payload = _jsonl_bytes(rows)
            outputs[f"cases/{protocol}.jsonl"] = payload
            case_stats[protocol]["sha256"] = hashlib.sha256(payload).hexdigest()
        for protocol, text in systems.items():
            payload = (json.dumps(
                {"protocol": protocol, "system": text}, ensure_ascii=False, indent=2
            ) + "\n").encode("utf-8")
            outputs[f"cases/{protocol}.system.json"] = payload

    result: dict[str, Any] = {
        "stage": "data-slice",
        "inputs": [str(item) for item in inputs],
        "arms": arms,
        "out": str(out),
        "report": str(report) if report else None,
        "records_in": records_in,
        "per_protocol": dict(sorted(per_protocol.items())),
        "unclassified": unclassified,
        "invalid": invalid,
        "duplicates": duplicates,
        "cross_arm_dropped": cross_arm_dropped,
        "cross_arm_passages": cross_arm_passages,
        "outside_arms": outside_arms,
        "outside_arms_per_protocol": _outside_counts(placed, owner),
        "per_arm": per_arm,
        "per_arm_protocol": per_arm_protocol,
        "dropped_total": unclassified + outside_arms + cross_arm_dropped + duplicates,
        "gates": {
            "min_per_protocol": min_per_protocol,
            "min_per_arm": min_per_arm,
            "ok": True,
        },
        "files": files,
        "cases": case_stats,
        "cases_dir": str(cases_dir) if cases_dir else None,
    }

    if not execute:
        result["executed"] = False
        return result

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_path.name}.", dir=out_path.parent))
    try:
        _write_tree(tmp, outputs)
        slice_report = dict(result)
        slice_report["executed"] = True
        (tmp / "slice.json").write_text(
            json.dumps(slice_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_tree(tmp, out_path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if cases_dir is not None:
        case_root = Path(cases_dir)
        case_root.parent.mkdir(parents=True, exist_ok=True)
        case_tmp = Path(tempfile.mkdtemp(prefix=f".{case_root.name}.", dir=case_root.parent))
        try:
            case_outputs = {
                relative[len("cases/"):]: payload
                for relative, payload in outputs.items()
                if relative.startswith("cases/")
            }
            _write_tree(case_tmp, case_outputs)
            _replace_tree(case_tmp, case_root)
        except BaseException:
            shutil.rmtree(case_tmp, ignore_errors=True)
            raise

    if report is not None:
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
        report_tmp.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(report_tmp, report_path)

    result["executed"] = True
    return result
