"""Dataset mining: read pipeline artifacts, normalise to chat turns, filter, dedupe, split.

The pipeline emits JSONL where a line is either ``{system, user, reply}`` or a
``messages`` array of ``{role, content}`` objects, plus ``.json`` files holding a list of
such records.  Everything here is deterministic: records are ordered and split by a
SHA-256 digest of their normalised messages, never by ``random``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from msft.config import MsftError, RunConfig

ROLES = ("system", "user", "assistant")
SUFFIXES = frozenset({".jsonl", ".json"})
APPROX_TOKENS_METHOD = "ceil(chars/3.2)"

_TAG = re.compile(r"^[A-Za-z0-9_+.#-]{1,32}$")


@dataclass
class DataStats:
    """What a dataset build read, kept and threw away."""

    records_in: int = 0
    records_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    train: int = 0
    val: int = 0
    chars: int = 0
    approx_tokens: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    approx_tokens_method: str = APPROX_TOKENS_METHOD

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view of the stats."""
        return asdict(self)


# --------------------------------------------------------------------------- reading


def _input_files(path: Path) -> list[Path]:
    """Return the readable .jsonl/.json files for a file or directory input, sorted."""
    if path.is_file():
        if path.suffix.lower() not in SUFFIXES:
            raise MsftError(f"data input must be .jsonl or .json: {path}")
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES
        )
    raise MsftError(f"data input not found: {path}")


def _read_jsonl(path: Path) -> Iterator[dict]:
    """Yield every JSON object in a JSONL file; blank lines and non-objects are skipped."""
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise MsftError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
            if isinstance(raw, dict):
                yield raw


def _read_json_array(path: Path) -> Iterator[dict]:
    """Yield the objects of a JSON array, or a single object, from a .json file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MsftError(f"{path}: invalid JSON: {exc.msg}") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise MsftError(f"{path}: expected a JSON list of records")
    for raw in data:
        if isinstance(raw, dict):
            yield raw


def iter_records(path: Path) -> Iterator[dict]:
    """Yield raw record dicts from one .jsonl/.json file or a directory tree, sorted by path."""
    for file in _input_files(Path(path)):
        if file.suffix.lower() == ".json":
            yield from _read_json_array(file)
        else:
            yield from _read_jsonl(file)


# ---------------------------------------------------------------------- normalising


def _text(value: Any) -> str | None:
    """Return the string unchanged when it carries non-whitespace text, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _is_fence_only(content: str) -> bool:
    """True when a reply is nothing but fence markers (optionally with a language tag)."""
    text = content.strip()
    if text == "```":
        return True
    if len(text) < 6 or not text.startswith("```") or not text.endswith("```"):
        return False
    inner = text[3:-3].strip()
    if not inner:
        return True
    lines = inner.splitlines()
    return len(lines) == 1 and bool(_TAG.match(lines[0]))


def _from_messages(raw_messages: Any) -> list[dict] | None:
    """Turn a messages array into clean turns, dropping empties and unknown roles."""
    if not isinstance(raw_messages, list):
        return None
    messages: list[dict] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str):
            continue
        role = role.strip().lower()
        if role not in ROLES:
            continue
        text = _text(content)
        if text is None:
            continue
        messages.append({"role": role, "content": text})
    return messages


def _from_fields(raw: dict) -> list[dict] | None:
    """Turn system/user/reply fields into a message list; None when no field is usable."""
    system = _text(raw.get("system"))
    user = _text(raw.get("user"))
    reply = _text(raw.get("reply"))
    if user is None and reply is None:
        return None
    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if user is not None:
        messages.append({"role": "user", "content": user})
    if reply is not None:
        messages.append({"role": "assistant", "content": reply})
    return messages


def _reject(messages: list[dict]) -> str | None:
    """Name the reason a cleaned message list cannot be trained on, or None."""
    if not any(m["role"] == "user" for m in messages):
        return "no_user"
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    if not assistants:
        return "no_assistant"
    if any(_is_fence_only(text) for text in assistants):
        return "code_fence"
    return None


def _normalize(raw: Any) -> tuple[dict | None, str | None]:
    """Normalise one raw record and say why it was rejected, if it was."""
    if not isinstance(raw, dict):
        return None, "invalid"
    messages = _from_messages(raw.get("messages")) or _from_fields(raw)
    if messages is None:
        return None, "no_messages"
    reason = _reject(messages)
    if reason is not None:
        return None, reason
    source = raw.get("source")
    source = source.strip() if isinstance(source, str) else ""
    return {"messages": messages, "source": source}, None


def normalize_record(raw: dict) -> dict | None:
    """Return ``{"messages": [...], "source": str}`` or None when the record is unusable."""
    record, _ = _normalize(raw)
    return record


# ------------------------------------------------------------------------ rendering


def render_chat(record: dict) -> str:
    """Render a record with a plain ``<|role|>`` template when no chat template exists."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        normalized = normalize_record(record)
        messages = normalized["messages"] if normalized else []
    blocks: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        blocks.append(f"<|{role.strip().lower()}|>\n{content}")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- building


def _approx_tokens(chars: int) -> int:
    """Estimate tokens for Cyrillic-heavy text as ceil(chars/3.2), with no tokenizer."""
    # 1/3.2 == 5/16 exactly, so keep the arithmetic in integers and off the float edge.
    return (chars * 5 + 15) // 16


def _digest(messages: list[dict]) -> str:
    """Stable SHA-256 over the normalised messages, used for dedupe, order and split."""
    payload = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _in_val(digest: str, val_ratio: float) -> bool:
    """Deterministically place a record in val by comparing its digest to val_ratio."""
    if val_ratio <= 0.0:
        return False
    return int(digest[:8], 16) / float(1 << 32) < val_ratio


def _bump(counter: dict[str, int], reason: str) -> None:
    """Increment one dropped-by-reason counter."""
    counter[reason] = counter.get(reason, 0) + 1


def _assistant_text(messages: list[dict]) -> str:
    """Return the last assistant content of an already validated message list."""
    return next(m["content"] for m in reversed(messages) if m["role"] == "assistant")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as UTF-8 JSONL, one object per line, keys sorted for stable diffs."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")


def build_dataset(cfg: RunConfig, *, execute: bool = False) -> DataStats:
    """Filter, dedupe and split ``cfg.data.input``; write train/val/stats only when executing.

    ``stats.chars`` counts assistant characters kept; ``stats.approx_tokens`` is
    ``ceil(chars/3.2)`` because no tokenizer is available offline.
    """
    data = cfg.data
    root = Path(data.input)
    if not root.is_absolute():
        root = Path(cfg.base_dir) / root

    dropped: dict[str, int] = {}
    kept: list[tuple[str, dict, int]] = []
    seen: set[str] = set()
    records_in = 0

    for raw in iter_records(root):
        records_in += 1
        record, reason = _normalize(raw)
        if record is None:
            _bump(dropped, reason or "invalid")
            continue
        messages = record["messages"]
        answer = _assistant_text(messages)
        if len(answer) < data.min_chars:
            _bump(dropped, "too_short")
            continue
        if len(answer) > data.max_chars:
            _bump(dropped, "too_long")
            continue
        if data.max_tokens is not None:
            total = sum(len(m["content"]) for m in messages)
            if _approx_tokens(total) > data.max_tokens:
                _bump(dropped, "too_many_tokens")
                continue
        digest = _digest(messages)
        if data.dedupe and digest in seen:
            _bump(dropped, "duplicate")
            continue
        seen.add(digest)
        kept.append((digest, record, len(answer)))

    if records_in == 0:
        raise MsftError(f"no records found in {root}")

    kept.sort(
        key=lambda item: (
            item[0],
            item[1]["source"],
            json.dumps(item[1]["messages"], ensure_ascii=False, sort_keys=True),
        )
    )

    train: list[dict] = []
    val: list[dict] = []
    chars = 0
    sources: dict[str, int] = {}
    for digest, record, answer_chars in kept:
        chars += answer_chars
        name = record["source"] or "unknown"
        sources[name] = sources.get(name, 0) + 1
        (val if _in_val(digest, data.val_ratio) else train).append(record)

    stats = DataStats(
        records_in=records_in,
        records_out=len(kept),
        dropped=dict(sorted(dropped.items())),
        train=len(train),
        val=len(val),
        chars=chars,
        approx_tokens=_approx_tokens(chars),
        sources=dict(sorted(sources.items())),
        approx_tokens_method=APPROX_TOKENS_METHOD,
    )

    if execute:
        out = cfg.dataset_path
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "train.jsonl", train)
        _write_jsonl(out / "val.jsonl", val)
        payload = json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        (out / "stats.json").write_text(payload + "\n", encoding="utf-8")

    return stats
