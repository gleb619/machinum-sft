"""`msft arena`: the same prompts through several models, scored by several frameworks.

`eval` scores a recorded file, `probe` smokes one model, this compares. Every candidate gets
one deterministic request per case (``think: false``, ``temperature 0``, a fixed seed); the
judge runs once afterwards, over the collected replies, so the GPU swaps models once.

The report never claims a winner: the table has a sort order, and a model whose paired chrF
difference from the leader straddles zero is marked ``≈``. ``docs/arena.md`` says why.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msft import metrics, probe
from msft.config import MsftError, redact

# evaluate's checker is reused so a leaderboard `schema_validity` means exactly what the
# `schema_validity` gate in `msft eval` means.
from msft.evaluate import _evaluate_case

ENDPOINT_ENV = "MSFT_ARENA_ENDPOINT"
DEFAULT_SEED = 7
DEFAULT_NUM_PREDICT = 4096
DEFAULT_TIMEOUT_S = 300.0
MAX_JUDGE_CHARS = 6000
NOISE_CASES = 30

#: The columns the leaders table ranks one model per axis on. Never summed, never weighted.
LEADER_AXES = (
    "schema_validity",
    "contract",
    "chrf",
    "glossary",
    "judge_fidelity",
    "judge_fluency",
    "speed_index",
)

JUDGE_SYSTEM = (
    "You rate a translation from English into Russian. "
    "Score fidelity from 1 (meaning lost) to 5 (meaning preserved) and fluency from "
    "1 (unnatural Russian) to 5 (natural Russian). Reply with JSON only: "
    '{"fidelity": N, "fluency": N}'
)


@dataclass(frozen=True)
class Case:
    """One arena case, resolved from a JSONL record or an ad-hoc prompt.

    ``task`` and ``expect`` are the instruction half of the case: ``task`` labels what was
    asked for (``summarize``, ``ner``, ``proofread``, ``translate``) so the leaderboard can be
    split by it, and ``expect`` describes the output shape the instruction demanded. A case
    with no ``expect`` has no contract to break and is left out of the contract mean.
    """

    case: str
    prompt: str
    source: str
    reference: str | None
    expected_lines: int | None
    terms: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    error: str | None = None
    task: str | None = None
    expect: dict[str, Any] | None = None


def _log(message: str) -> None:
    """Progress line on stderr, so a `--json` run keeps stdout to one object."""
    print(redact(message), file=sys.stderr)


# --- network seams --------------------------------------------------------


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST one ollama request; tests replace this whole function."""
    return probe._post_json(url, payload, timeout)


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    """GET one JSON document; tests replace this whole function."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}, method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise MsftError(f"endpoint returned {type(parsed).__name__}, expected an object")
    return parsed


def _tags_url(endpoint: str) -> str:
    """The ``/api/tags`` URL for a base endpoint or an endpoint already ending in /api/chat."""
    base = endpoint.rstrip("/")
    if base.endswith("/api/chat"):
        base = base[: -len("/api/chat")]
    return base if base.endswith("/api/tags") else f"{base}/api/tags"


def resolve_endpoint(endpoint: str | None) -> str:
    """`--endpoint`, else ``MSFT_ARENA_ENDPOINT``; there is no default on purpose."""
    value = (endpoint or os.environ.get(ENDPOINT_ENV, "")).strip()
    if not value:
        raise MsftError(f"no endpoint: pass --endpoint or set {ENDPOINT_ENV}")
    return value


# --- cases ----------------------------------------------------------------


def _terms(value: object) -> list[dict[str, str]]:
    """Keep only ``{"src": str, "tgt": str}`` entries."""
    if not isinstance(value, list):
        return []
    terms: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = item.get("src")
        target = item.get("tgt")
        if isinstance(source, str) and isinstance(target, str):
            terms.append({"src": source, "tgt": target})
    return terms


def _tags(value: object) -> list[str]:
    """Keep the string tags only, in order."""
    if not isinstance(value, list):
        return []
    return [tag for tag in value if isinstance(tag, str)]


def _expected_lines(value: object) -> int | None:
    """An integer line count, or None when the record does not carry one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _task(value: object) -> str | None:
    """The task label as written; the canonical four are summarize, ner, proofread, translate.

    Any other non-empty string is kept as its own label instead of being rejected: a case file
    may need a fifth task, and silently dropping the label would hide it from the per-task
    breakdown rather than let it show up there.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _expect(value: object) -> dict[str, Any] | None:
    """A normalised ``expect`` object, or None when the record carries no usable one.

    Only the fields the contract scorer reads are kept, and their types are checked here
    rather than at scoring time: ``str(a_list)`` inside a violation message is less useful than
    a rule that never sees the bad value. An unknown ``format`` is kept on purpose, so the
    scorer reports it as a violation instead of the loader silently making the case untested.
    """
    if not isinstance(value, dict):
        return None
    fmt = value.get("format")
    if not isinstance(fmt, str) or not fmt.strip():
        return None
    expect: dict[str, Any] = {"format": fmt.strip()}
    keys = value.get("keys")
    if isinstance(keys, list):
        kept = [key for key in keys if isinstance(key, str) and key]
        if kept:
            expect["keys"] = kept
    for name in ("bullets_min", "pairs_min"):
        number = value.get(name)
        if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
            expect[name] = number
    if value.get("unchanged") is True:
        expect["unchanged"] = True
    unchanged_line = value.get("unchanged_line")
    if isinstance(unchanged_line, str) and unchanged_line:
        expect["unchanged_line"] = unchanged_line
    return expect


def _case_from_record(record: dict[str, Any], index: int) -> Case:
    """One JSONL record as a `Case`; a missing prompt becomes an error case."""
    name = str(record.get("case") or f"case {index}")
    prompt = probe._prompt(record)
    if prompt is None:
        return Case(
            case=name,
            prompt="",
            source="",
            reference=None,
            expected_lines=None,
            error="case has no prompt, user or messages text",
        )
    raw_source = record.get("source")
    source = raw_source if isinstance(raw_source, str) and raw_source else prompt
    raw_reference = record.get("reference")
    reference = raw_reference if isinstance(raw_reference, str) and raw_reference else None
    return Case(
        case=name,
        prompt=prompt,
        source=source,
        reference=reference,
        expected_lines=_expected_lines(record.get("expected_lines")),
        terms=_terms(record.get("terms")),
        tags=_tags(record.get("tags")),
        task=_task(record.get("task")),
        expect=_expect(record.get("expect")),
    )


def _case_from_line(line: str, lineno: int) -> Case:
    """One JSONL line as a `Case`; bad JSON becomes an error case instead of an exception."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return Case(
            case=f"line {lineno}",
            prompt="",
            source="",
            reference=None,
            expected_lines=None,
            error=f"invalid JSON: {exc}",
        )
    if not isinstance(record, dict):
        return Case(
            case=f"line {lineno}",
            prompt="",
            source="",
            reference=None,
            expected_lines=None,
            error=f"expected a JSON object, got {type(record).__name__}",
        )
    return _case_from_record(record, lineno)


def resolve_cases(cases: Path | None, prompts: list[str]) -> list[Case]:
    """Resolve `--cases` or `--prompt` into one deduplicated list of cases."""
    if cases is not None and prompts:
        raise MsftError("pass either --cases or --prompt, not both")
    if cases is None and not prompts:
        raise MsftError("no cases: pass --cases FILE or at least one --prompt TEXT")
    resolved: list[Case] = []
    if cases is not None:
        path = Path(cases)
        if not path.is_file():
            raise MsftError(f"cases not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip():
                resolved.append(_case_from_line(line, lineno))
    else:
        for index, text_prompt in enumerate(prompts, 1):
            resolved.append(
                Case(
                    case=f"ad-hoc-{index}",
                    prompt=text_prompt,
                    source=text_prompt,
                    reference=None,
                    expected_lines=None,
                )
            )
    seen: set[str] = set()
    for case in resolved:
        if case.case in seen:
            raise MsftError(f"duplicate case name: {case.case}")
        seen.add(case.case)
    return resolved


# --- preflight and candidates --------------------------------------------


def preflight(url: str, models: list[str], *, timeout: float) -> dict[str, str]:
    """Check every model against ``GET /api/tags`` and return name -> digest."""
    data = _get_json(_tags_url(url), timeout)
    entries = data.get("models")
    if not isinstance(entries, list):
        raise MsftError("GET /api/tags returned no 'models' list")
    available: dict[str, str] = {}
    for item in entries:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            digest = item.get("digest")
            available[item["name"]] = digest if isinstance(digest, str) else ""
    missing = [
        name
        for name in models
        if name not in available and f"{name}:latest" not in available
    ]
    if missing:
        known = ", ".join(sorted(available)) or "none"
        raise MsftError(f"unknown model(s): {', '.join(missing)}; available: {known}")
    return {
        name: available.get(name, available.get(f"{name}:latest", "")) for name in models
    }


def _request(
    model: str, prompt: str, system: str, *, seed: int, num_predict: int
) -> dict[str, Any]:
    """probe's chat shape plus the two arena sampling options."""
    payload = probe.build_request(model, prompt, system)
    payload["options"]["seed"] = seed
    payload["options"]["num_predict"] = num_predict
    return payload


def _number(value: object) -> float | None:
    """A finite number from the endpoint, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _timing(data: dict[str, Any]) -> dict[str, Any]:
    """Durations and counters from one ollama reply; token rate excludes model load."""
    total = _number(data.get("total_duration"))
    load = _number(data.get("load_duration"))
    eval_ns = _number(data.get("eval_duration"))
    output_tokens = _number(data.get("eval_count"))
    prompt_tokens = _number(data.get("prompt_eval_count"))
    rate = None
    if output_tokens is not None and eval_ns:
        rate = round(output_tokens / (eval_ns / 1e9), 2)
    return {
        "total_ms": round(total / 1e6, 1) if total is not None else None,
        "load_ms": round(load / 1e6, 1) if load is not None else None,
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "tokens_per_s": rate,
    }


def _thinking(data: dict[str, Any]) -> str | None:
    """The reasoning field of an ollama reply, when the model produced one anyway."""
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("thinking"), str):
        return message["thinking"]
    return None


def _row_metrics(reply: str, case: Case) -> dict[str, Any]:
    """Per-reply chrF/BLEU against the reference, plus the script flags."""
    row: dict[str, Any] = {"script": metrics.script_flags(reply, case.source)}
    if case.reference:
        row["chrf"] = metrics.chrf(reply, case.reference)
        row["bleu"] = metrics.bleu([reply], [case.reference])
    return row


def _row_contract(reply: str, case: Case) -> dict[str, Any]:
    """The per-reply contract score and its violations, or nothing when the case has no `expect`.

    Nothing rather than a zero: a case that never asked for a shape cannot be failed on one,
    and ``score`` only averages the cases that carry an ``expect``.
    """
    value, violations = metrics.contract(reply, case.expect)
    if value is None:
        return {}
    return {"contract": value, "contract_violations": violations}


def run_model(
    url: str,
    model: str,
    cases: list[Case],
    *,
    system: str,
    seed: int,
    num_predict: int,
    timeout: float,
) -> list[dict[str, Any]]:
    """One row per case; a failed request is a row with `_error`, never an exception."""
    chat = probe.chat_url(url)
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {"case": case.case, "prompt_logged": case.prompt}
        if case.task is not None:
            row["task"] = case.task
        if case.expected_lines is not None:
            row["expected_lines"] = case.expected_lines
        if case.reference is not None:
            row["reference"] = case.reference
        if case.error is not None:
            row["_error"] = case.error
            rows.append(row)
            continue
        payload = _request(model, case.prompt, system, seed=seed, num_predict=num_predict)
        try:
            data = _post_json(chat, payload, timeout)
            reply = probe._reply(data)
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the arena
            row["_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row["reply"] = reply
        row["metrics"] = _row_metrics(reply, case)
        row.update(_row_contract(reply, case))
        row["timing"] = _timing(data)
        thinking = _thinking(data)
        if thinking is not None:
            row["thinking_chars"] = len(thinking)
        reason = data.get("done_reason")
        if isinstance(reason, str):
            row["done_reason"] = reason
        rows.append(row)
    return rows


# --- judge ----------------------------------------------------------------


def _clip(text: str) -> str:
    """Truncate to the judge window."""
    return text[:MAX_JUDGE_CHARS]


def judge_prompt(case: Case, reply: str) -> str:
    """The judge's user message: source, optional reference, candidate. Never a model name."""
    parts = ["SOURCE:", _clip(case.source), ""]
    if case.reference:
        parts.extend(["REFERENCE (may be imperfect):", _clip(case.reference), ""])
    parts.extend(
        [
            "CANDIDATE:",
            _clip(reply),
            "",
            'Reply with {"fidelity": N, "fluency": N}',
        ]
    )
    return "\n".join(parts)


def judge_rows(
    url: str,
    judge: str,
    replies: dict[str, list[dict[str, Any]]],
    cases: list[Case],
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    """Score every successful candidate reply with the judge; one row per call, no retries."""
    chat = probe.chat_url(url)
    by_name = {case.case: case for case in cases}
    rows: list[dict[str, Any]] = []
    for model, model_rows in replies.items():
        for row in model_rows:
            reply = row.get("reply")
            case = by_name.get(str(row.get("case")))
            if not isinstance(reply, str) or case is None:
                continue
            payload = {
                "model": judge,
                "think": False,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_prompt(case, reply)},
                ],
            }
            try:
                data = _post_json(chat, payload, timeout)
                raw = probe._reply(data)
            except Exception as exc:  # noqa: BLE001 - a judge failure is a row, not a crash
                rows.append(
                    {
                        "case": row["case"],
                        "model": model,
                        "judge_error": f"{type(exc).__name__}: {exc}",
                        "raw": "",
                    }
                )
                continue
            parsed = metrics.parse_judge(raw)
            if parsed is None:
                rows.append(
                    {
                        "case": row["case"],
                        "model": model,
                        "judge_error": "unparsable judge reply",
                        "raw": raw[:200],
                    }
                )
                continue
            rows.append(
                {
                    "case": row["case"],
                    "model": model,
                    "fidelity": parsed["fidelity"],
                    "fluency": parsed["fluency"],
                }
            )
    return rows


# --- scoring --------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean rounded to four places, or None for an empty list."""
    return round(sum(values) / len(values), 4) if values else None


def _passes_gates(row: dict[str, Any], index: int) -> bool:
    """The same all-applicable-gates verdict `run_gates` gives a record."""
    _, outcomes = _evaluate_case(row, index)
    applicable = [ok for ok, _ in outcomes.values() if ok is not None]
    return bool(applicable) and all(applicable)


def _contract_score(row: dict[str, Any]) -> float:
    """One row's contract score; an unanswered case with an `expect` counts as 0.0.

    Not skipped: a case that asked for a shape and got no reply at all is the worst possible
    compliance, and leaving it out would let a model that dropped half the cases look exact.
    """
    value = row.get("contract")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return round(float(value), 4)


def _task_stats(rows: list[dict[str, Any]], cases: list[Case]) -> dict[str, dict[str, Any]]:
    """Per-task contract and chrF means, so bullets-best and translation-worst share a table.

    The keys are the case records' ``task`` labels; cases without one are absent here rather
    than lumped into an ``unknown`` bucket that would hide which label was forgotten.
    """
    by_case = {str(row.get("case")): row for row in rows}
    totals: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.task is None:
            continue
        bucket = totals.setdefault(
            case.task, {"n": 0, "answered": 0, "contract": [], "chrf": []}
        )
        row = by_case.get(case.case, {})
        bucket["n"] += 1
        if isinstance(row.get("reply"), str):
            bucket["answered"] += 1
        if case.expect is not None:
            bucket["contract"].append(_contract_score(row))
        if case.reference:
            reply = row.get("reply")
            text = reply if isinstance(reply, str) else ""
            bucket["chrf"].append(metrics.chrf(text, case.reference))
    return {
        task: {
            "n": bucket["n"],
            "answered": bucket["answered"],
            "n_contract": len(bucket["contract"]),
            "contract": _mean(bucket["contract"]),
            "n_ref": len(bucket["chrf"]),
            "chrf": _mean(bucket["chrf"]),
        }
        for task, bucket in totals.items()
    }


def score(
    replies: dict[str, list[dict[str, Any]]], cases: list[Case]
) -> dict[str, dict[str, Any]]:
    """Per-model column values; a failed row counts as answered=false and as 0 where scored."""
    referenced = [case for case in cases if case.reference]
    result: dict[str, dict[str, Any]] = {}
    for model, rows in replies.items():
        by_case = {str(row.get("case")): row for row in rows}
        answered = sum(1 for row in rows if isinstance(row.get("reply"), str))
        passed = sum(1 for index, row in enumerate(rows, 1) if _passes_gates(row, index))

        chrf_values: list[float] = []
        hypotheses: list[str] = []
        references: list[str] = []
        for case in referenced:
            row = by_case.get(case.case, {})
            reply = row.get("reply")
            text = reply if isinstance(reply, str) else ""
            chrf_values.append(metrics.chrf(text, case.reference or ""))
            hypotheses.append(text)
            references.append(case.reference or "")

        glossary_values: list[float] = []
        for case in cases:
            if not case.terms:
                continue
            row = by_case.get(case.case, {})
            reply = row.get("reply")
            text = reply if isinstance(reply, str) else ""
            hits = metrics.glossary_hits(text, case.terms)
            glossary_values.append(hits if hits is not None else 0.0)

        contract_values = [
            _contract_score(by_case.get(case.case, {}))
            for case in cases
            if case.expect is not None
        ]

        scripts = [
            row["metrics"]["script"]
            for row in rows
            if isinstance(row.get("metrics"), dict) and "script" in row["metrics"]
        ]
        rates = [
            row["timing"]["tokens_per_s"]
            for row in rows
            if isinstance(row.get("timing"), dict)
            and row["timing"].get("tokens_per_s") is not None
        ]
        loads = [
            row["timing"]["load_ms"] / 1000.0
            for row in rows
            if isinstance(row.get("timing"), dict)
            and row["timing"].get("load_ms") is not None
        ]
        latencies = [
            row["timing"]["total_ms"] / 1000.0
            for row in rows
            if isinstance(row.get("timing"), dict)
            and row["timing"].get("total_ms") is not None
        ]
        result[model] = {
            "answered": answered,
            "total": len(rows),
            "schema_validity": round(passed / len(rows), 4) if rows else 0.0,
            "contract": _mean(contract_values),
            "n_contract": len(contract_values),
            "chrf": _mean(chrf_values),
            "bleu": metrics.bleu(hypotheses, references) if referenced else None,
            "glossary": _mean(glossary_values),
            "n_ref": len(chrf_values),
            "n_glossary": len(glossary_values),
            "mixed_script": sum(int(item["mixed_script_tokens"]) for item in scripts),
            "source_leak": sum(int(item["source_leak_lines"]) for item in scripts),
            "cyr_ratio": _mean([float(item["cyrillic_ratio"]) for item in scripts]),
            "tokens_per_s": _mean([float(value) for value in rates]),
            "p50_reply_s": round(statistics.median(latencies), 2) if latencies else None,
            "worst_reply_s": round(max(latencies), 2) if latencies else None,
            "load_s": _mean(loads),
            "tasks": _task_stats(rows, cases),
        }

    # speed_index is relative to this run, so it can only be computed once every model has a
    # rate: 1.0 is the fastest model here, and the rest is a linear share of that.
    fastest = max(
        (stat["tokens_per_s"] for stat in result.values() if stat["tokens_per_s"]), default=None
    )
    for stat in result.values():
        rate = stat["tokens_per_s"]
        stat["speed_index"] = (
            round(float(rate) / float(fastest), 4)
            if fastest and rate is not None
            else None
        )
    return result


def _judge_stats(judge_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-model fidelity/fluency lists from the successful judge rows."""
    stats: dict[str, dict[str, Any]] = {}
    for row in judge_rows:
        if "judge_error" in row:
            continue
        bucket = stats.setdefault(str(row["model"]), {"fidelity": [], "fluency": []})
        bucket["fidelity"].append(float(row["fidelity"]))
        bucket["fluency"].append(float(row["fluency"]))
    return stats


def _leaders(rows: list[dict[str, Any]]) -> dict[str, str]:
    """The model that leads each axis, one axis at a time.

    A self-judged model is skipped on the judge axes for the same reason its judge mean is
    excluded from the ranking: a model grading itself is not evidence. Ties break on the model
    name so two identical runs print the same table, and an axis no model has a value on is
    simply absent rather than filled with a default.
    """
    leaders: dict[str, str] = {}
    for axis in LEADER_AXES:
        best: tuple[float, str] | None = None
        for row in rows:
            if axis.startswith("judge_") and row.get("judge_star"):
                continue
            value = row.get(axis)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            candidate = (float(value), str(row["model"]))
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] < best[1]
            ):
                best = candidate
        if best is not None:
            leaders[axis] = best[1]
    return leaders


def _near_leader(
    leader_rows: list[dict[str, Any]], rows: list[dict[str, Any]], cases: list[Case]
) -> bool:
    """True when the paired chrF difference from the leader has a 95% interval containing 0."""
    referenced = {case.case for case in cases if case.reference}
    leader_scores = {
        str(row.get("case")): row.get("metrics", {}).get("chrf")
        for row in leader_rows
        if isinstance(row.get("reply"), str)
    }
    differences: list[float] = []
    for row in rows:
        if not isinstance(row.get("reply"), str) or row.get("case") not in referenced:
            continue
        other = leader_scores.get(str(row.get("case")))
        mine = row.get("metrics", {}).get("chrf")
        if other is None or mine is None:
            continue
        differences.append(float(other) - float(mine))
    if len(differences) < 2:
        return False
    mean = statistics.fmean(differences)
    half_width = 1.96 * statistics.stdev(differences) / math.sqrt(len(differences))
    return mean - half_width <= 0.0 <= mean + half_width


def leaderboard(
    models: list[str],
    replies: dict[str, list[dict[str, Any]]],
    cases: list[Case],
    judge_rows: list[dict[str, Any]],
    *,
    self_judge: str | None,
    rank_by: str | None = None,
) -> dict[str, Any]:
    """The sortable table plus the honesty notes; never a single composite score.

    `rank_by` names ONE axis (any column the leaders table ranks plus `tok_per_s`/`answered`)
    and re-sorts the table by it alone. It exists because "which model is best" is four
    different questions now — the default order prefers a protocol-clean reply, but a user who
    cares about throughput must be able to ask for the fastest model without a weighted sum
    inventing a winner.
    """
    stats = score(replies, cases)
    judged = _judge_stats(judge_rows)
    n_ref = sum(1 for case in cases if case.reference)
    rows: list[dict[str, Any]] = []
    for model in models:
        stat = stats[model]
        bucket = judged.get(model, {"fidelity": [], "fluency": []})
        fidelity = _mean(bucket["fidelity"])
        fluency = _mean(bucket["fluency"])
        pairs = [
            (fidelity_value + fluency_value) / 2
            for fidelity_value, fluency_value in zip(
                bucket["fidelity"], bucket["fluency"], strict=True
            )
        ]
        starred = self_judge is not None and model == self_judge
        rows.append(
            {
                "model": model,
                **stat,
                "judge_fidelity": fidelity,
                "judge_fluency": fluency,
                "judge_mean": None if starred else _mean(pairs),
                "judge_star": starred,
                "judged": len(bucket["fidelity"]),
                "near_leader": False,
            }
        )

    if n_ref:
        # With cases that carry an `expect`, format compliance outranks translation similarity:
        # an unusable shape is a retry in the pipeline, a two-point chrF gap may be noise.
        if any(case.expect is not None for case in cases):
            sort_key = "schema_validity, then contract, then chrF"
            fields = ("schema_validity", "contract", "chrf")
        else:
            sort_key = "schema_validity, then chrF"
            fields = ("schema_validity", "chrf")
    elif judge_rows:
        sort_key = "schema_validity, then judge mean"
        fields = ("schema_validity", "judge_mean")
    else:
        sort_key = "answered"
        fields = ("answered",)

    if rank_by is not None:
        axis = str(rank_by).strip()
        allowed = {*LEADER_AXES, "tok_per_s", "answered", "p50_reply_s"}
        if axis not in allowed:
            raise MsftError(
                f"unknown --rank-by {rank_by!r}: pick one of {', '.join(sorted(allowed))}"
            )
        # `p50_reply_s` is a latency, so a smaller number wins; every other axis is a score.
        ascending = axis == "p50_reply_s"
        fields = (axis,)
        sort_key = f"requested: {axis}{' (lowest first)' if ascending else ''}"
    else:
        ascending = False

    def order(row: dict[str, Any]) -> tuple[Any, ...]:
        """Descending on the sort fields, then the model name so ties are reproducible."""
        ranked = tuple(
            (row.get(name) or 0.0) if ascending else -(row.get(name) or 0.0) for name in fields
        )
        return ranked + (row["model"],)

    rows.sort(key=order)
    leader = rows[0]["model"] if rows else None
    leader_rows = replies.get(leader, []) if leader is not None else []
    leader_validity = rows[0]["schema_validity"] if rows else None
    for row in rows[1:]:
        # A chrF tie is only worth marking when the protocol agrees: a model that fails
        # line parity is not "indistinguishable", it is unusable in the pipeline.
        row["near_leader"] = (
            row["schema_validity"] == leader_validity
            and _near_leader(leader_rows, replies.get(row["model"], []), cases)
        )

    notes = [
        "a failed request is `answered` = no and scores 0 for schema, chrF and BLEU on that case",
    ]
    if n_ref:
        notes.append(
            f"chrF and BLEU average over {n_ref} referenced case(s) of {len(cases)}"
        )
    else:
        notes.append(
            "no references in this case set: chrF and BLEU are not computed and the ranking "
            "is format + judge only"
        )
    if len(cases) < NOISE_CASES:
        notes.append(
            f"{len(cases)} case(s): under {NOISE_CASES}, so the differences are within noise "
            "and the order is not a ranking"
        )
    if judge_rows:
        notes.append(
            "the judge run is not retried: an unparsed judge reply is a failed row, not a zero"
        )
    else:
        notes.append("no judge: fidelity and fluency are not measured")
    if self_judge is not None:
        notes.append(
            f"{self_judge} is both candidate and judge: its judge column is marked * and "
            "excluded from the judge mean and the ranking"
        )
    notes.append(
        "tok/s is eval tokens per eval second; model load time is excluded and sits in `load s`"
    )
    if any(case.expect is not None for case in cases):
        notes.append(
            "`contract` is the mean over the cases that carry an `expect`; a case without one "
            "is left out of the mean, never scored 0, and an unanswered contract case is 0"
        )
    if any(case.task is not None for case in cases):
        notes.append(
            "`tasks` in this payload and the per-task table in the markdown split the same "
            "means by the case's `task` label"
        )
    if any(row["speed_index"] is not None for row in rows):
        notes.append(
            "`speed_index` = tok/s ÷ the fastest model's tok/s in THIS run: 1.0 is this run's "
            "fastest, so it is a share, not a rate, and it is never averaged with quality"
        )
    leaders = _leaders(rows)
    if leaders:
        notes.append(
            "`leaders` names one model per axis; the axes are not comparable to each other "
            "and no axis sums them"
        )
    else:
        notes.append("`leaders` is empty: no axis had a value to rank")
    return {
        "models": models,
        "cases": len(cases),
        "n_ref": n_ref,
        "n_contract": sum(1 for case in cases if case.expect is not None),
        "sort_key": sort_key,
        "leader": leader,
        "leaders": leaders,
        "self_judge": self_judge,
        "judge_ran": bool(judge_rows),
        "rows": rows,
        "notes": notes,
    }


# --- report ---------------------------------------------------------------


def _format_optional(value: float | None, digits: int) -> str:
    """A formatted number or `-` when the column does not apply."""
    return "-" if value is None else f"{value:.{digits}f}"


#: Decimals per axis in the leaders table: the columns have different natural precision.
_AXIS_DIGITS = {
    "schema_validity": 3,
    "contract": 3,
    "chrf": 1,
    "glossary": 2,
    "judge_fidelity": 2,
    "judge_fluency": 2,
    "speed_index": 3,
}


def _leaders_table(board: dict[str, Any]) -> list[str]:
    """The `axis | model | value` table, or nothing when no axis had a value.

    It sits above the model table because it answers the question the model table cannot: not
    "which model is best" but "which model leads this one axis". The heading says out loud that
    the axes are not comparable, since a 1.000 on `speed_index` and a 1.000 on `schema_validity`
    are shares of two unrelated denominators.
    """
    leaders = board.get("leaders") or {}
    if not leaders:
        return []
    by_model = {row["model"]: row for row in board["rows"]}
    lines = [
        "who leads on each axis — the axes are not comparable to each other:",
        "",
        "| axis | model | value |",
        "| --- | --- | --- |",
    ]
    for axis, model in leaders.items():
        value = by_model.get(model, {}).get(axis)
        digits = _AXIS_DIGITS.get(axis, 3)
        lines.append(f"| {axis} | {model} | {_format_optional(value, digits)} |")
    lines.append("")
    return lines


def _task_table(board: dict[str, Any]) -> list[str]:
    """The per-task breakdown: the same model can top bullets and trail translation."""
    labels = sorted({task for row in board["rows"] for task in (row.get("tasks") or {})})
    if not labels:
        return []
    lines = [
        "per task — the same means split by each case's `task` label:",
        "",
        "| model | task | answered n/m | contract | chrF |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in board["rows"]:
        stats = row.get("tasks") or {}
        for task in labels:
            stat = stats.get(task)
            if stat is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["model"],
                        f"{task} ({stat['n']})",
                        f"{stat['answered']}/{stat['n']}",
                        _format_optional(stat["contract"], 3),
                        _format_optional(stat["chrf"], 1),
                    ]
                )
                + " |"
            )
    lines.append("")
    return lines


def _note_cell(row: dict[str, Any], leader: str | None) -> str:
    """The note column: leader, ≈ leader, or empty."""
    parts: list[str] = []
    if row["model"] == leader:
        parts.append("leader")
    elif row["near_leader"]:
        parts.append("≈ chrF")
    if row["judge_star"]:
        parts.append("self-judged*")
    return ", ".join(parts)


def format_report(board: dict[str, Any]) -> str:
    """The markdown table plus the notes that keep it honest."""
    n_ref = board["n_ref"]
    n_contract = board.get("n_contract", 0)
    judged_total = sum(int(row["judged"]) for row in board["rows"])
    if board["judge_ran"]:
        judge_header = f"judge f/f (judged {judged_total})"
    else:
        judge_header = "judge f/f (judged 0)"
    header = [
        "model",
        "answered n/m",
        "schema_validity",
        f"contract ({n_contract})",
        f"chrF ({n_ref})",
        f"BLEU ({n_ref})",
        "glossary",
        judge_header,
        "mixed_script",
        "source_leak",
        "cyr_ratio",
        "tok/s",
        "speed_index",
        "p50 s",
        "worst s",
        "load s",
        "note",
    ]
    lines = [f"arena leaderboard — sorted by {board['sort_key']}", ""]
    lines.extend(_leaders_table(board))
    lines.extend(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
    )
    for row in board["rows"]:
        if board["judge_ran"] and row["judged"]:
            star = "*" if row["judge_star"] else ""
            judge_cell = (
                f"{_format_optional(row['judge_fidelity'], 2)}/"
                f"{_format_optional(row['judge_fluency'], 2)}"
                f" ({row['judged']}/{row['answered']}){star}"
            )
        else:
            judge_cell = "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    f"{row['answered']}/{row['total']}",
                    f"{row['schema_validity']:.3f}",
                    _format_optional(row["contract"], 3),
                    _format_optional(row["chrf"], 1),
                    _format_optional(row["bleu"], 1),
                    _format_optional(row["glossary"], 2),
                    judge_cell,
                    str(row["mixed_script"]),
                    str(row["source_leak"]),
                    _format_optional(row["cyr_ratio"], 2),
                    _format_optional(row["tokens_per_s"], 1),
                    _format_optional(row["speed_index"], 3),
                    _format_optional(row["p50_reply_s"], 2),
                    _format_optional(row["worst_reply_s"], 2),
                    _format_optional(row["load_s"], 2),
                    _note_cell(row, board["leader"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(_task_table(board))
    lines.extend(["notes:"])
    lines.extend(f"- {note}" for note in board["notes"])
    return redact("\n".join(lines))


def _plan(
    url: str, models: list[str], cases: list[Case], judge: str | None, out: Path
) -> dict[str, Any]:
    """What a real run would spend, without touching the endpoint."""
    usable = sum(1 for case in cases if case.error is None)
    return {
        "dry_run": True,
        "endpoint": redact(url),
        "models": models,
        "judge": judge,
        "cases": len(cases),
        "error_cases": len(cases) - usable,
        "contract_cases": sum(1 for case in cases if case.expect is not None),
        "tasks": sorted({case.task for case in cases if case.task is not None}),
        "candidate_requests": usable * len(models),
        "judge_calls": usable * len(models) if judge else 0,
        "out": str(out),
        "ok": True,
    }


def format_plan(plan: dict[str, Any]) -> str:
    """The `--dry-run` text: what would be sent, to whom, and where it would land."""
    lines = [
        "arena dry run: no request was made",
        f"  endpoint: {plan['endpoint']}",
        f"  models: {', '.join(plan['models']) or '-'} ({len(plan['models'])})",
        f"  cases: {plan['cases']} (unusable: {plan['error_cases']})",
        f"  contract cases: {plan.get('contract_cases', 0)}",
        f"  tasks: {', '.join(plan.get('tasks') or []) or '-'}",
        f"  candidate requests: {plan['candidate_requests']}",
        f"  judge calls: {plan['judge_calls']}",
        f"  out: {plan['out']}",
    ]
    return redact("\n".join(lines))


# --- artifacts ------------------------------------------------------------


def _slug(model: str) -> str:
    """A file-safe reply name for a model tag."""
    return model.replace(":", "_").replace("/", "_")


def _case_record(case: Case) -> dict[str, Any]:
    """One resolved case as it is written to cases.jsonl."""
    record: dict[str, Any] = {
        "case": case.case,
        "prompt": case.prompt,
        "source": case.source,
    }
    if case.task is not None:
        record["task"] = case.task
    if case.expect is not None:
        record["expect"] = case.expect
    if case.reference is not None:
        record["reference"] = case.reference
    if case.expected_lines is not None:
        record["expected_lines"] = case.expected_lines
    if case.terms:
        record["terms"] = case.terms
    if case.tags:
        record["tags"] = case.tags
    if case.error is not None:
        record["_error"] = case.error
    return record


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON object per line, UTF-8, trailing newline only when non-empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(body + ("\n" if rows else ""), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one indented JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    """The repo commit and dirty flag, or nulls when git is unavailable."""
    state: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return state
    if commit.returncode == 0:
        state["commit"] = commit.stdout.strip() or None
    if status.returncode == 0:
        state["dirty"] = bool(status.stdout.strip())
    return state


def _default_out() -> Path:
    """`out/arena/<UTC timestamp>` — `out/` is git-ignored."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("out") / "arena" / stamp


def run(
    *,
    endpoint: str | None = None,
    models: list[str] | tuple[str, ...] = (),
    cases: Path | str | None = None,
    prompts: list[str] | tuple[str, ...] = (),
    system: str = probe.SYSTEM_PROMPT,
    judge: str | None = None,
    allow_self_judge: bool = False,
    seed: int = DEFAULT_SEED,
    num_predict: int = DEFAULT_NUM_PREDICT,
    timeout: float = DEFAULT_TIMEOUT_S,
    out: Path | str | None = None,
    dry_run: bool = False,
    rank_by: str | None = None,
) -> dict[str, Any]:
    """Preflight, run every candidate, judge once, write the artifacts; never raises on a row."""
    url = resolve_endpoint(endpoint)
    names = list(dict.fromkeys(name for name in models if name))
    if not names:
        raise MsftError("no models: pass --models a,b,c")
    slugs = [_slug(name) for name in names]
    if len(set(slugs)) != len(slugs):
        raise MsftError(f"model names collide as file names: {', '.join(names)}")
    resolved = resolve_cases(Path(cases) if cases is not None else None, list(prompts))
    self_judge = judge if judge is not None and judge in names else None
    if self_judge is not None and not allow_self_judge:
        raise MsftError(
            f"judge {judge} is also a candidate; pass --allow-self-judge to run it anyway"
        )
    out_dir = Path(out) if out is not None else _default_out()
    if dry_run:
        return _plan(url, names, resolved, judge, out_dir)

    wanted = list(dict.fromkeys(names + ([judge] if judge else [])))
    digests = preflight(url, wanted, timeout=timeout)
    replies: dict[str, list[dict[str, Any]]] = {}
    walls: dict[str, float] = {}
    for name in names:
        started = time.monotonic()
        replies[name] = run_model(
            url,
            name,
            resolved,
            system=system,
            seed=seed,
            num_predict=num_predict,
            timeout=timeout,
        )
        walls[name] = round(time.monotonic() - started, 1)
        answered = sum(1 for row in replies[name] if isinstance(row.get("reply"), str))
        _log(f"arena {name}: {answered}/{len(resolved)} answered in {walls[name]:.1f}s")

    judged = judge_rows(url, judge, replies, resolved, timeout=timeout) if judge else []
    board = leaderboard(names, replies, resolved, judged, self_judge=self_judge, rank_by=rank_by)

    replies_dir = out_dir / "replies"
    cases_path = out_dir / "cases.jsonl"
    judge_path = out_dir / "judge.jsonl"
    board_path = out_dir / "leaderboard.json"
    report_path = out_dir / "leaderboard.md"
    meta_path = out_dir / "meta.json"
    _write_jsonl(cases_path, [_case_record(case) for case in resolved])
    for name, slug in zip(names, slugs, strict=True):
        _write_jsonl(replies_dir / f"{slug}.jsonl", replies[name])
    _write_jsonl(judge_path, judged)
    _write_json(board_path, board)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_report(board) + "\n", encoding="utf-8")
    meta = {
        "schema": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": redact(url),
        "models": [{"name": name, "digest": digests.get(name)} for name in names],
        "judge": judge,
        "judge_digest": digests.get(judge) if judge else None,
        "rubric_sha256": hashlib.sha256(JUDGE_SYSTEM.encode("utf-8")).hexdigest(),
        "system": system,
        "request": {
            "think": False,
            "stream": False,
            "options": {"temperature": 0, "seed": seed, "num_predict": num_predict},
        },
        "cases_sha256": _sha256_file(cases_path),
        "git": _git_state(),
        "wall_s": walls,
        "cases": len(resolved),
        "n_ref": board["n_ref"],
        "n_contract": board["n_contract"],
        "self_judge": self_judge,
    }
    _write_json(meta_path, meta)

    judge_failed = any("judge_error" in row for row in judged)
    answered_all = all(row["answered"] == len(resolved) for row in board["rows"])
    board.update(
        {
            "endpoint": redact(url),
            "judge": judge,
            "wall_s": walls,
            "out": str(out_dir),
            "artifacts": {
                "cases": str(cases_path),
                "replies": {
                    name: str(replies_dir / f"{slug}.jsonl")
                    for name, slug in zip(names, slugs, strict=True)
                },
                "judge": str(judge_path),
                "leaderboard_json": str(board_path),
                "leaderboard_md": str(report_path),
                "meta": str(meta_path),
            },
            "ok": answered_all and not judge_failed,
        }
    )
    return board
