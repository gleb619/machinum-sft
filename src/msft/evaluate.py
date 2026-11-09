"""Protocol gates for logged model replies: pure text checks, no LLM call.

The shapes being checked come from the pipeline prompts (`llm.format.xml.md`,
`llm.format.properties.md`, `llm.format.plain.md`, `translate.system.md`):
one output unit per input line, `<result><id>N</id>...` blocks or `N=text` pairs with
contiguous 1-based ids, and an optional trailing `<terms>[...]</terms>` JSON array.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import MsftError

GATES = (
    "no_chatter",
    "xml_shape",
    "properties_shape",
    "line_parity",
    "terms_block",
)

#: Outcome of one check: ``None`` means "not applicable to this case".
Outcome = tuple[bool | None, str]

_RESULT_RE = re.compile(r"<result\b[^>]*>(.*?)</result>", re.S)
_ID_RE = re.compile(r"<id>\s*(\d+)\s*</id>")
_TERMS_RE = re.compile(r"<terms>\s*(.*?)\s*</terms>", re.S)
_PROPERTIES_RE = re.compile(r"^\s*(\d+)\s*=(.*)$")
_FENCE_RE = re.compile(r"```|~~~")
_CHATTER_RE = re.compile(
    r"^(?:sure|certainly|of course|here(?:'s| is| are)|below is"
    r"|thinking process|think\b|analy[sz]e the request|the user wants|let me "
    r"|вот |конечно|перевод:|хорошо,)",
    re.I,
)
# A reasoning transcript can appear anywhere, not only as the first line.
_THINK_BLOCK_RE = re.compile(r"(?:^|\n)\s*(?:thinking process\b|think\s*:)", re.I)
_XML_TERMS = ("text", "origin", "translation", "proofread")


@dataclass
class GateResult:
    """One protocol gate: did every applicable case pass, and at what rate."""

    gate: str
    passed: bool
    detail: str
    value: float | None = None


def _detect_mode(prompt: str, reply: str) -> str:
    """Pick the prompt's output mode from the logged prompt, falling back to the reply shape."""
    low = prompt.lower()
    if "xml mode" in low:
        return "xml"
    if "properties mode" in low:
        return "properties"
    if "plain mode" in low:
        return "plain"
    if "<result" in reply:
        return "xml"
    if any(_PROPERTIES_RE.match(line) for line in reply.splitlines() if line.strip()):
        return "properties"
    return "plain"


def _split_terms(reply: str) -> tuple[str, list[str]]:
    """Return the reply without its `<terms>` block(s), plus the raw block bodies."""
    return _TERMS_RE.sub("", reply), _TERMS_RE.findall(reply)


def _first_content_line(reply: str, mode: str) -> tuple[int, str]:
    """Index and text of the first line that looks like the mode's expected first token."""
    lines = reply.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if mode == "xml" and stripped.startswith("<"):
            return idx, stripped
        if mode == "properties" and _PROPERTIES_RE.match(stripped):
            return idx, stripped
        if mode == "plain":
            return idx, stripped
    return -1, ""


def _check_no_chatter(reply: str, mode: str) -> Outcome:
    """Fail on empty replies, code fences, chatty openers, think blocks and preamble text."""
    if not reply.strip():
        return False, "reply is empty"
    match = _FENCE_RE.search(reply)
    if match:
        return False, f"reply contains a code fence at offset {match.start()}"
    think = _THINK_BLOCK_RE.search(reply)
    if think:
        excerpt = reply[think.start() : think.start() + 60]
        return False, f"reasoning transcript at offset {think.start()}: {excerpt!r}"
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if lines and _CHATTER_RE.match(lines[0]):
        return False, f"chatty opener: {lines[0][:60]!r}"
    idx, token = _first_content_line(reply, mode)
    if idx < 0:
        return False, "no output line found"
    head = [line.strip() for line in reply.splitlines()[:idx] if line.strip()]
    if head:
        return False, f"{len(head)} preamble line(s) before {token[:40]!r}"
    return True, "no fences, no opener, no preamble"


def _check_xml_shape(reply: str, mode: str) -> Outcome:
    """Check `<result>` blocks: ids contiguous from 1, no duplicates, content child present."""
    stripped, _ = _split_terms(reply)
    blocks = _RESULT_RE.findall(stripped)
    if not blocks:
        if mode == "xml":
            return False, "no <result> blocks in the reply"
        return None, "not an xml case"
    ids: list[int] = []
    for pos, body in enumerate(blocks, 1):
        id_match = _ID_RE.search(body)
        if not id_match:
            return False, f"result block #{pos} has no <id>"
        ids.append(int(id_match.group(1)))
        if not any(f"<{field}>" in body for field in _XML_TERMS):
            return False, (
                f"result block #{pos} has no content child "
                f"({'|'.join(_XML_TERMS)})"
            )
    dupes = sorted({value for value in ids if ids.count(value) > 1})
    if dupes:
        return False, f"duplicate id(s): {dupes}"
    expected = list(range(1, len(ids) + 1))
    if ids != expected:
        return False, f"ids {ids[:10]} are not contiguous 1..{len(ids)}"
    return True, f"{len(ids)} result block(s) with ids 1..{len(ids)}"


def _check_properties_shape(reply: str, mode: str) -> Outcome:
    """Check `N=text` lines: every non-empty line parses and ids run 1..N in order."""
    if mode != "properties":
        return None, "not a properties case"
    stripped, _ = _split_terms(reply)
    ids: list[int] = []
    for line in stripped.splitlines():
        if not line.strip():
            continue
        match = _PROPERTIES_RE.match(line)
        if not match:
            return False, f"line is not of the form N=text: {line.strip()[:60]!r}"
        ids.append(int(match.group(1)))
    if not ids:
        return False, "no N=text lines in the reply"
    expected = list(range(1, len(ids) + 1))
    if ids != expected:
        return False, f"ids {ids[:10]} are not contiguous 1..{len(ids)}"
    return True, f"{len(ids)} N=text line(s)"


def _check_line_parity(reply: str, expected: Any) -> Outcome:
    """Compare the number of non-empty output lines with the expected count."""
    if expected is None:
        return None, "expected_lines not set"
    if isinstance(expected, bool) or not isinstance(expected, int):
        return False, f"expected_lines is not an integer: {expected!r}"
    want = expected
    stripped, _ = _split_terms(reply)
    got = len([line for line in stripped.splitlines() if line.strip()])
    if got != want:
        return False, f"expected {want} line(s), got {got}"
    return True, f"{got} line(s)"


def _check_terms(reply: str) -> Outcome:
    """Check at most one `<terms>` block whose body is a JSON array of term objects."""
    bodies = _TERMS_RE.findall(reply)
    if len(bodies) > 1:
        return False, f"{len(bodies)} <terms> blocks; expected at most one"
    if not bodies:
        if "<terms>" in reply:
            return False, "unterminated <terms> block"
        return True, "no <terms> block"
    body = bodies[0].strip()
    try:
        data = json.loads(body or "[]")
    except json.JSONDecodeError as exc:
        return False, f"<terms> is not valid JSON: {exc}"
    if not isinstance(data, list):
        return False, f"<terms> must be a JSON array, got {type(data).__name__}"
    for pos, item in enumerate(data, 1):
        if not isinstance(item, dict):
            return False, f"<terms> entry #{pos} is not an object"
        term = item.get("term")
        if not isinstance(term, str) or not term.strip():
            return False, f"<terms> entry #{pos} has no non-empty 'term' string"
    return True, f"<terms> array with {len(data)} entr{'y' if len(data) == 1 else 'ies'}"


def _evaluate_case(record: dict[str, Any], index: int) -> tuple[str, dict[str, Outcome]]:
    """Run every applicable check on one record, never raising."""
    name = str(record.get("case") or f"case {index}")
    if "_error" in record:
        message = str(record["_error"])
        return name, {gate: (False, message) for gate in GATES}
    reply = record.get("reply")
    if not isinstance(reply, str):
        return name, {gate: (False, "'reply' is missing or not a string") for gate in GATES}
    prompt = record.get("prompt_logged")
    if not isinstance(prompt, str):
        prompt = ""
    try:
        mode = _detect_mode(prompt, reply)
        outcomes: dict[str, Outcome] = {
            "no_chatter": _check_no_chatter(reply, mode),
            "xml_shape": _check_xml_shape(reply, mode),
            "properties_shape": _check_properties_shape(reply, mode),
            "line_parity": _check_line_parity(reply, record.get("expected_lines")),
            "terms_block": _check_terms(reply),
        }
    except Exception as exc:  # a malformed case must become a failed gate, not a traceback
        message = f"{type(exc).__name__}: {exc}"
        return name, {gate: (False, message) for gate in GATES}
    return name, outcomes


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into records; a bad line becomes a record that fails every gate."""
    cases: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            cases.append({"case": f"line {lineno}", "_error": f"invalid JSON: {exc}"})
            continue
        if not isinstance(record, dict):
            cases.append(
                {
                    "case": f"line {lineno}",
                    "_error": f"expected a JSON object, got {type(record).__name__}",
                }
            )
            continue
        cases.append(record)
    return cases


def _summarise(name: str, rows: list[tuple[str, bool, str]]) -> GateResult:
    """Collapse per-case outcomes into one gate result with the pass fraction as its value."""
    applicable = [(case, ok, detail) for case, ok, detail in rows if ok is not None]
    if not applicable:
        return GateResult(gate=name, passed=True, detail="not applicable in any case", value=None)
    failures = [(case, detail) for case, ok, detail in applicable if not ok]
    fraction = (len(applicable) - len(failures)) / len(applicable)
    if len(applicable) == 1:
        case, ok, detail = applicable[0]
        return GateResult(
            gate=name,
            passed=bool(ok),
            detail=f"{case}: {detail}"[:400],
            value=round(fraction, 4),
        )
    if failures:
        shown = "; ".join(f"{case}: {detail}" for case, detail in failures[:3])
        more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        detail = f"{len(failures)}/{len(applicable)} case(s) failed: {shown}{more}"
    else:
        detail = f"{len(applicable)}/{len(applicable)} case(s) passed"
    return GateResult(
        gate=name,
        passed=not failures,
        detail=detail[:400],
        value=round(fraction, 4),
    )


def run_gates(predictions: Path, *, as_json: bool = False) -> list[GateResult]:
    """Check a JSONL of `case`/`prompt_logged`/`reply` records and return one result per gate."""
    path = Path(predictions)
    if not path.exists():
        raise MsftError(f"predictions not found: {path}")
    cases = _load_cases(path)
    rows: dict[str, list[tuple[str, bool, str]]] = {gate: [] for gate in GATES}
    passed_cases = 0
    failed_names: list[str] = []
    for index, record in enumerate(cases, 1):
        name, outcomes = _evaluate_case(record, index)
        applicable = [ok for ok, _ in outcomes.values() if ok is not None]
        if applicable and all(applicable):
            passed_cases += 1
        else:
            failed_names.append(name)
        for gate, (ok, detail) in outcomes.items():
            rows.setdefault(gate, []).append((name, ok, detail))

    results = [_summarise(gate, rows[gate]) for gate in GATES]
    total = len(cases)
    fraction = passed_cases / total if total else 0.0
    if not total:
        detail = f"no prediction cases in {path.name}"
    elif failed_names:
        shown = ", ".join(failed_names[:5])
        more = f" (+{len(failed_names) - 5} more)" if len(failed_names) > 5 else ""
        detail = (
            f"{passed_cases}/{total} case(s) pass every applicable gate; "
            f"failing: {shown}{more}"
        )
    else:
        detail = f"{passed_cases}/{total} case(s) pass every applicable gate"
    results.append(
        GateResult(
            gate="schema_validity",
            passed=bool(total) and passed_cases == total,
            detail=detail,
            value=round(fraction, 4),
        )
    )

    if as_json:
        payload = {
            "predictions": str(path),
            "ok": all(result.passed for result in results),
            "schema_validity": round(fraction, 4),
            "cases": total,
            "gates": [asdict(result) for result in results],
        }
        print(json.dumps(payload, ensure_ascii=False))
    return results
