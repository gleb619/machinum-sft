"""Serve-path smoke test: `msft probe`.

Posts one ``/api/chat`` request per case with ``think: false`` and ``stream: false``, writes the
replies to ``probe-predictions.jsonl`` next to the cases file, then runs the same gates as
``msft eval``. This is the check that made the 4B artifact pass: without ``think: false`` the
same GGUF answers with a reasoning transcript, and without the GGUF's own chat template it
answers in a format it never saw.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from msft.config import MsftError, redact
from msft.evaluate import run_gates

SYSTEM_PROMPT = "Translate English prose into Russian. Reply with the translation only."
REQUEST_TIMEOUT_S = 120
PREDICTIONS_NAME = "probe-predictions.jsonl"


def _log(message: str) -> None:
    """Progress line on stderr, so a `--json` run keeps stdout to one object."""
    print(redact(message), file=sys.stderr)


def chat_url(endpoint: str) -> str:
    """The ``/api/chat`` URL for a base endpoint (or the endpoint itself when already complete)."""
    base = endpoint.rstrip("/")
    return base if base.endswith("/api/chat") else f"{base}/api/chat"


def build_request(model: str, prompt: str, system: str) -> dict[str, Any]:
    """One ollama chat request with thinking disabled and deterministic sampling."""
    return {
        "model": model,
        "think": False,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST a JSON body with stdlib urllib and parse the JSON reply."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise MsftError(f"endpoint returned {type(parsed).__name__}, expected an object")
    return parsed


def _reply(data: dict[str, Any]) -> str:
    """The assistant text from an ollama ``/api/chat`` response."""
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(data.get("response"), str):
        return data["response"]
    raise MsftError("response has no message.content")


def _prompt(record: dict[str, Any]) -> str | None:
    """The user text of a case: ``prompt``, ``user`` or the last user message."""
    for key in ("prompt", "user"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL cases file; a bad line becomes a record that fails every gate."""
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


def run(
    endpoint: str,
    model: str,
    cases_path: str | Path,
    *,
    system: str = SYSTEM_PROMPT,
    timeout: float = REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Probe every case, write predictions, run the eval gates; returns a verdict."""
    path = Path(cases_path)
    if not path.is_file():
        raise MsftError(f"cases not found: {path}")
    url = chat_url(endpoint)
    records = _load_cases(path)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        name = str(record.get("case") or f"case {index}")
        if "_error" in record:
            rows.append({"case": name, "_error": str(record["_error"])})
            continue
        prompt = _prompt(record)
        if prompt is None:
            rows.append(
                {"case": name, "_error": "case has no prompt, user or messages text"}
            )
            continue
        logged = record.get("prompt_logged")
        row: dict[str, Any] = {
            "case": name,
            "prompt_logged": logged if isinstance(logged, str) else prompt,
        }
        if "expected_lines" in record:
            row["expected_lines"] = record["expected_lines"]
        try:
            data = _post_json(url, build_request(model, prompt, system), timeout)
            row["reply"] = _reply(data)
        except Exception as exc:  # noqa: BLE001 - one bad case must not kill the probe
            row["_error"] = f"{type(exc).__name__}: {exc}"
            _log(f"{name}: request failed: {row['_error']}")
        rows.append(row)

    predictions = path.parent / PREDICTIONS_NAME
    predictions.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    predictions.write_text(body + ("\n" if rows else ""), encoding="utf-8")

    gates = run_gates(predictions)
    gate_rows = [asdict(gate) for gate in gates]
    passed = bool(gate_rows) and all(bool(gate["passed"]) for gate in gate_rows)
    return {
        "ok": passed,
        "endpoint": redact(url),
        "model": model,
        "cases": len(rows),
        "predictions": str(predictions),
        "system": system,
        "request": {"think": False, "stream": False, "options": {"temperature": 0}},
        "gates": gate_rows,
    }


def format_report(result: dict[str, Any]) -> str:
    """One gate line per check plus the green/red verdict."""
    lines = [
        f"probe {result['endpoint']} model={result['model']} cases={result['cases']}",
        f"  predictions: {result['predictions']}",
    ]
    lines.extend(
        f"{'PASS' if gate['passed'] else 'FAIL'} {gate['gate']}: {gate['detail']}"
        for gate in result["gates"]
    )
    lines.append(
        f"gates {'green' if result['ok'] else 'red'} ({len(result['gates'])} checked)"
    )
    return redact("\n".join(lines))
