#!/usr/bin/env python3
"""Export summarizer and NER training triples from the shipped teacher tag.

There are no real summarize/NER prompt-reply exchanges on disk, so this is a *teacher export*
with the product's own prompts: the `summary.*` / `glossary.*` blocks the pipeline sends are
rendered here, answered by the shipped `machinum:4b` ollama tag at temperature 0 with
`think:false`, then normalised by the pipeline's TL254 `buildSftTriples`.

Plan by default; add `--execute` to send the requests:

    python scripts/export_protocol_triples.py --protocol all
    python scripts/export_protocol_triples.py --protocol all --execute

Raw exchanges land in `out/export/<protocol>/raw.jsonl`; the normalised triples land in
`data/summaries-v1/summaries.jsonl` and `data/ner-v1/ner.jsonl`.

The endpoint is resolved like the rest of the repo resolves the box, and the host is never
printed: display always uses `$MSFT_ENDPOINT`.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAPTERS = Path("/home/boris/WORKSPACE/machinum-sft-loop/books/book2")
DEFAULT_PROMPTS = Path("/home/boris/WORKSPACE/machinum-pipeline/packages/tools/prompts")
DEFAULT_TL254 = Path(
    "/home/boris/WORKSPACE/machinum-pipeline/packages/tools/dist/modification/sft-triples.js"
)
DEFAULT_TEACHER = "machinum:4b"
DEFAULT_OUT = REPO_ROOT / "out"
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_PORT = 7869
NOTE_COUNT = 10
MIN_RECORDS = 20
REQUEST_TIMEOUT_S = 600

PROTOCOLS = ("summary", "ner")
OUTPUT_PATHS = {
    "summary": ("summaries-v1", "summaries.jsonl"),
    "ner": ("ner-v1", "ner.jsonl"),
}
PROMPT_FILES = (
    "summary.system.md",
    "summary.user.md",
    "summary.output.bullets.md",
    "glossary.system.md",
    "glossary.user.md",
    "llm.format.plain.md",
)
CHAPTER_RE = re.compile(r"^chapter-(\d+)\.en\.md$")
FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.S)


# --- prompts -------------------------------------------------------------


def render(template: str, values: dict[str, str] | None = None) -> str:
    """Plain `{{var}}` substitution; unmatched slots stay untouched."""
    text = template
    for key, value in (values or {}).items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _read(prompts_dir: Path, name: str) -> str:
    return (prompts_dir / name).read_text(encoding="utf-8")


def build_prompts(protocol: str, prompts_dir: Path, chapter: str) -> tuple[str, str]:
    """The exact system/user pair the pipeline sends for this protocol."""
    if protocol == "summary":
        system = "\n\n".join(
            (
                render(_read(prompts_dir, "summary.system.md")),
                render(_read(prompts_dir, "llm.format.plain.md")),
                render(_read(prompts_dir, "summary.output.bullets.md")),
            )
        )
        user = render(
            _read(prompts_dir, "summary.user.md"),
            {"count": str(NOTE_COUNT), "text": chapter},
        )
        return system, user
    if protocol == "ner":
        system = render(_read(prompts_dir, "glossary.system.md"))
        user = render(
            _read(prompts_dir, "glossary.user.md"),
            {"known": "none", "candidates": "none", "text": chapter},
        )
        return system, user
    raise ValueError(f"unknown protocol: {protocol}")


def prompt_hashes(prompts_dir: Path) -> list[tuple[str, str]]:
    """sha256 of the six prompt blocks, in a stable order."""
    return [
        (name, hashlib.sha256((prompts_dir / name).read_bytes()).hexdigest())
        for name in PROMPT_FILES
    ]


# --- validation ----------------------------------------------------------


def _echo_form(text: str) -> str:
    """Whitespace/case-insensitive form, matching TL254's prompt-echo detection."""
    return " ".join(text.lower().split())


def clean_reply(protocol: str, reply: str) -> str:
    """Drop a lone markdown fence around an NER answer; keep the teacher's JSON itself."""
    text = reply.strip()
    if protocol == "ner":
        match = FENCE_RE.match(text)
        if match:
            text = match.group(1).strip()
    return text


def validate_reply(protocol: str, reply: str, user: str) -> tuple[bool, str]:
    """Accept only a usable answer: not empty, not a prompt echo, right shape."""
    text = reply.strip()
    if not text:
        return False, "empty reply"
    if _echo_form(text) == _echo_form(user):
        return False, "prompt echo"
    if protocol == "summary":
        bullets = [line for line in text.splitlines() if line.startswith("- ")]
        if len(bullets) < 3:
            return False, f"only {len(bullets)} bullet lines starting '- '"
        return True, "ok"
    if protocol == "ner":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc.msg}"
        if not isinstance(parsed, list):
            return False, "not a JSON array"
        if not parsed:
            return False, "empty JSON array"
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                return False, f"item {index} is not an object"
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                return False, f"item {index} has no name"
        return True, "ok"
    raise ValueError(f"unknown protocol: {protocol}")


# --- inputs and transport ------------------------------------------------


def parse_chapters(chapters_dir: Path, limit: int | None = None) -> list[tuple[int, Path]]:
    """`chapter-<N>.en.md`, numeric order; `index.md`/`plot.md` never match."""
    chapters: list[tuple[int, Path]] = []
    for path in chapters_dir.glob("chapter-*.en.md"):
        match = CHAPTER_RE.match(path.name)
        if match:
            chapters.append((int(match.group(1)), path))
    chapters.sort()
    return chapters[:limit] if limit is not None else chapters


def resolve_endpoint(explicit: str | None = None) -> str:
    """`--endpoint`, else `MSFT_EXPORT_ENDPOINT`, else the repo's own remote host + :7869."""
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("MSFT_EXPORT_ENDPOINT", "").strip()
    if env:
        return env.rstrip("/")
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from msft.remote import remote_host  # local import: the module stays stdlib-only

    host = remote_host().split("@")[-1]
    return f"http://{host}:{DEFAULT_PORT}"


def scrub_endpoint(text: str, endpoint: str) -> str:
    """Replace the real endpoint (and its host) with the display placeholder."""
    host = endpoint.split("://", 1)[-1].split("/", 1)[0]
    return text.replace(endpoint, "$MSFT_ENDPOINT").replace(host, "$MSFT_ENDPOINT")


def chat_once(endpoint: str, teacher: str, system: str, user: str, timeout: float) -> str:
    """One ollama `/api/chat` request with thinking disabled and deterministic sampling."""
    base = endpoint.rstrip("/")
    url = base if base.endswith("/api/chat") else f"{base}/api/chat"
    payload = {
        "model": teacher,
        "think": False,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - the caller retries and counts the failure
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from endpoint: {exc.msg}") from exc
    message = parsed.get("message") if isinstance(parsed, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("response has no message.content")
    return content


# --- output --------------------------------------------------------------


def write_atomic(path: Path, text: str) -> None:
    """Write `text` via a temp file in the same directory, then `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def run_tl254(raw_path: Path, tl254: Path) -> tuple[list[str], dict[str, Any]]:
    """Pipe raw.jsonl through the pipeline's TL254 `buildSftTriples` (node v22)."""
    snippet = (
        "import {buildSftTriples} from " + json.dumps(str(tl254)) + ";\n"
        "let data = '';\n"
        "process.stdin.setEncoding('utf8');\n"
        "process.stdin.on('data', (chunk) => { data += chunk; });\n"
        "process.stdin.on('end', () => {\n"
        "  const result = buildSftTriples(data);\n"
        "  process.stdout.write(JSON.stringify({jsonl: result.jsonl, stats: result.stats}));\n"
        "});\n"
    )
    raw = raw_path.read_text(encoding="utf-8")
    try:
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", snippet],
            input=raw,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("node is not on PATH (TL254 needs node v22)") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError(f"TL254 failed rc={proc.returncode}: {detail[-1] if detail else ''}")
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TL254 returned non-JSON: {exc.msg}") from exc
    lines = [line for line in parsed.get("jsonl", "").splitlines() if line.strip()]
    stats = parsed.get("stats") or {}
    return lines, stats


# --- export --------------------------------------------------------------


def export_protocol(
    protocol: str,
    chapters: list[tuple[int, Path]],
    prompts_dir: Path,
    endpoint: str,
    teacher: str,
    retries: int,
    out_dir: Path,
    tl254: Path,
    data_root: Path = DATA_ROOT,
    *,
    timeout: float = REQUEST_TIMEOUT_S,
    sleep: float = 1.0,
    min_records: int = MIN_RECORDS,
    chat: Any = chat_once,
) -> dict[str, Any]:
    """Run one protocol, write raw + normalised JSONL, and return its accounting."""
    started = time.monotonic()
    raw_records: list[dict[str, Any]] = []
    failed: list[int] = []
    stripped: list[int] = []

    for number, path in chapters:
        chapter = path.read_text(encoding="utf-8")
        system, user = build_prompts(protocol, prompts_dir, chapter)
        accepted: str | None = None
        reason = "no attempt"
        for attempt in range(retries + 1):
            try:
                reply = chat(endpoint, teacher, system, user, timeout)
            except Exception as exc:  # noqa: BLE001 - one chapter must not kill the export
                reply = None
                reason = f"request failed: {scrub_endpoint(str(exc), endpoint)}"
            if reply is not None:
                cleaned = clean_reply(protocol, reply)
                if cleaned != reply.strip():
                    if number not in stripped:
                        stripped.append(number)
                    print(f"clean chapter={number} reason=markdown-fence")
                reply = cleaned
                ok, reason = validate_reply(protocol, reply, user)
                if ok:
                    accepted = reply
                    break
            if attempt < retries:
                print(f"retry chapter={number} attempt={attempt + 1} reason={reason}")
                if sleep:
                    time.sleep(sleep)
        if accepted is None:
            print(f"FAIL chapter={number} reason={reason}")
            failed.append(number)
            continue
        print(f"ok chapter={number} chars={len(accepted)}")
        raw_records.append(
            {
                "protocol": protocol,
                "chapter": number,
                "system": system,
                "user": user,
                "reply": accepted,
            }
        )

    raw_path = out_dir / "export" / protocol / "raw.jsonl"
    raw_text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in raw_records)
    write_atomic(raw_path, raw_text)

    result: dict[str, Any] = {
        "protocol": protocol,
        "chapters": len(chapters),
        "accepted": len(raw_records),
        "failed": failed,
        "stripped": stripped,
        "raw": raw_path,
        "recordsIn": 0,
        "recordsOut": 0,
        "dropped": {},
        "triples": None,
        "lines": 0,
        "ok": False,
        "seconds": 0.0,
    }
    if not raw_records:
        result["seconds"] = time.monotonic() - started
        print(f"REFUSE {protocol}: no accepted records")
        return result

    try:
        lines, stats = run_tl254(raw_path, tl254)
    except RuntimeError as exc:
        result["seconds"] = time.monotonic() - started
        print(f"REFUSE {protocol}: {exc}")
        return result

    records_in = int(stats.get("recordsIn", 0))
    records_out = int(stats.get("recordsOut", 0))
    dropped = stats.get("dropped") or {}
    result.update(recordsIn=records_in, recordsOut=records_out, dropped=dropped)
    print(
        f"tl254 {protocol}: recordsIn={records_in} recordsOut={records_out} "
        f"dropped={json.dumps(dropped, sort_keys=True)}"
    )

    if records_out < min_records:
        print(f"REFUSE {protocol}: recordsOut={records_out} < {min_records}; triples not written")
    elif records_out != records_in:
        print(
            f"REFUSE {protocol}: recordsOut={records_out} != recordsIn={records_in}; "
            "partial dataset not written"
        )
    else:
        directory, filename = OUTPUT_PATHS[protocol]
        triples_path = data_root / directory / filename
        write_atomic(triples_path, "".join(line + "\n" for line in lines))
        result["triples"] = triples_path
        result["lines"] = len(lines)
        result["ok"] = not failed
        print(f"wrote {triples_path} lines={len(lines)}")

    result["seconds"] = time.monotonic() - started
    return result


def plan_protocol(protocol: str, chapters: list[tuple[int, Path]], prompts_dir: Path) -> None:
    """Print what would be sent, without touching the endpoint or writing anything."""
    for number, path in chapters:
        chapter = path.read_text(encoding="utf-8")
        system, user = build_prompts(protocol, prompts_dir, chapter)
        print(
            f"plan protocol={protocol} chapter={number} chars={len(chapter)} "
            f"system={len(system)} user={len(user)}"
        )


# --- CLI -----------------------------------------------------------------


def _selected_protocols(values: list[str] | None) -> list[str]:
    if not values:
        return list(PROTOCOLS)
    selected: list[str] = []
    for value in values:
        candidates = PROTOCOLS if value == "all" else (value,)
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        action="append",
        choices=[*PROTOCOLS, "all"],
        help="repeatable, or `all` (default: all)",
    )
    parser.add_argument("--teacher", default=DEFAULT_TEACHER, help="ollama tag to query")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="base URL or /api/chat URL; default: the repo's remote host on :7869",
    )
    parser.add_argument("--chapters", type=Path, default=DEFAULT_CHAPTERS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--tl254", type=Path, default=DEFAULT_TL254)
    parser.add_argument("--retries", type=int, default=3, help="retries after the first attempt")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N chapters")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="root for raw exchanges")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S)
    parser.add_argument("--execute", action="store_true", help="send the requests")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocols = _selected_protocols(args.protocol)

    if not args.chapters.is_dir():
        print(f"chapters directory not found: {args.chapters}", file=sys.stderr)
        return 2
    chapters = parse_chapters(args.chapters, args.limit)
    if not chapters:
        print(f"no chapter-<N>.en.md under {args.chapters}", file=sys.stderr)
        return 2
    if not args.prompts.is_dir():
        print(f"prompts directory not found: {args.prompts}", file=sys.stderr)
        return 2
    missing = [name for name in PROMPT_FILES if not (args.prompts / name).is_file()]
    if missing:
        print(f"missing prompt files: {', '.join(missing)}", file=sys.stderr)
        return 2

    for name, digest in prompt_hashes(args.prompts):
        print(f"prompt {name} sha256={digest}")

    if not args.execute:
        for protocol in protocols:
            plan_protocol(protocol, chapters, args.prompts)
        print(f"plan: {len(protocols) * len(chapters)} request(s) would be sent; pass --execute")
        return 0

    if not args.tl254.is_file():
        print(f"TL254 not found: {args.tl254}", file=sys.stderr)
        return 2
    endpoint = resolve_endpoint(args.endpoint)
    out_dir = args.out.resolve()
    print(
        f"execute endpoint=$MSFT_ENDPOINT teacher={args.teacher} "
        f"protocols={','.join(protocols)} chapters={len(chapters)}"
    )

    results = []
    for protocol in protocols:
        results.append(
            export_protocol(
                protocol,
                chapters,
                args.prompts,
                endpoint,
                args.teacher,
                args.retries,
                out_dir,
                args.tl254,
                timeout=args.timeout,
            )
        )

    ok = True
    for result in results:
        ok = ok and bool(result["ok"])
        print(
            f"summary {result['protocol']}: chapters={result['chapters']} "
            f"accepted={result['accepted']} failed={result['failed']} "
            f"stripped={result['stripped']} "
            f"recordsIn={result['recordsIn']} recordsOut={result['recordsOut']} "
            f"dropped={json.dumps(result['dropped'], sort_keys=True)} "
            f"lines={result['lines']} wall={result['seconds']:.1f}s"
        )
    print("RESULT: done" if ok else "RESULT: failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
