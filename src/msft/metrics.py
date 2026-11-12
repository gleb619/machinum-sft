"""Scoring primitives for `msft arena`: pure functions, stdlib only, no I/O.

These are internal variants, not paper implementations: ``chrf`` drops whitespace before
counting character n-grams, ``bleu`` uses add-one smoothing above unigram and a corpus
brevity penalty, and ``script_flags`` keeps four independent numbers instead of folding them
into one "quality" score. ``contract`` is the same idea applied to the instruction: it
measures whether the reply has the shape the prompt asked for, separately from whether the
words are right. ``docs/arena.md`` says why.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_FENCE_RE = re.compile(r"```|~~~")

# Cyrillic and its supplements; everything else falls back to the Unicode character name.
_CYRILLIC_RANGES = (
    (0x0400, 0x052F),
    (0x1C80, 0x1C8F),
    (0x2DE0, 0x2DFF),
    (0xA640, 0xA69F),
)


def _normalise(text: str) -> str:
    """NFC-normalise and drop every whitespace character, without changing case."""
    return "".join(ch for ch in unicodedata.normalize("NFC", text) if not ch.isspace())


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """Character n-gram counts of an already-normalised string."""
    return Counter(text[index : index + n] for index in range(len(text) - n + 1))


def chrf(hyp: str, ref: str, *, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score in 0..100 for one hypothesis against one reference."""
    hypothesis = _normalise(hyp)
    reference = _normalise(ref)
    if not hypothesis or not reference:
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for n in range(1, max_n + 1):
        hyp_grams = _char_ngrams(hypothesis, n)
        ref_grams = _char_ngrams(reference, n)
        hyp_total = sum(hyp_grams.values())
        ref_total = sum(ref_grams.values())
        if not hyp_total or not ref_total:
            continue
        matched = sum(min(count, ref_grams[gram]) for gram, count in hyp_grams.items())
        precisions.append(matched / hyp_total)
        recalls.append(matched / ref_total)
    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if not denominator:
        return 0.0
    score = (1 + beta_sq) * precision * recall / denominator
    return round(100.0 * score, 4)


def bleu(hyps: list[str], refs: list[str], *, max_n: int = 4) -> float:
    """Corpus BLEU in 0..100 over paired hypotheses and references."""
    if len(hyps) != len(refs):
        raise ValueError("bleu needs one reference per hypothesis")
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_tokens = 0
    ref_tokens = 0
    for hyp, ref in zip(hyps, refs, strict=True):
        hypothesis = _TOKEN_RE.findall(hyp)
        reference = _TOKEN_RE.findall(ref)
        hyp_tokens += len(hypothesis)
        ref_tokens += len(reference)
        for n in range(1, max_n + 1):
            hyp_grams = Counter(
                tuple(hypothesis[index : index + n])
                for index in range(len(hypothesis) - n + 1)
            )
            ref_grams = Counter(
                tuple(reference[index : index + n]) for index in range(len(reference) - n + 1)
            )
            matches[n - 1] += sum(
                min(count, ref_grams[gram]) for gram, count in hyp_grams.items()
            )
            totals[n - 1] += sum(hyp_grams.values())
    if not totals[0] or not matches[0]:
        return 0.0
    precisions = [
        matched / total if order == 1 else (matched + 1) / (total + 1)
        for order, (matched, total) in enumerate(zip(matches, totals, strict=True), 1)
    ]
    log_sum = sum(math.log(value) for value in precisions)
    brevity = 1.0 if hyp_tokens > ref_tokens else math.exp(1 - ref_tokens / hyp_tokens)
    return round(100.0 * brevity * math.exp(log_sum / max_n), 4)


def _script(ch: str) -> str:
    """Lower-case script name for a letter, by range first and Unicode name second."""
    code = ord(ch)
    if any(low <= code <= high for low, high in _CYRILLIC_RANGES):
        return "cyrillic"
    name = unicodedata.name(ch, "")
    return name.split(" ", 1)[0].lower() if name else "other"


def script_flags(reply: str, source: str) -> dict[str, float | int]:
    """Four independent script signals for one reply; never folded into a single number."""
    letters = [ch for ch in reply if ch.isalpha()]
    cyrillic = sum(1 for ch in letters if _script(ch) == "cyrillic")
    mixed = 0
    for token in reply.split():
        scripts = {_script(ch) for ch in token if ch.isalpha()}
        if len(scripts) > 1:
            mixed += 1
    source_lines = {line.strip() for line in source.splitlines() if line.strip()}
    leaked = sum(
        1
        for line in reply.splitlines()
        if line.strip() and line.strip() in source_lines
    )
    length_ratio = len(reply) / len(source) if source else 0.0
    return {
        "cyrillic_ratio": round(cyrillic / len(letters), 4) if letters else 0.0,
        "mixed_script_tokens": mixed,
        "source_leak_lines": leaked,
        "length_ratio": round(length_ratio, 4),
    }


def glossary_hits(reply: str, terms: list[dict[str, str]]) -> float | None:
    """Fraction of the term list's target strings present in the reply, or None without terms."""
    targets = [
        term["tgt"]
        for term in terms
        if isinstance(term, dict) and isinstance(term.get("tgt"), str) and term["tgt"]
    ]
    if not targets:
        return None
    present = sum(1 for target in targets if target in reply)
    return round(present / len(targets), 4)


def _first_object(text: str) -> str | None:
    """The first balanced ``{...}`` in the text, ignoring braces inside JSON strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _score(value: object) -> int | None:
    """A 1..5 score from an int, an integral float or a digit string; anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        return None
    return number if 1 <= number <= 5 else None


def parse_judge(text: str) -> dict[str, int] | None:
    """``{"fidelity": n, "fluency": n}`` from a judge reply, or None when it is not scorable."""
    body = _first_object(_FENCE_RE.sub("", text))
    if body is None:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    fidelity = _score(data.get("fidelity"))
    fluency = _score(data.get("fluency"))
    if fidelity is None or fluency is None:
        return None
    return {"fidelity": fidelity, "fluency": fluency}


# --- the instruction contract --------------------------------------------
#
# Everything above scores the words. This section scores the SHAPE: a summariser that returns
# a paragraph where the prompt asked for a bullet list has failed the request even when every
# word is right, and no translation metric sees that. Each format is a small rule list, and
# the score is the fraction of rules the reply satisfies so a near-miss stays visible.

#: The output formats a case's ``expect.format`` may name.
CONTRACT_FORMATS = ("bullet_list", "json_object", "term_table", "edit_pairs", "plain_text")

#: The three bullet markers the pipeline prompts use: ``- ``, ``* `` and ``1. ``.
_BULLET_RE = re.compile(r"^(?:[-*]\s+|\d{1,3}\.\s+)")
_HEADING_RE = re.compile(r"^#{1,6}(?:\s|$)")
_BOLD_RE = re.compile(r"\*\*|__[^_]+__")
#: ``term -> translation``, ``term: translation`` and ``term — translation``; arrow first so a
#: source term that itself contains a colon still splits on the arrow.
_PAIR_RE = re.compile(r"^(?P<src>.+?)\s*(?:->|→|—|:)\s*(?P<tgt>.+)$")

Rule = tuple[str, bool, str]


def _int_rule(expect: dict[str, Any], name: str, default: int) -> int:
    """A non-negative integer rule parameter, defaulted when the record does not carry one."""
    value = expect.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _fenced_region(text: str) -> tuple[str, str, str]:
    """The body of the first fenced block, plus the text before and after it.

    A fence is stripped rather than penalised: ```json around the object is a shape the
    instruction asked for. Text outside the fence is still chatter and still a violation.
    """
    start = text.find("```")
    end = text.find("```", start + 3) if start >= 0 else -1
    if start < 0 or end < 0:
        return text, "", ""
    body = text[start + 3 : end]
    lines = body.split("\n")
    # Skip a language tag (```json) when the object starts on the next line.
    if lines and lines[0].strip() and "{" not in lines[0]:
        lines = lines[1:]
    return "\n".join(lines), text[:start], text[end + 3 :]


def _bullet_checks(reply: str, expect: dict[str, Any]) -> list[Rule]:
    """Bullets only, enough of them, and at most a one-line lead-in before the list."""
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    bullets = [line for line in lines if _BULLET_RE.match(line)]
    minimum = _int_rule(expect, "bullets_min", 3)
    first = next((index for index, line in enumerate(lines) if _BULLET_RE.match(line)), None)
    lead = 0 if first is None else first
    strays = (
        0
        if first is None
        else sum(1 for line in lines[first:] if not _BULLET_RE.match(line))
    )
    return [
        (
            "bullets_only",
            bool(bullets) and not strays,
            "no bullet line"
            if not bullets
            else f"{strays} non-bullet line(s) inside the list",
        ),
        (
            "bullets_min",
            len(bullets) >= minimum,
            f"{len(bullets)} bullet(s) < bullets_min {minimum}",
        ),
        (
            "lead_in",
            first is not None and lead <= 1,
            "no bullet list" if first is None else f"{lead} prose line(s) before the list",
        ),
    ]


def _json_checks(reply: str, expect: dict[str, Any]) -> list[Rule]:
    """One JSON object, every required key, and no chatter on either side of it.

    Chatter is reported as a violation and never as a parse failure: "Here you go:" before a
    perfectly good object is a format breach, and the leaderboard should say which one.
    """
    body, before, after = _fenced_region(reply)
    found = _first_object(body)
    required = expect.get("keys")
    if found is None:
        # No object at all: only the two rules that can be judged are listed, so a reply with
        # no JSON scores 0 instead of collecting free points for "no chatter around" nothing.
        return [
            ("json_object", False, "no JSON object found in the reply"),
            (
                "keys",
                False,
                f"no object to check {len(required or [])} required key(s) in",
            ),
        ]
    parsed: object = None
    try:
        parsed = json.loads(found)
    except json.JSONDecodeError:
        parsed = None
    offset = body.find(found)
    head = (before + body[:offset]).strip()
    tail = (body[offset + len(found) :] + after).strip()
    missing: list[str] = []
    if isinstance(required, list):
        missing = [
            str(key)
            for key in required
            if not isinstance(parsed, dict) or str(key) not in parsed
        ]
    return [
        ("json_object", isinstance(parsed, dict), "reply does not contain one JSON object"),
        (
            "keys",
            not missing,
            f"missing key(s): {', '.join(missing[:5])}"
            if missing
            else f"{len(required or [])} required key(s) present",
        ),
        ("no_leading_chatter", not head, f"text before the object: {head[:40]!r}"),
        ("no_trailing_chatter", not tail, f"text after the object: {tail[:40]!r}"),
    ]


def _pair_lines(reply: str, frozen: str | None = None) -> tuple[list[tuple[str, str]], list[str]]:
    """Parsed ``source -> target`` pairs, plus the non-empty lines that do not have that shape.

    A leading bullet marker is stripped first: ``- deck -> палуба`` is the same pair as
    ``deck -> палуба``, and rejecting it would score punctuation, not compliance. ``frozen`` is
    the line the case says must not change; it is exempt from the pair shape because it is not
    an edit, it is the evidence that the proofreader left that line alone.
    """
    pairs: list[tuple[str, str]] = []
    bad: list[str] = []
    for line in (line.strip() for line in reply.splitlines()):
        if not line:
            continue
        if frozen is not None and line == frozen:
            continue
        match = _PAIR_RE.match(_BULLET_RE.sub("", line, count=1).strip())
        if match is None:
            bad.append(line)
            continue
        pairs.append((match.group("src").strip(), match.group("tgt").strip()))
    return pairs, bad


def _shape_detail(pairs: list[tuple[str, str]], bad: list[str]) -> str:
    """One short line saying why the pair shape failed."""
    if not pairs:
        return "no pair lines"
    if not bad:
        return f"{len(pairs)} pair line(s)"
    return f"{len(bad)} line(s) are not `term -> translation`: {bad[0][:40]!r}"


def _table_checks(reply: str, expect: dict[str, Any]) -> list[Rule]:
    """Term table: every line a pair, no repeated source term, at least ``pairs_min``."""
    pairs, bad = _pair_lines(reply)
    minimum = _int_rule(expect, "pairs_min", 1)
    seen: set[str] = set()
    reported: set[str] = set()
    duplicates: list[str] = []
    for source, _ in pairs:
        key = source.casefold()
        if key in seen and key not in reported:
            reported.add(key)
            duplicates.append(source)
        seen.add(key)
    return [
        ("pairs_shaped", bool(pairs) and not bad, _shape_detail(pairs, bad)),
        (
            "no_duplicate_terms",
            not duplicates,
            f"duplicate source term(s): {', '.join(duplicates[:5])}",
        ),
        (
            "pairs_min",
            len(pairs) >= minimum,
            f"{len(pairs)} pair(s) < pairs_min {minimum}",
        ),
    ]


def _edit_checks(reply: str, expect: dict[str, Any]) -> list[Rule]:
    """Edit pairs: every line ``old -> new``, no conflicting edits, and the frozen line intact.

    ``old -> old`` is allowed and counted as an unchanged line, which is why this rule is
    "no conflicting edits" rather than term_table's "no duplicate source term": a proofread
    reply may legitimately repeat a word it left alone.
    """
    needle = expect.get("unchanged_line")
    frozen = needle.strip() if isinstance(needle, str) and needle.strip() else None
    pairs, bad = _pair_lines(reply, frozen)
    minimum = _int_rule(expect, "pairs_min", 1)
    first_target: dict[str, str] = {}
    reported: set[str] = set()
    conflicts: list[str] = []
    for source, target in pairs:
        key = source.casefold()
        previous = first_target.get(key)
        if previous is None:
            first_target[key] = target
        elif previous != target and key not in reported:
            reported.add(key)
            conflicts.append(source)
    checks: list[Rule] = [
        ("pairs_shaped", bool(pairs) and not bad, _shape_detail(pairs, bad)),
        (
            "no_conflicting_edits",
            not conflicts,
            f"conflicting edit(s) for: {', '.join(conflicts[:5])}",
        ),
        (
            "pairs_min",
            len(pairs) >= minimum,
            f"{len(pairs)} pair(s) < pairs_min {minimum}",
        ),
    ]
    if expect.get("unchanged") is True:
        needle = expect.get("unchanged_line")
        if not isinstance(needle, str) or not needle:
            checks.append(
                (
                    "unchanged_line",
                    False,
                    "expect.unchanged is true but expect.unchanged_line is missing",
                )
            )
        else:
            checks.append(
                (
                    "unchanged_line",
                    needle in reply,
                    f"line that must not change is missing: {needle[:40]!r}",
                )
            )
    return checks


def _plain_checks(reply: str, expect: dict[str, Any]) -> list[Rule]:
    """`just the text`: no fence, no bold, no heading, no bullet."""
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    heading = next((line for line in lines if _HEADING_RE.match(line)), "")
    bullet = next((line for line in lines if _BULLET_RE.match(line)), "")
    return [
        ("no_fence", not _FENCE_RE.search(reply), "reply contains a code fence"),
        ("no_bold", not _BOLD_RE.search(reply), "reply contains **bold** markup"),
        ("no_heading", not heading, f"markdown heading: {heading[:40]!r}"),
        ("no_bullet", not bullet, f"bullet line: {bullet[:40]!r}"),
    ]


_CONTRACT_CHECKS = {
    "bullet_list": _bullet_checks,
    "json_object": _json_checks,
    "term_table": _table_checks,
    "edit_pairs": _edit_checks,
    "plain_text": _plain_checks,
}


def contract(reply: str, expect: dict[str, Any] | None) -> tuple[float | None, list[str]]:
    """Format-compliance score in 0..1 and the rules the reply broke, or ``(None, [])``.

    The score is the fraction of the format's rules the reply satisfies, not a pass/fail:
    a bullet list carrying two of the three required bullets scores 0.667, so the distance
    from the contract stays visible instead of collapsing to a boolean. A case that carries
    no ``expect`` has no contract to break, so its score is ``None`` and the leaderboard
    leaves it out of the mean rather than averaging in a zero it did not earn.
    """
    if not expect:
        return None, []
    fmt = str(expect.get("format") or "")
    checker = _CONTRACT_CHECKS.get(fmt)
    if checker is None:
        return 0.0, [f"unknown format: {fmt!r}"]
    if not reply.strip():
        # An empty reply satisfies no format; scoring its four absent properties as "fine"
        # would give a model 0.8 for saying nothing.
        return 0.0, ["non_empty: reply is empty"]
    checks = checker(reply, expect)
    violations = [f"{label}: {detail}" for label, ok, detail in checks if not ok]
    passed = sum(1 for _, ok, _ in checks if ok)
    return round(passed / len(checks), 4), violations
