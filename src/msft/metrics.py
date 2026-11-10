"""Scoring primitives for `msft arena`: pure functions, stdlib only, no I/O.

These are internal variants, not paper implementations: ``chrf`` drops whitespace before
counting character n-grams, ``bleu`` uses add-one smoothing above unigram and a corpus
brevity penalty, and ``script_flags`` keeps four independent numbers instead of folding them
into one "quality" score. ``docs/arena.md`` says why.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter

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
