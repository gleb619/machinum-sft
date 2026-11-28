"""Protocol classifier: which SFT protocol a training record belongs to.

Every artifact this repo trains on carries the system prompt the teacher was given, and that
prompt is the only anchor shared by all of them: the ``source`` field names the protocol in
some files and is absent in others. So classification is an ordered, case-insensitive regex
table over the system turn, first match wins, ``None`` meaning "this record claims no
protocol".

The order is not cosmetic, and neither is the choice of anchor. Measured on the 341 local
records (2026-11-28) the literal table in ``out/weekend/W4/PLAN.md`` §3.2 misfires on all of
today's artifacts: the translation prompt grew the same ``<terms>`` block the NER prompt was
anchored on, the summary prompt contains "Proofread" in its checklist, and the NER prompt
says "notable terms"/"Glossary Extractor" rather than "named entities". So the anchors below
are the specific ones that survive: ``summar`` is tested before ``proofread``, NER by its own
title phrase, proofread in the present tense ("proofreads web novel chapters").
"""

from __future__ import annotations

import re
from typing import Any

#: Canonical protocol names, in classifier order. ``translate``/``summarize``/``ner``/
#: ``proofread`` are the four task labels `arena` scores; ``typofix`` is the fifth slice.
PROTOCOLS = ("typofix", "summarize", "ner", "proofread", "translate")

#: ``(protocol, pattern)`` in the order they are tried. Anchors are quoted from the prompts.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("typofix", re.compile(r"fix only obvious typos", re.I)),
    ("summarize", re.compile(r"summar", re.I)),
    ("ner", re.compile(r"notable terms|glossary extractor", re.I)),
    ("proofread", re.compile(r"proofreads", re.I)),
    ("translate", re.compile(r"translate|translator", re.I)),
)


def system_text(record: dict[str, Any]) -> str:
    """The system turn of a raw record: the ``system`` field, else the first system message."""
    raw = record.get("system")
    if isinstance(raw, str) and raw.strip():
        return raw
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def classify(system: str) -> str | None:
    """Protocol named by a system prompt, or None when no rule matches."""
    for protocol, pattern in RULES:
        if pattern.search(system):
            return protocol
    return None


def classify_record(record: dict[str, Any]) -> str | None:
    """Protocol for a raw record, read from whichever system turn it carries."""
    return classify(system_text(record))
