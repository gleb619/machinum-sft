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


#: Gates each task is scored on. ``translate`` is the translation contract and keeps every
#: check. The other protocols carry no ``<terms>``, XML or ``N=text`` contract, so
#: ``no_chatter`` is the only universal one, plus ``line_parity`` where the output is one unit
#: per input unit (proofread items, typofix lines). Measured on the W4 cases (2026-11-28): the
#: gold reference of every non-translate protocol failed ``terms_block`` 24/24 and therefore
#: ``schema_validity`` 0/24 — a contract those protocols were never asked for. ``schema_validity``
#: is absent here on purpose: it is the roll-up over whichever checks apply.
TASK_GATES: dict[str, tuple[str, ...]] = {
    "translate": ("no_chatter", "xml_shape", "properties_shape", "line_parity", "terms_block"),
    "summarize": ("no_chatter",),
    "ner": ("no_chatter",),
    "proofread": ("no_chatter", "line_parity"),
    "typofix": ("no_chatter", "line_parity"),
}

#: Task a case is scored as when its ``task`` is missing or not one of the protocols.
DEFAULT_TASK = "translate"


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


def task_label(task: Any) -> str:
    """The scoring label for a case's ``task``: a known protocol, else `DEFAULT_TASK`."""
    if isinstance(task, str):
        candidate = task.strip().lower()
        if candidate in TASK_GATES:
            return candidate
    return DEFAULT_TASK


def gates_for(task: Any) -> tuple[str, ...]:
    """Gates a case with this ``task`` is scored on; unknown or missing means translate."""
    return TASK_GATES[task_label(task)]
