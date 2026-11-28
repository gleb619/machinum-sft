"""Tests for the protocol classifier: the anchors, their order, and what must not match."""

from __future__ import annotations

import pytest

from msft.protocols import PROTOCOLS, RULES, classify, classify_record, system_text

# Excerpts of the three real system prompts in `data/` (2026-11-28), plus the two export
# protocols. The excerpts keep the phrases the rules are anchored on, not the whole prompts.
TRANSLATE = (
    "# System Prompt: Literary Translator\n\nYou are a professional literary translator. "
    "Translate the user text from English to Russian. Output only the translation.\n\n"
    "After the translated lines, append exactly one <terms>[...]</terms> block: a JSON array "
    'of {"term","ru","type"} objects for English names or terms in the source that are NOT in '
    "the glossary above."
)
PROOFREAD = (
    "# SYSTEM PROMPT\n\nYou are an AI assistant that proofreads web novel chapters from "
    "English, using preformatted XML input. Each paragraph is wrapped in a `<result>` element."
)
TYPOFIX = (
    "Fix only obvious typos and punctuation mistakes. Keep the meaning, wording, [bracketed] "
    "system messages and the number of lines unchanged. Reply with the corrected text only."
)
SUMMARIZE = (
    "## System Prompt: Web Novel Summary Notes Generator\n\nYou are a precise and concise AI "
    "assistant tasked with producing plain text summary notes for a web novel's text.\n\n"
    "6. **Proofread:** Eliminate any grammatical or spelling mistakes before finalizing."
)
NER = (
    "# System Prompt: Web Novel Glossary Extractor\n\nYou are a precise and concise AI "
    "assistant tasked with producing a glossary of key terms, concepts, and entities.\n\n"
    "For the provided text you must analyze it to identify and define **notable terms**."
)

ARENA_TASKS = ("translate", "summarize", "ner", "proofread")


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        (TYPOFIX, "typofix"),
        (PROOFREAD, "proofread"),
        (NER, "ner"),
        (SUMMARIZE, "summarize"),
        (TRANSLATE, "translate"),
    ],
)
def test_classifies_the_real_prompts(system: str, expected: str) -> None:
    assert classify(system) == expected


def test_summarize_is_tested_before_proofread() -> None:
    # The summary prompt's checklist contains "Proofread:"; the generic rule must not win.
    assert "proofread" in SUMMARIZE.lower()
    assert classify(SUMMARIZE) == "summarize"


def test_translate_prompt_is_not_mistaken_for_ner() -> None:
    # The product translation prompt carries the same <terms> block the NER prompt uses, so
    # "<terms>" must not be an anchor on its own.
    assert "<terms>" in TRANSLATE
    assert classify(TRANSLATE) == "translate"
    assert classify("Only emit a <terms>[] block.") is None


def test_typofix_wins_over_proofread_when_both_phrases_appear() -> None:
    system = "Fix only obvious typos. You proofread nothing here."
    assert classify(system) == "typofix"


def test_unrelated_or_empty_prompts_are_unclassified() -> None:
    assert classify("") is None
    assert classify("You are a helpful assistant.") is None


def test_protocols_covers_every_arena_task_plus_typofix() -> None:
    assert set(ARENA_TASKS) <= set(PROTOCOLS)
    assert "typofix" in PROTOCOLS
    assert {protocol for protocol, _ in RULES} == set(PROTOCOLS)


def test_system_text_reads_field_and_messages() -> None:
    assert system_text({"system": "hello"}) == "hello"
    assert system_text({"messages": [{"role": "user", "content": "x"},
                                     {"role": "system", "content": "from messages"}]}) == (
        "from messages"
    )
    assert system_text({"user": "no system turn"}) == ""


def test_classify_record_accepts_both_record_shapes() -> None:
    assert classify_record({"system": NER, "user": "u", "reply": "r"}) == "ner"
    record = {"messages": [{"role": "system", "content": SUMMARIZE},
                           {"role": "user", "content": "u"},
                           {"role": "assistant", "content": "r"}]}
    assert classify_record(record) == "summarize"
    assert classify_record({"user": "u", "reply": "r"}) is None
