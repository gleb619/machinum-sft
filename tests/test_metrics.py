"""Tests for `msft.metrics`: pure functions, hand-computed values, no network."""

from __future__ import annotations

import pytest

from msft import metrics

# --- chrf -----------------------------------------------------------------


def test_chrf_identical_is_100_and_disjoint_is_0() -> None:
    assert metrics.chrf("привет мир", "привет мир") == 100.0
    assert metrics.chrf("abc", "xyz") == 0.0


def test_chrf_hand_computed_pair() -> None:
    # abcd/abce: P1=R1=0.75, P2=R2=2/3, P3=R3=0.5, P4=0; the mean is 23/48.
    assert metrics.chrf("abcd", "abce") == pytest.approx(47.9167, abs=1e-4)
    # ab/ac: only unigrams and bigrams survive; P=R=0.25, F=0.25.
    assert metrics.chrf("ab", "ac") == pytest.approx(25.0, abs=1e-4)


def test_chrf_ignores_whitespace() -> None:
    assert metrics.chrf("the cat\nsat", "the\tcat sat") == 100.0
    assert metrics.chrf(" a b ", "ab") == 100.0


def test_chrf_empty_input_and_short_pairs() -> None:
    assert metrics.chrf("", "abc") == 0.0
    assert metrics.chrf("abc", "") == 0.0
    assert metrics.chrf("", "") == 0.0
    # Two characters cannot form orders 3..6; the surviving orders still average to 100.
    assert metrics.chrf("ab", "ab") == 100.0
    assert metrics.chrf("a", "a", max_n=1) == 100.0


# --- bleu -----------------------------------------------------------------


def test_bleu_hand_computed_corpus_with_brevity_penalty() -> None:
    # c=6, r=8 -> BP=exp(-1/3); p=(4/6, 3/5, 2/3, 1) -> 100*BP*exp(mean(ln p)).
    score = metrics.bleu(
        ["the cat sat", "the the the"],
        ["the cat sat on the mat", "the dog"],
    )
    assert score == pytest.approx(51.4905, abs=1e-4)


def test_bleu_zero_unigram_match_is_zero() -> None:
    assert metrics.bleu(["q"], ["z"]) == 0.0
    assert metrics.bleu([], []) == 0.0


def test_bleu_requires_paired_lists() -> None:
    with pytest.raises(ValueError):
        metrics.bleu(["a"], [])


# --- script flags ---------------------------------------------------------


def test_script_flags_cjk_token_inside_cyrillic() -> None:
    flags = metrics.script_flags("палубный甲板 стоит", "the deck stands")
    assert flags["mixed_script_tokens"] == 1
    assert flags["cyrillic_ratio"] == pytest.approx(13 / 15, abs=1e-4)
    assert flags["source_leak_lines"] == 0
    assert flags["length_ratio"] == pytest.approx(16 / 15, abs=1e-4)


def test_script_flags_counts_a_copied_source_line() -> None:
    flags = metrics.script_flags(
        "The deck stands\nпалуба и мачта", "The deck stands\nпалуба и мачта"
    )
    assert flags["source_leak_lines"] == 2
    assert flags["mixed_script_tokens"] == 0


def test_script_flags_without_letters() -> None:
    flags = metrics.script_flags("123 456", "")
    assert flags["cyrillic_ratio"] == 0.0
    assert flags["mixed_script_tokens"] == 0
    assert flags["source_leak_lines"] == 0
    assert flags["length_ratio"] == 0.0


# --- glossary -------------------------------------------------------------


def test_glossary_hits_hit_miss_and_no_terms() -> None:
    terms = [{"src": "deck", "tgt": "палуба"}, {"src": "mast", "tgt": "мачта"}]
    assert metrics.glossary_hits("палуба и всё", terms) == 0.5
    assert metrics.glossary_hits("палуба и мачта", terms) == 1.0
    assert metrics.glossary_hits("ничего", terms) == 0.0
    assert metrics.glossary_hits("палуба", []) is None


# --- judge parsing --------------------------------------------------------


def test_parse_judge_clean_json() -> None:
    assert metrics.parse_judge('{"fidelity": 4, "fluency": 5}') == {
        "fidelity": 4,
        "fluency": 5,
    }


def test_parse_judge_fenced_and_chatty() -> None:
    fenced = '```json\n{"fidelity": 3, "fluency": 4}\n```'
    assert metrics.parse_judge(fenced) == {"fidelity": 3, "fluency": 4}
    chatty = 'Sure! Here is my rating: {"fidelity": 5, "fluency": 5} Done.'
    assert metrics.parse_judge(chatty) == {"fidelity": 5, "fluency": 5}


def test_parse_judge_string_and_integral_float_scores() -> None:
    assert metrics.parse_judge('{"fidelity": "4", "fluency": 5.0}') == {
        "fidelity": 4,
        "fluency": 5,
    }


def test_parse_judge_ignores_braces_inside_strings() -> None:
    text = '{"fidelity": 4, "fluency": 5, "note": "a } brace"}'
    assert metrics.parse_judge(text) == {"fidelity": 4, "fluency": 5}


def test_parse_judge_rejects_out_of_range_missing_and_garbage() -> None:
    assert metrics.parse_judge('{"fidelity": 6, "fluency": 3}') is None
    assert metrics.parse_judge('{"fidelity": 0, "fluency": 3}') is None
    assert metrics.parse_judge('{"fidelity": 4}') is None
    assert metrics.parse_judge("no json here") is None
    assert metrics.parse_judge('{"fidelity": true, "fluency": 3}') is None
