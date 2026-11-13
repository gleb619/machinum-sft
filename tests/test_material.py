"""Tests for `msft material` — the daily research state machine. Offline: no network, no docker.

Every remote call goes through two module-level seams (`run_webbridge`, `post_page`) and the
distill step goes through `MSFT_AT_BIN`, which points at a fake script here. Nothing in this
file opens a socket.
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from msft import material
from msft.cli import main
from msft.material import ConfigError

DAY = "2026-11-10"

# The autouse `offline_probes` fixture replaces `material.run_webbridge` module-wide; the one
# test that verifies the seam itself needs the original function.
REAL_RUN_WEBBRIDGE = material.run_webbridge

BASE_CONFIG: dict[str, Any] = {
    "product": {
        "name": "machinum-sft",
        "brief": "A small local model that translates English prose into Russian.",
        "keywords": ["lora", "qlora", "peft", "sft", "dataset", "vram"],
        "negative": ["hiring", "webinar"],
        "domain_terms": [
            "llm", "language model", "fine-tun", "adapter", "peft", "qlora", "neural",
            "transformer", "нейросет", "модел", "обучен", "токениз", "весов",
        ],
    },
    "seeds": [
        {"kind": "query", "site": "habr", "value": "LoRA fine-tuning LLM"},
        {"kind": "query", "site": "github", "value": "lora fine-tuning"},
        {"kind": "query", "site": "hf", "value": "lora"},
    ],
    "limits": {
        "queries_per_run": 6,
        "items_per_run": 12,
        "items_per_site": 6,
        "max_frontier": 200,
        "children_per_item": 3,
        "fetch_timeout_s": 120,
        "ask_timeout_s": 240,
        # Pinned to the old one-shot behaviour so a count in an older test still means what
        # it meant. The shipped defaults (3 / true) are covered by their own tests below,
        # which override these two keys explicitly.
        "refresh_days": 0,
        "synthesize_queries": False,
    },
    "scoring": {
        "keep_min": 0.35,
        "expand_min": 0.60,
        "recency_days": [30, 180],
        "site_prior": {"github": 0.05, "hf": 0.05, "habr": 0.0},
    },
    "expansion": {
        "allow_kinds": ["tag", "author", "repo", "model"],
        "per_site": {"habr": ["tag", "author"], "github": ["repo"], "hf": ["model"]},
    },
    "distill": {"agent": "ds-code", "timeout_ms": 900000, "notes_dir": "notes"},
    "memory": {
        "workspace": "default",
        "project": "data",
        "path_prefix": "projects/machinum-sft/material",
    },
}


# --- fixtures and helpers -------------------------------------------------


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    """Write a material YAML derived from BASE_CONFIG, with a `section.key` override syntax."""
    data = copy.deepcopy(BASE_CONFIG)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition("__")
        if key:
            data[section][key] = value
        else:
            data[section] = value
    path = tmp_path / "material.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> material.MaterialCfg:
    """A config whose material_dir lives under tmp_path; the memory token comes from the env."""
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    monkeypatch.setenv("MSFT_MEMORY_TOKEN", "test-token")
    path = write_config(tmp_path)
    return material.load_material_config(path)


@pytest.fixture(autouse=True)
def offline_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may open a socket for Habr's comment JSON: the seam answers "nothing" by default.

    A test that wants a comment tree patches `_fetch_comments` itself; this default keeps every
    other test exactly as offline as it was before comments joined the fetch path.
    """
    monkeypatch.setattr(
        material, "_fetch_comments", lambda article, timeout: (None, "offline test")
    )


@pytest.fixture(autouse=True)
def offline_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may run a real URL probe: every `--dry-run` resolve is answered locally.

    A probe is a subprocess, and `link_target` now asks one for every link a page carries, so
    without this fixture the `add` and link tests would shell out to the real tunnelstack. The
    fake says "every shape is readable" — which, thanks to fail-open probing, leaves every
    existing classification exactly as it was. A test that wants a refusal cans
    `("extract", site, url, "--dry-run"): (2, "", "site config error: …")` in its `Bridge`,
    and `install()` still overrides this fixture per test.
    """

    def fake(argv: list[str], timeout: float, json_flag: bool = True) -> tuple[int, str, str]:
        if argv and argv[-1] == "--dry-run":
            return 0, json.dumps({"url": argv[2]}), ""
        return 1, "", "offline test: no canned response"

    monkeypatch.setattr(material, "run_webbridge", fake)


@pytest.fixture
def seeded(cfg: material.MaterialCfg) -> material.MaterialCfg:
    """The same config, with `msft material seed` already run once."""
    material.seed(cfg)
    return cfg


class Bridge:
    """Stand-in for `run_webbridge`: canned payloads, every call recorded."""

    def __init__(self, table: dict[tuple[str, ...], Any]) -> None:
        self.table = table
        self.calls: list[tuple[list[str], float]] = []

    def __call__(
        self, argv: list[str], timeout: float, json_flag: bool = True
    ) -> tuple[int, str, str]:
        self.calls.append((list(argv), timeout))
        value = self.table.get(tuple(argv))
        if value is None:
            return 1, "", f"no canned response for {' '.join(argv)}"
        if isinstance(value, tuple):
            return value
        return 0, json.dumps(value, ensure_ascii=False), ""

    @property
    def argv(self) -> list[list[str]]:
        return [call[0] for call in self.calls]


def install(monkeypatch: pytest.MonkeyPatch, table: dict[tuple[str, ...], Any]) -> Bridge:
    bridge = Bridge(table)
    monkeypatch.setattr(material, "run_webbridge", bridge)
    return bridge


def habr_item(url: str, title: str, *, summary: str, published: str, tags: list[str], author: str):
    """One Habr RSS entry in the shape the webbridge client emits."""
    return {
        "title": title,
        "guid": url,
        "link": f"{url}?utm_source=habrahabr&utm_medium=rss",
        "description": summary,
        "pubDate": published,
        "creator": author,
        "category": tags[0] if tags else None,
    }


def habr_search(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"site": "habr", "op": "search", "mode": "rss", "items": items, "count": len(items)}


def github_search(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"site": "github", "op": "search", "mode": "api", "items": items, "count": len(items)}


def hf_search(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"site": "hf", "op": "search", "mode": "api", "items": items, "count": len(items)}


STRONG = habr_item(
    "https://habr.com/ru/articles/900001/",
    "LoRA fine-tuning LLM на практике",
    summary="<p>Обучение LoRA адаптера: VRAM, датасет 12k примеров, QLoRA</p>",
    published="Mon, 09 Nov 2026 10:00:00 +0000",
    tags=["machine learning"],
    author="alice",
)
WEAK = habr_item(
    "https://habr.com/ru/articles/900002/",
    "LoRA adapter notes",
    summary="llm notes",
    published="Mon, 02 Jan 2017 10:00:00 +0000",
    tags=["ml"],
    author="bob",
)


# --- config ---------------------------------------------------------------


def test_shipped_config_loads_and_validates() -> None:
    cfg = material.load_material_config()
    assert cfg.product.name == "machinum-sft"
    assert "qlora" in cfg.product.keywords
    # The list carries both alphabets: the corpus is mostly Russian, and an English-only keyword
    # list scores an on-topic habr post at 0 (found on 2026-11-12, six good posts dropped).
    assert any(re.search(r"[а-яё]", word) for word in cfg.product.keywords), "no Russian stems"
    # Explicit reads come first — they are the ones a human asked for.
    assert [seed.kind for seed in cfg.seeds[:4]] == ["url"] * 4
    assert len(cfg.seeds) >= 10
    assert {seed.site for seed in cfg.seeds} <= set(material.SITES), "a seed names a known site"
    assert {"twitter", "grok"} <= {seed.site for seed in cfg.seeds}, "social sources are wired"
    assert cfg.limits.items_per_run == 12
    assert cfg.limits.refresh_days == 3, "the daily loop must re-ask its questions"
    assert cfg.limits.synthesize_queries is True
    assert cfg.limits.links_per_item == 4
    assert cfg.scoring.expand_min == 0.6
    assert cfg.scoring.site_prior["twitter"] == 0.02
    assert cfg.memory.path_prefix == "projects/machinum-sft/material"
    assert cfg.distill.agent == "ds-code"


def test_config_missing_section_is_a_config_error(tmp_path: Path) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    del data["scoring"]
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing section"):
        material.load_material_config(path)


def test_config_missing_key_is_a_config_error(tmp_path: Path) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    del data["limits"]["max_frontier"]
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="max_frontier"):
        material.load_material_config(path)


def test_config_unknown_key_and_bad_site_are_config_errors(tmp_path: Path) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    data["extra"] = 1
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown top-level"):
        material.load_material_config(path)

    data = copy.deepcopy(BASE_CONFIG)
    data["seeds"][0]["site"] = "reddit"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="site"):
        material.load_material_config(path)


def test_config_not_found_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        material.load_material_config(tmp_path / "absent.yaml")


# --- seed -----------------------------------------------------------------


def test_seed_is_idempotent(cfg: material.MaterialCfg) -> None:
    first = material.seed(cfg)
    assert first["added"] == 3
    assert first["frontier"] == 3

    second = material.seed(cfg)
    assert second["added"] == 0
    assert second["existing"] == 3
    assert second["frontier"] == 3


def test_seed_dry_run_adds_nothing(cfg: material.MaterialCfg) -> None:
    result = material.seed(cfg, dry_run=True)
    assert result["added"] == 3
    assert not cfg.db_path.exists()


# --- explicit urls --------------------------------------------------------

ARTICLE = "https://habr.com/ru/articles/1044422/"
CANONICAL = "https://habr.com/ru/articles/1044422"
NOISE = {
    "site": "habr",
    "url": CANONICAL,
    "title": "Как я чинил забор",
    "text": "доски, гвозди",
}


def test_site_for_url_maps_known_hosts_and_refuses_unknown() -> None:
    assert material.site_for_url(ARTICLE) == "habr"
    assert material.site_for_url("https://github.com/huggingface/peft") == "github"
    assert material.site_for_url("https://huggingface.co/org/tiny-lora") == "hf"
    assert material.site_for_url("https://www.habr.com/ru/articles/1/") == "habr"
    assert material.site_for_url("https://m.habr.com/ru/articles/1/") == "habr"
    with pytest.raises(ConfigError, match="no site recipe"):
        material.site_for_url("https://example.com/x")


def test_site_for_url_maps_hft_and_its_subdomains() -> None:
    assert material.site_for_url("https://hft.md/uncensored-ai/") == "hft"
    assert material.site_for_url("https://www.hft.md/uncensored-ai") == "hft"
    assert material.site_for_url("https://hft.md/uncensored-ai") == "hft"
    assert "hft" not in material.SITES, "url-only: hft.md is not a search-capable seed site"


def test_site_for_url_maps_x_and_twitter_to_twitter() -> None:
    for url in (
        "https://x.com/u/status/1",
        "https://twitter.com/u/status/1",
        "https://www.x.com/u/status/1",
    ):
        assert material.site_for_url(url) == "twitter"


def test_rescore_brings_back_what_an_older_brief_threw_away(cfg: material.MaterialCfg) -> None:
    """Re-targeting the brief must reach the posts yesterday's keywords discarded."""
    url = "https://habr.com/ru/articles/424242/"
    with material.State(cfg.db_path) as state:
        state.conn.execute(
            "INSERT INTO items (url, site, title, summary, score, status, first_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (url, "habr", "LoRA адаптеры для LLM на практике", "", 0.0, "scored_out",
             "2026-11-01T00:00:00Z"),
        )
        state.conn.commit()

    dry = material.rescore(cfg, dry_run=True)
    assert dry["scored_out"] == 1
    assert dry["flipped"] == 1
    assert "would keep" in material.format_rescore(dry)
    with material.State(cfg.db_path) as state:
        row = state.conn.execute("SELECT status, score FROM items WHERE url = ?", (url,)).fetchone()
    assert row["status"] == "scored_out", "a dry run must not touch the row"

    real = material.rescore(cfg)
    assert real["flipped"] == 1 and real["still_out"] == 0
    with material.State(cfg.db_path) as state:
        row = state.conn.execute("SELECT status, score FROM items WHERE url = ?", (url,)).fetchone()
    assert row["status"] == "new"
    assert row["score"] >= cfg.scoring.keep_min
    assert material.rescore(cfg)["scored_out"] == 0, "a flipped row is no longer scored_out"


def test_rescore_leaves_a_genuinely_off_topic_row_out(cfg: material.MaterialCfg) -> None:
    url = "https://habr.com/ru/articles/434343/"
    with material.State(cfg.db_path) as state:
        state.conn.execute(
            "INSERT INTO items (url, site, title, summary, score, status, first_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (url, "habr", "Как я готовил паэлью", "кулинария", 0.0, "scored_out",
             "2026-11-01T00:00:00Z"),
        )
        state.conn.commit()
    result = material.rescore(cfg)
    assert result["flipped"] == 0 and result["still_out"] == 1
    with material.State(cfg.db_path) as state:
        row = state.conn.execute("SELECT status FROM items WHERE url = ?", (url,)).fetchone()
    assert row["status"] == "scored_out"


def test_run_fetches_a_kept_but_unread_item_with_no_query_at_all(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `rescore` flip must actually be read; the query loop alone would never fetch it."""
    url = "https://habr.com/ru/articles/454545/"
    with material.State(cfg.db_path) as state:
        state.conn.execute(
            "INSERT INTO items (url, site, title, summary, score, status, first_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (url, "habr", "Структурированный вывод LLM", "", 0.5, "new", "2026-11-01T00:00:00Z"),
        )
        state.conn.commit()

    seen: list[str] = []

    def fake_fetch(_cfg, _entry, target, _day):  # noqa: ANN001 - test seam
        seen.append(target)
        return material.Fetched(body="длинный текст про LLM", raw=None, article="длинный текст")

    monkeypatch.setattr(material, "_fetch_body", fake_fetch)
    result = material.run(cfg, day=DAY, limit=0)
    assert seen == [url], "the pending row must be fetched even though no query ran"
    assert result["fetched"] == 1
    with material.State(cfg.db_path) as state:
        row = state.conn.execute(
            "SELECT status, raw_path FROM items WHERE url = ?", (url,)
        ).fetchone()
    assert row["status"] == "fetched"
    assert row["raw_path"]


def test_extract_error_rejects_a_wrong_or_empty_page() -> None:
    """A stale tab returns another page; that is an error, not a 3-byte article."""
    habr = "https://habr.com/ru/articles/1057472/"
    ok = {"url": habr, "text": "настоящая статья про LLM", "found": True}
    assert material._extract_error(ok, habr) is None
    # m.habr.com is the same page; www is dropped by the comparison.
    assert material._extract_error({**ok, "url": "https://m.habr.com/x"}, habr) is None
    assert "found nothing" in (material._extract_error({**ok, "found": False}, habr) or "")
    google = {**ok, "url": "https://www.google.com/search?udm=50&q=x"}
    assert "google.com" in (material._extract_error(google, habr) or "")
    # the client prints `""` for an empty extraction, and that used to pass as a body
    assert "empty text" in (material._extract_error({**ok, "text": '""'}, habr) or "")
    assert "empty text" in (material._extract_error({**ok, "text": "   \n "}, habr) or "")


def test_same_host_aliases_x_and_twitter() -> None:
    assert material._same_host("x.com", "twitter.com")
    assert material._same_host("www.x.com", "twitter.com")
    assert material._same_host("m.habr.com", "habr.com")
    assert not material._same_host("x.com", "example.com")


def test_fetch_body_marks_a_stale_tab_as_an_error(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the seam: the payload of the wrong page never becomes a body."""
    url = "https://habr.com/ru/articles/1057472/"
    payload = {
        "site": "habr",
        "found": False,
        "op": "extract",
        "url": "https://www.google.com/search?udm=50&q=go",
        "text": '""',
    }
    monkeypatch.setattr(
        material, "run_webbridge", lambda argv, timeout: (0, json.dumps(payload), "")
    )
    result = material._fetch_body(cfg, {"site": "habr", "url": url}, url, DAY)
    assert result.error and "google.com" in result.error
    assert result.body == ""


def test_add_urls_canonicalises_dedupes_and_is_idempotent(cfg: material.MaterialCfg) -> None:
    first = material.add_urls(cfg, [ARTICLE, ARTICLE, ARTICLE + "?utm_source=x"])
    assert first["urls"] == 3
    assert first["added"] == 1, "three spellings of one URL are one candidate"
    assert first["would_add"] == [f"url/habr: {CANONICAL}"]
    assert first["frontier"] == 1

    with material.State(cfg.db_path) as state:
        row = state.conn.execute("SELECT * FROM candidates").fetchone()
    assert row is not None
    assert row["kind"] == "url"
    assert row["value"] == CANONICAL
    assert row["site"] == "habr"
    assert row["priority"] == 1.0
    assert row["reason"] == "manual"

    second = material.add_urls(cfg, [ARTICLE])
    assert second["added"] == 0
    assert second["existing"] == 1
    assert second["known"] == [f"url/habr: {CANONICAL}"]
    assert second["frontier"] == 1


def test_add_urls_dry_run_writes_nothing(cfg: material.MaterialCfg) -> None:
    result = material.add_urls(cfg, [ARTICLE], dry_run=True)
    assert not cfg.db_path.exists()
    assert result["dry_run"] is True
    assert result["added"] == 1
    assert result["would_add"] == [f"url/habr: {CANONICAL}"]
    assert result["frontier"] == 1


def test_add_urls_unknown_host_is_a_config_error(cfg: material.MaterialCfg) -> None:
    with pytest.raises(ConfigError, match="no site recipe"):
        material.add_urls(cfg, ["https://example.com/x"])
    assert not cfg.db_path.exists(), "a rejected URL must not create state"


HFT_PAGE = "https://hft.md/uncensored-ai"


def test_hft_links_are_now_fetchable_not_hints() -> None:
    """`URL_SITES` is what `link_target` consults, so the recipe also un-hints future links."""
    assert material.link_target(HFT_PAGE, {}) == ("url", "hft")


def test_add_urls_promotes_a_hint_instead_of_duplicating_it(cfg: material.MaterialCfg) -> None:
    """A link first seen as a dead hint must become the operator's fetch row, not a twin."""
    with material.State(cfg.db_path) as state:
        assert state.add_candidate(
            "hint", HFT_PAGE, "hft.md",
            parent="https://x.com/u/status/1", reason="hint:hft.md", status="hint",
        )

    first = material.add_urls(cfg, [HFT_PAGE])
    assert first["added"] == 1, "the hint became fetchable, so it counts as new work"
    assert first["existing"] == 0
    assert first["frontier"] == 1

    with material.State(cfg.db_path) as state:
        rows = state.conn.execute(
            "SELECT * FROM candidates WHERE value = ?", (HFT_PAGE,)
        ).fetchall()
    assert len(rows) == 1, "the hint row is converted in place, not shadowed"
    row = rows[0]
    assert row["kind"] == "url"
    assert row["site"] == "hft"
    assert row["priority"] == 1.0
    assert row["reason"] == "manual"
    assert row["status"] == "new"
    assert row["attempts"] == 0
    assert row["error"] is None
    assert row["parent"] == "https://x.com/u/status/1", "provenance of the lead survives"

    second = material.add_urls(cfg, [HFT_PAGE])
    assert second["added"] == 0
    assert second["existing"] == 1
    assert second["known"] == [f"url/hft: {HFT_PAGE}"]
    with material.State(cfg.db_path) as state:
        count = state.conn.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE value = ?", (HFT_PAGE,)
        ).fetchone()["n"]
    assert count == 1


def test_candidate_argv_maps_url_to_extract() -> None:
    argv = material.candidate_argv({"kind": "url", "site": "habr", "value": CANONICAL})
    assert argv == ["extract", "habr", CANONICAL]


# --- a vendor docs site (unsloth) ----------------------------------------

UNSLOTH_PAGE = "https://unsloth.ai/docs/models/qwen3.8"
UNSLOTH_MD = UNSLOTH_PAGE + ".md"
UNSLOTH_BODY = """\
# Qwen3.8 - How to Run Locally

Fine-tune Qwen3.8 with LoRA on a single 24 GB GPU. The adapter fits when the base model is
loaded in 4-bit and the sequence length stays under 4096.

## Install

```bash
pip install unsloth
```

| Quant | VRAM |
| --- | --- |
| Q4_K_M | 17 GB |
| Q8_0 | 30 GB |
"""
# The shape tunnelstack's `extract` returns for `mode: text`: the markdown twin, the H1 as
# `title`, and the whole page in `text` (no `found` key, so it must not trip that check).
UNSLOTH_PAYLOAD = {
    "site": "unsloth",
    "op": "extract",
    "mode": "text",
    "url": UNSLOTH_MD,
    "title": "Qwen3.8 - How to Run Locally",
    "text": UNSLOTH_BODY,
    "chars": len(UNSLOTH_BODY),
}


def test_site_for_url_maps_unsloth_and_its_markdown_twin() -> None:
    assert material.site_for_url(UNSLOTH_PAGE) == "unsloth"
    assert material.site_for_url(UNSLOTH_MD) == "unsloth", "the `.md` twin is the same site"
    assert material.site_for_url("https://www.unsloth.ai/docs") == "unsloth"


def test_unsloth_is_a_known_seed_site(tmp_path: Path) -> None:
    path = write_config(
        tmp_path, seeds=[{"kind": "url", "site": "unsloth", "value": UNSLOTH_PAGE}]
    )
    cfg = material.load_material_config(path)
    assert cfg.seeds[0].site == "unsloth"


def test_items_from_payload_unsloth_text_mode_is_the_markdown_body(
    cfg: material.MaterialCfg,
) -> None:
    cand = {"kind": "url", "site": "unsloth", "value": UNSLOTH_PAGE}
    items = material.items_from_payload(UNSLOTH_PAYLOAD, cand, cfg)

    assert len(items) == 1
    item = items[0]
    assert item["title"] == "Qwen3.8 - How to Run Locally", "the H1 is the title"
    assert item["url"] == UNSLOTH_MD, "the item is the page the recipe actually read"
    assert item["summary"] == material.strip_html(UNSLOTH_BODY)
    assert "mode: text" not in item["summary"], "not the rendered key: value flattening"


def test_extract_error_accepts_the_markdown_twin() -> None:
    assert material._extract_error(UNSLOTH_PAYLOAD, UNSLOTH_PAGE) is None
    empty = {**UNSLOTH_PAYLOAD, "text": ""}
    assert "empty text" in (material._extract_error(empty, UNSLOTH_PAGE) or "")


def test_fetch_body_keeps_the_whole_markdown_twin(cfg: material.MaterialCfg) -> None:
    """`summary` is a 1200-char prefix; the raw file must be the whole `.md` twin."""
    body = UNSLOTH_BODY + "\n\n" + ("LoRA adapter notes for Qwen3.8. " * 200)
    assert len(body) > 3 * 1200
    payload = {**UNSLOTH_PAYLOAD, "text": body, "chars": len(body)}
    cand = {"kind": "url", "site": "unsloth", "value": UNSLOTH_PAGE}
    item = material.items_from_payload(payload, cand, cfg)[0]

    fetched = material._fetch_body(cfg, item, UNSLOTH_PAGE, DAY)
    assert fetched.error is None
    assert fetched.body == body
    assert fetched.article == body


def test_url_seed_is_ranked_before_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    path = write_config(
        tmp_path,
        seeds=[
            {"kind": "query", "site": "habr", "value": "LoRA fine-tuning LLM"},
            {"kind": "url", "site": "habr", "value": ARTICLE},
        ],
    )
    cfg = material.load_material_config(path)
    material.seed(cfg)
    with material.State(cfg.db_path) as state:
        first = state.candidates(limit=10)[0]
    assert first["kind"] == "url"
    assert first["priority"] == 1.0


def test_run_fetches_an_explicit_url_below_keep_min(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = install(monkeypatch, {("extract", "habr", CANONICAL): NOISE})
    material.add_urls(cfg, [CANONICAL])
    result = material.run(cfg, limit=1, day=DAY, sites=["habr"])

    assert ["extract", "habr", CANONICAL] in bridge.argv
    assert bridge.argv[0] == ["extract", "habr", CANONICAL, "--dry-run"], (
        "add probes the shape before queueing, then the run does the real extract"
    )
    assert result["items"] == 1
    assert result["kept"] == 1
    assert result["fetched"] == 1
    assert result["ok"] is True
    with material.State(cfg.db_path) as state:
        row = state.item(CANONICAL)
    assert row is not None
    assert row["status"] == "fetched"
    assert row["score"] == 0.0, "the score is recorded, it just does not veto"
    raws = list((cfg.raw_dir / DAY).glob("*.txt"))
    assert len(raws) == 1


def test_run_still_drops_an_unscored_search_item(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    noise = habr_item(
        NOISE["url"] + "/",
        NOISE["title"],
        summary="<p>доски, гвозди</p>",
        published="Sun, 08 Nov 2026 10:00:00 +0000",
        tags=["diy"],
        author="carol",
    )
    install(monkeypatch, {("search", "habr", "LoRA fine-tuning LLM"): habr_search([noise])})
    result = material.run(seeded, limit=1, day=DAY, sites=["habr"])
    assert result["items"] == 1
    assert result["kept"] == 0
    assert result["fetched"] == 0
    with material.State(seeded.db_path) as state:
        row = state.item(CANONICAL)
    assert row is not None
    assert row["status"] == "scored_out", "the keep_min bypass is url-only"
    assert row["score"] < seeded.scoring.keep_min


def test_cli_material_add_json(
    cfg: material.MaterialCfg, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["material", "add", ARTICLE, "--json", "--config", str(cfg.path)]
    assert main(argv) == 0
    payload = one_object(capsys.readouterr().out)
    assert payload["added"] == 1
    assert payload["would_add"] == [f"url/habr: {CANONICAL}"]

    assert main(argv) == 0
    again = one_object(capsys.readouterr().out)
    assert again["added"] == 0
    assert again["existing"] == 1


# --- canonicalisation, hashing, normalisation -----------------------------


def test_canonical_url_drops_utm_fragment_and_trailing_slash() -> None:
    raw = "HTTPS://Habr.com/ru/articles/900001/?utm_source=habrahabr&utm_medium=rss#comments"
    assert material.canonical_url(raw) == "https://habr.com/ru/articles/900001"
    assert material.canonical_url("https://Example.COM/Path/") == "https://example.com/Path"
    assert material.canonical_url("https://habr.com/ru/articles/900001") == (
        material.canonical_url("https://habr.com/ru/articles/900001/?utm_campaign=x")
    )


def test_canonical_url_keeps_meaningful_query() -> None:
    url = "https://hf.co/api/models?search=lora&limit=10"
    assert material.canonical_url(url) == url


def test_content_sha_is_stable_across_formatting() -> None:
    first = material.content_sha("A title", "<p>Body</p>\n\n\ntext")
    second = material.content_sha("A title", "<div>Body</div>\n\ntext")
    assert first == second
    assert material.content_sha("Other", "Body text") != first


def test_normalize_text_strips_blocks_tags_and_blank_runs() -> None:
    raw = (
        "<html><head><style>p{color:red}</style><script>var x=1;</script></head>"
        "<body><nav>menu</nav><h1>Title</h1><p>First   line   </p><p>Second</p>"
        "<aside>ad</aside><footer>foot</footer></body></html>"
    )
    out = material.normalize_text(raw)
    assert "color:red" not in out
    assert "var x" not in out
    assert "menu" not in out
    assert "ad" not in out
    assert "foot" not in out
    assert "Title" in out and "First line" in out
    assert "\n\n\n" not in out
    assert "First   line" not in out


def test_normalize_text_caps_at_120k() -> None:
    raw = "<p>" + "x" * 150_000 + "</p><script>var x=1;</script>"
    out = material.normalize_text(raw)
    assert len(out) == material.TEXT_LIMIT == 120_000
    assert set(out) == {"x"}, "the cap still runs after tags and script blocks are stripped"
    assert "var x" not in out


def test_slug_is_60_chars_plus_sha1_suffix() -> None:
    slug = material.slugify("https://habr.com/ru/articles/900001/")
    base, _, digest = slug.rpartition("-")
    assert len(base) <= 60
    assert len(digest) == 8
    assert slug == material.slugify("https://habr.com/ru/articles/900001/")
    assert slug != material.slugify("https://habr.com/ru/articles/900002/")


# --- scoring --------------------------------------------------------------


def item(**overrides: Any) -> dict[str, Any]:
    base = {
        "url": "https://example.com/x",
        "site": "habr",
        "title": "",
        "summary": "",
        "published": None,
        "tags": [],
        "stars": 0,
        "downloads": 0,
    }
    base.update(overrides)
    return base


def test_score_drops_radio_lora(cfg: material.MaterialCfg) -> None:
    radio = item(
        title="LoRaWAN gateway for the farm",
        summary="A radio sensor network over 868 MHz; a dataset of readings",
    )
    assert material.score_item(radio, cfg, material.date_cls(2026, 11, 10)) == 0.0


def test_score_keeps_llm_lora(cfg: material.MaterialCfg) -> None:
    adapter = item(
        title="QLoRA fine-tuning a 7B LLM",
        summary="an adapter on one 16 GB card",
        published="2026-11-01",
    )
    score = material.score_item(adapter, cfg, material.date_cls(2026, 11, 10))
    assert score >= cfg.scoring.keep_min
    assert score > 0.6


def test_score_negative_keyword_is_zero(cfg: material.MaterialCfg) -> None:
    spam = item(title="LoRA webinar: hiring now", summary="llm adapter")
    assert material.score_item(spam, cfg) == 0.0


def test_score_recency_bonus(cfg: material.MaterialCfg) -> None:
    fresh = item(title="LoRA adapter", summary="llm", published="2026-11-01")
    old = item(title="LoRA adapter", summary="llm", published="2016-11-01")
    today = material.date_cls(2026, 11, 10)
    delta = material.score_item(fresh, cfg, today) - material.score_item(old, cfg, today)
    assert round(delta, 3) == 0.2


def test_score_stars_bonus(cfg: material.MaterialCfg) -> None:
    popular = item(site="github", title="LoRA", summary="llm adapter", stars=5_000)
    quiet = item(site="github", title="LoRA", summary="llm adapter", stars=3)
    today = material.date_cls(2026, 11, 10)
    delta = material.score_item(popular, cfg, today) - material.score_item(quiet, cfg, today)
    assert round(delta, 3) == 0.15


def test_score_is_clamped_and_rounded(cfg: material.MaterialCfg) -> None:
    loud = item(
        title="LoRA QLoRA PEFT SFT dataset VRAM",
        summary="llm adapter обучение",
        published="2026-11-09",
    )
    score = material.score_item(loud, cfg, material.date_cls(2026, 11, 10))
    assert score == 1.0
    assert score == round(score, 3)


# --- run ------------------------------------------------------------------


def test_run_collects_fetches_and_dedupes(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    twin = dict(STRONG)
    twin["guid"] = "https://habr.com/ru/articles/900003/"
    twin["link"] = "https://habr.com/ru/articles/900003/"
    noise = habr_item(
        "https://habr.com/ru/articles/900004/",
        "Как я чинил забор",
        summary="<p>доски, гвозди</p>",
        published="Sun, 08 Nov 2026 10:00:00 +0000",
        tags=["diy"],
        author="carol",
    )
    bridge = install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG, twin, noise]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "<p>Full article body about LoRA, QLoRA and a 12k dataset.</p>",
            },
        },
    )
    result = material.run(seeded, limit=1, day=DAY, sites=["habr"])

    assert result["queries"] == 1
    assert result["items"] == 3
    assert result["kept"] == 1
    assert result["duplicates"] == 1
    assert result["fetched"] == 1
    assert result["errors"] == 0
    assert result["ok"] is True

    raw = sorted(path.name for path in (seeded.raw_dir / DAY).iterdir())
    assert len(raw) == 2
    assert raw[0].endswith(".json") and raw[1].endswith(".txt")
    text = (seeded.raw_dir / DAY / raw[1]).read_text(encoding="utf-8")
    assert "Full article body about LoRA" in text
    assert ["extract", "habr", "https://habr.com/ru/articles/900001"] in bridge.argv


def test_run_marks_a_failed_cli_as_candidate_error_and_continues(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        {
            ("search", "github", "lora fine-tuning"): (2, "", "site config error: boom"),
            ("search", "hf", "lora"): hf_search(
                [{"id": "org/tiny-lora", "downloads": 25_000, "likes": 40,
                  "pipeline": "text-generation", "library": "peft"}]
            ),
        },
    )
    result = material.run(seeded, day=DAY, sites=["github", "hf"])
    assert result["queries"] == 2
    assert result["errors"] == 1
    assert result["ok"] is False
    assert result["items"] == 1
    assert result["fetched"] == 1

    with material.State(seeded.db_path) as state:
        rows = {row["site"]: dict(row) for row in state.conn.execute("SELECT * FROM candidates")}
    assert rows["github"]["status"] == "error"
    assert "boom" in (rows["github"]["error"] or "")
    assert rows["hf"]["status"] == "done"


def test_run_treats_zero_results_as_done_not_an_error(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search that matches nothing exits 1 in the tunnelstack client but still prints JSON.

    Observed live: the `author/DaytonCavalet` expansion node ("articles by an author of a
    high-scoring post") answers with zero RSS items, the client exits 1, and an earlier
    runner treated that as a failure — which aborted the collect step and, through the
    daily script, the whole cron run. A frontier node that has been explored and came back
    empty is *done*, not broken.
    """
    install(
        monkeypatch,
        {
            ("search", "hf", "lora"): (
                1,  # exit 1: "no text extracted" — the client's empty-result contract
                json.dumps({"site": "hf", "op": "search", "count": 0, "items": []}),
                "hf: (no text extracted)",
            ),
        },
    )
    result = material.run(seeded, day=DAY, sites=["hf"])
    assert result["ok"] is True
    assert result["errors"] == 0
    assert result["queries"] == 1
    assert result["items"] == 0

    with material.State(seeded.db_path) as state:
        rows = {row["site"]: dict(row) for row in state.conn.execute("SELECT * FROM candidates")}
    assert rows["hf"]["status"] == "done"
    assert rows["hf"]["error"] is None


def test_run_retries_an_error_candidate_until_the_budget_runs_out(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient failure must not delete a frontier node for ever.

    The live case: the first cron fire failed on a site that was mid-deploy. Without a retry
    budget the `error` row is permanent, so the discovery path that produced it dies with one
    bad afternoon.
    """
    install(
        monkeypatch,
        {
            ("search", "hf", "lora"): (2, "", "HTTP 502 from the site"),
        },
    )
    first = material.run(seeded, day=DAY, sites=["hf"])
    assert first["queries"] == 1
    assert first["errors"] == 1

    second = material.run(seeded, day=DAY, sites=["hf"])
    assert second["queries"] == 1, "an errored candidate is retried on the next run"
    assert second["errors"] == 1

    third = material.run(seeded, day=DAY, sites=["hf"])
    assert third["queries"] == 0, "retry_errors=2 is the budget, not an infinite loop"
    assert third["ok"] is True

    with material.State(seeded.db_path) as state:
        rows = [dict(r) for r in state.conn.execute(
            "SELECT status, attempts FROM candidates WHERE site = 'hf'"
        )]
        assert all(row["status"] == "error" and row["attempts"] == 2 for row in rows)
        # The hf node is no longer eligible; the habr/github seeds were never touched
        # (the run was scoped with sites=["hf"]), so they still count as frontier.
        assert state.frontier() == 2
        assert state.conn.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE site = 'hf' AND status = 'new'"
        ).fetchone()["n"] == 0


def test_run_skips_a_url_two_candidates_both_return(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = habr_search([WEAK])
    with material.State(seeded.db_path) as state:
        state.add_candidate("query", "QLoRA обучение больших языковых моделей", "habr")
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): payload,
            ("search", "habr", "QLoRA обучение больших языковых моделей"): payload,
        },
    )
    result = material.run(seeded, day=DAY, sites=["habr"])
    assert result["queries"] == 2
    assert result["items"] == 1
    assert result["seen_before"] == 1
    with material.State(seeded.db_path) as state:
        assert state.items_total() == 1


def test_run_enqueues_children_once_and_respects_no_expand(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "body about LoRA and QLoRA",
            },
        },
    )
    first = material.run(seeded, day=DAY, sites=["habr"])
    assert first["children"] >= 1
    with material.State(seeded.db_path) as state:
        kinds = {row["kind"] for row in state.conn.execute("SELECT kind FROM candidates")}
        assert "tag" in kinds and "author" in kinds
        before = state.candidates_count("new")

    # The query itself is done, so the second run only executes the children — and the same
    # tag/author must not be added twice.
    second = material.run(seeded, day=DAY, sites=["habr"], limit=0)
    assert second["children"] == 0
    with material.State(seeded.db_path) as state:
        assert state.candidates_count("new") == before

    third = material.run(seeded, day=DAY, sites=["habr"], no_expand=True)
    assert third["children"] == 0


def test_run_below_expand_threshold_enqueues_nothing(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([WEAK]),
            ("extract", "habr", "https://habr.com/ru/articles/900002"): {
                "site": "habr",
                "text": "old llm notes",
            },
        },
    )
    result = material.run(seeded, day=DAY, sites=["habr"])
    assert result["kept"] == 1
    assert result["kept_by_site"] == {"habr": 1}
    with material.State(seeded.db_path) as state:
        children = state.conn.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE parent IS NOT NULL"
        ).fetchone()["n"]
    assert children == 0


def test_run_respects_frontier_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    path = write_config(
        tmp_path,
        limits__max_frontier=1,
        limits__children_per_item=5,
        seeds=[{"kind": "query", "site": "habr", "value": "LoRA fine-tuning LLM"}],
    )
    cfg = material.load_material_config(path)
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "body about LoRA",
            },
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"])
    assert result["children"] == 1
    assert result["frontier_full"] == 1


def test_run_children_per_item_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    path = write_config(tmp_path, limits__children_per_item=1)
    cfg = material.load_material_config(path)
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "body about LoRA",
            },
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"])
    assert result["children"] == 1


def test_run_dry_run_calls_nothing(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge = install(monkeypatch, {})
    material.seed(cfg)
    result = material.run(cfg, day=DAY, dry_run=True)
    assert result["dry_run"] is True
    assert result["queries_planned"] == 3
    assert bridge.calls == []
    with material.State(cfg.db_path) as state:
        assert state.items_total() == 0


def test_run_payload_shape_fetches_github_and_hf(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        {
            ("search", "github", "lora fine-tuning"): github_search(
                [
                    {
                        "repo": "huggingface/peft",
                        "stars": 15_000,
                        "language": "Python",
                        "forks": 1_400,
                        "open_issues": 60,
                        "pushed": "2026-11-09T10:00:00Z",
                        "description": "State-of-the-art parameter-efficient fine-tuning",
                        "url": "https://github.com/huggingface/peft",
                    }
                ]
            ),
            ("search", "hf", "lora"): hf_search(
                [
                    {
                        "id": "org/tiny-lora",
                        "downloads": 25_000,
                        "likes": 40,
                        "pipeline": "text-generation",
                        "library": "peft",
                    }
                ]
            ),
        },
    )
    result = material.run(seeded, day=DAY, sites=["github", "hf"])
    assert result["items"] == 2
    assert result["fetched"] == 2
    assert set(result["kept_by_site"]) == {"github", "hf"}
    files = sorted(path.name for path in (seeded.raw_dir / DAY).glob("*.txt"))
    assert len(files) == 2
    body = (seeded.raw_dir / DAY / files[0]).read_text(encoding="utf-8")
    assert "repo:" in body or "id:" in body


def test_run_empty_body_is_an_error(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "   ",
            },
        },
    )
    result = material.run(seeded, day=DAY, sites=["habr"])
    assert result["errors"] == 1
    assert result["fetched"] == 0
    assert result["ok"] is False


# --- pack and distill -----------------------------------------------------


def prepare_fetched(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> material.MaterialCfg:
    """One strong habr item, fetched, so distill has something to pack."""
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Body " + "z" * 9_000,
            },
        },
    )
    material.run(cfg, day=DAY, sites=["habr"], no_expand=True)
    return cfg


def fake_at(
    tmp_path: Path,
    *,
    status: str = "ok",
    exit_code: int = 0,
    write_notes: bool = True,
    record_size: Path | None = None,
) -> Path:
    """A stand-in for `at`: reads the pack, writes one note per slug, prints a JSON summary."""
    script = tmp_path / "fake-at"
    record = (
        f"pathlib.Path({str(record_size)!r}).open('a')"
        ".write(str(len(pack.read_bytes())) + '\\n')\n"
        if record_size is not None
        else ""
    )
    note_body = (
        """
slugs = [line.split(": ", 1)[1] for line in pack.read_text().splitlines()
         if line.startswith("- slug: ")]
for slug in slugs:
    (notes / f"{slug}.md").write_text(
        "---\\n"
        "title: A packed item\\n"
        "site: habr\\n"
        "url: https://habr.com/ru/articles/900001/\\n"
        "tags: [lora, qlora]\\n"
        "published: 2026-11-09\\n"
        "score: 1.0\\n"
        "experiment: Try r=32 with completion-only loss on 200 pairs\\n"
        "---\\n\\n# A packed item\\n\\n## What it is\\n\\nA note.\\n"
    )
"""
        if write_notes
        else "\nslugs = []\n"
    )
    script.write_text(
        f"""#!/usr/bin/env python3
import json, pathlib, sys
pack = pathlib.Path(sys.argv[sys.argv.index("-f") + 1])
{record}notes = pathlib.Path({str(tmp_path / "material" / "notes")!r}) / pack.stem
notes.mkdir(parents=True, exist_ok=True)
{note_body}
print("Now using node v22.20.0 (npm v10.9.3)")
print(json.dumps({{"status": {status!r}, "files": len(slugs)}}))
sys.exit({exit_code})
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def seed_fetched_day(
    cfg: material.MaterialCfg, count: int, *, body_chars: int = 3_000
) -> list[str]:
    """`count` fetched items on DAY, each with its own raw body file, all at score 1.0.

    Returned in `items_for_date` order (`score DESC, url ASC`), which is the order `_pack_slice`
    slices in.
    """
    urls: list[str] = []
    with material.State(cfg.db_path) as state:
        for index in range(1, count + 1):
            url = f"https://habr.com/ru/articles/9100{index:02d}/"
            raw = cfg.material_dir / "raw" / DAY / f"habr-{index:02d}.txt"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("Body " + "z" * body_chars, encoding="utf-8")
            row = material._new_item_row(
                {"url": url, "site": "habr", "title": f"Item {index}"}, 1.0, "fetched"
            )
            row["raw_path"] = cfg.display(raw)
            state.add_item(row)
            urls.append(url)
    return urls


def pack_bodies(text: str) -> list[str]:
    """The `text` fenced body of each pack section, in section order."""
    return [chunk.split("\n```", 1)[0] for chunk in text.split("```text\n")[1:]]


def test_pack_contains_brief_rules_schema_and_carries_a_body_past_8k(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_fetched(cfg, monkeypatch)
    result = material.distill(cfg, day=DAY, dry_run=True)
    assert result["dry_run"] is True
    assert result["items"] == 1
    assert not (cfg.pack_dir / f"{DAY}.md").exists()

    with material.State(cfg.db_path) as state:
        items = state.items_for_date(DAY)
    text = material._pack_text(cfg, DAY, items)
    assert "A small local model that translates English prose" in text
    assert "docs/material/notes/" in text
    assert "## What it is" in text and "## Experiment" in text
    assert "https://habr.com/ru/articles/900001" in text
    # One item alone has the whole light-day budget: the 9 005-char body arrives whole, where the
    # old fixed 8 000-char slice would have cut 1 005 chars off it.
    assert material._pack_body_budget(items) == material.PACK_TEXT_MAX
    assert pack_bodies(text)[0].count("z") == 9_000
    assert "обрезано" not in text, "nothing was cut, so no marker"


def test_distill_dry_run_never_runs_at(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_fetched(cfg, monkeypatch)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("distill --dry-run must not start the agent")

    monkeypatch.setattr(material.subprocess, "run", explode)
    result = material.distill(cfg, day=DAY, dry_run=True)
    assert result["items"] == 1
    assert result["command"][0].endswith("at")
    assert "--timeout" in result["command"]


def test_distill_success_writes_notes_experiments_and_status(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path)))

    result = material.distill(cfg, day=DAY)
    assert result["ok"] is True
    assert result["packed"] == 1
    assert result["distilled"] == 1
    assert result["missing_notes"] == 0
    assert result["experiments"] == 1
    assert (cfg.pack_dir / f"{DAY}.md").is_file()
    assert "completion-only loss" in (cfg.material_dir / "experiments.md").read_text()

    with material.State(cfg.db_path) as state:
        items = state.items_for_date(DAY)
    assert items[0]["status"] == "distilled"
    assert items[0]["digest_path"].endswith(".md")

    again = material.distill(cfg, day=DAY)
    assert again["items"] == 0
    assert "completion-only loss" in (cfg.material_dir / "experiments.md").read_text()
    assert (cfg.material_dir / "experiments.md").read_text().count("completion-only") == 1


def test_distill_failure_keeps_items_packed(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv(
        "MSFT_AT_BIN", str(fake_at(tmp_path, status="failed", exit_code=1, write_notes=False))
    )

    result = material.distill(cfg, day=DAY)
    assert result["ok"] is False
    assert result["distilled"] == 0
    with material.State(cfg.db_path) as state:
        items = state.items_for_date(DAY)
    assert items[0]["status"] == "packed"

    # A retry packs the still-packed item again rather than losing it.
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path)))
    retry = material.distill(cfg, day=DAY)
    assert retry["items"] == 1
    assert retry["distilled"] == 1


def test_distill_books_the_notes_a_timed_out_agent_did_write(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`at` killed at its timeout after writing every note is a done digest, not a failure."""
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv(
        "MSFT_AT_BIN", str(fake_at(tmp_path, status="failed", exit_code=1, write_notes=True))
    )

    result = material.distill(cfg, day=DAY)
    assert result["agent_failed"] is True
    assert result["ok"] is True, "every expected note is on disk"
    assert result["distilled"] == 1
    assert result["missing_notes"] == 0
    assert result["experiments"] == 1
    with material.State(cfg.db_path) as state:
        items = state.items_for_date(DAY)
    assert items[0]["status"] == "distilled"

    # Nothing is left to do, so tomorrow's run does not re-pack and re-read the same page.
    assert material.distill(cfg, day=DAY)["items"] == 0


def test_distill_packs_a_bounded_slice_and_defers_the_rest(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One `at` run carries one prompt, so a backlog day drains one pack per run."""
    urls = seed_fetched_day(cfg, material.PACK_MAX_ITEMS + 3, body_chars=12_000)
    size_file = tmp_path / "pack-size"
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path, record_size=size_file)))

    result = material.distill(cfg, day=DAY)

    assert result["ok"] is True
    assert result["items"] == material.PACK_MAX_ITEMS, "one `at` run gets one prompt's worth"
    assert result["packed"] == material.PACK_MAX_ITEMS
    assert result["deferred"] == 3
    assert result["missing_notes"] == 0
    sizes = size_file.read_text().split()
    assert len(sizes) == 1, "exactly one `at` call for the day's first pack"
    assert int(sizes[0]) <= material.PACK_MAX_BYTES

    # The budget is shared, not raised: 13 items settle on the 8 000-char floor, so a busy day
    # still packs the same 10 items it did before the budget was shared.
    bodies = pack_bodies((cfg.pack_dir / f"{DAY}.md").read_text(encoding="utf-8"))
    assert len(bodies) == material.PACK_MAX_ITEMS
    assert all(body.count("z") <= material.PACK_TEXT_LIMIT for body in bodies), (
        "the carried article text stays at the floor; the truncation marker rides on top"
    )

    with material.State(cfg.db_path) as state:
        statuses = {str(row["url"]): row["status"] for row in state.items_for_date(DAY)}
    assert [statuses[url] for url in urls[: material.PACK_MAX_ITEMS]] == (
        ["distilled"] * material.PACK_MAX_ITEMS
    )
    assert [statuses[url] for url in urls[material.PACK_MAX_ITEMS :]] == ["fetched"] * 3, (
        "the deferred rows stay fetched, not packed; the next run packs them"
    )

    rest = material.distill(cfg, day=DAY)
    assert rest["items"] == 3
    assert rest["deferred"] == 0
    assert len(size_file.read_text().split()) == 2, "the remainder is one more `at` call"
    with material.State(cfg.db_path) as state:
        statuses = {str(row["url"]): row["status"] for row in state.items_for_date(DAY)}
    assert set(statuses.values()) == {"distilled"}, "a backlog drains, it does not deadlock"


def test_pack_gives_a_single_item_the_light_day_budget(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One item on a light day may fill the pack: no other item is there to be starved."""
    seed_fetched_day(cfg, 1, body_chars=100_000)
    size_file = tmp_path / "pack-size"
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path, record_size=size_file)))

    result = material.distill(cfg, day=DAY)

    assert result["items"] == 1
    assert result["deferred"] == 0
    pack = (cfg.pack_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert len(pack.encode("utf-8")) <= material.PACK_MAX_BYTES
    assert int(size_file.read_text().split()[0]) == len(pack.encode("utf-8"))
    body = pack_bodies(pack)[0]
    assert body.count("z") == material.PACK_TEXT_MAX - len("Body "), "the body fills the budget"
    assert "обрезано" in body, "the cut is marked, so the agent knows to read the raw file"
    assert "## What it is" in pack and "## Experiment" in pack
    assert "- raw: " in pack


def test_two_big_items_both_fit_the_pack(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shared reserve is what lets two 60 000-char bodies share one pack."""
    seed_fetched_day(cfg, 2, body_chars=60_000)
    size_file = tmp_path / "pack-size"
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path, record_size=size_file)))

    result = material.distill(cfg, day=DAY)

    assert result["items"] == 2, "both items fit one `at` call"
    assert result["deferred"] == 0
    assert len(size_file.read_text().split()) == 1
    pack = (cfg.pack_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert len(pack.encode("utf-8")) <= material.PACK_MAX_BYTES
    with material.State(cfg.db_path) as state:
        items = state.items_for_date(DAY)
    assert material._pack_body_budget(items) == 46_250
    bodies = pack_bodies(pack)
    assert len(bodies) == 2
    assert [body.count("z") for body in bodies] == [46_250 - len("Body ")] * 2
    assert all(len(body) > material.PACK_TEXT_LIMIT for body in bodies)
    assert all("обрезано" in body for body in bodies)


def test_a_full_ten_item_pack_still_fits_in_one_at_call(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reserve must cover the overhead of a full pack, or item ten is silently deferred."""
    size_file = tmp_path / "pack-size"
    seed_fetched_day(cfg, material.PACK_MAX_ITEMS, body_chars=9_600)
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path, record_size=size_file)))

    result = material.distill(cfg, day=DAY)

    assert result["items"] == material.PACK_MAX_ITEMS, "a full day is still one `at` call"
    assert result["deferred"] == 0
    assert len(size_file.read_text().split()) == 1
    pack = (cfg.pack_dir / f"{DAY}.md").read_text(encoding="utf-8")
    assert len(pack.encode("utf-8")) <= material.PACK_MAX_BYTES
    budget = (material.PACK_MAX_BYTES - material.PACK_RESERVE_BYTES) // material.PACK_MAX_ITEMS
    assert budget == 9_250
    bodies = pack_bodies(pack)
    assert len(bodies) == material.PACK_MAX_ITEMS
    assert [body.count("z") for body in bodies] == [budget - len("Body ")] * len(bodies)


def test_distill_takes_one_item_even_past_the_byte_cap(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A section bigger than `PACK_MAX_BYTES` still packs: an empty pack would deadlock the day."""
    url = seed_fetched_day(cfg, 1)[0]
    with material.State(cfg.db_path) as state:
        for index in range(400):
            state.add_candidate(
                "hint",
                f"https://example.com/p/{index:04d}/" + "x" * 280,
                "example.com",
                parent=url,
                reason="hint:example.com",
                status="hint",
            )
    with material.State(cfg.db_path) as state:
        items = state.items_for_date(DAY, statuses=("fetched", "packed"))
    assert len(material._pack_text(cfg, DAY, items).encode("utf-8")) > material.PACK_MAX_BYTES

    monkeypatch.setenv(
        "MSFT_AT_BIN", str(fake_at(tmp_path, record_size=tmp_path / "pack-size"))
    )
    result = material.distill(cfg, day=DAY)

    assert result["items"] == 1, "the first item is always taken, whatever its section costs"
    assert result["ok"] is True
    assert result["deferred"] == 0


# --- memorize -------------------------------------------------------------


def test_memorize_dry_run_payload_shape(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path)))
    material.distill(cfg, day=DAY)

    posted: list[Any] = []
    monkeypatch.setattr(material, "post_page", lambda *args: posted.append(args) or (201, "{}"))
    result = material.memorize(cfg, day=DAY, dry_run=True)

    assert result["items"] == 1
    assert posted == []
    payload = result["payloads"][0]
    assert payload["workspace"] == "default"
    assert payload["project"] == "data"
    assert payload["kind"] == "note"
    assert payload["path"].startswith(f"projects/machinum-sft/material/{DAY}-")
    assert payload["path"].endswith(".md")
    assert "material" in payload["tags"] and "habr" in payload["tags"]
    assert "lora" in payload["tags"]
    assert payload["body"].startswith("# ")


def test_memorize_is_idempotent_and_writes_the_index(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path)))
    material.distill(cfg, day=DAY)

    posts: list[tuple[str, str, dict[str, Any]]] = []

    def fake_post(url: str, token: str, payload: dict[str, Any]) -> tuple[int, str]:
        posts.append((url, token, payload))
        return 201, '{"ok":true}'

    monkeypatch.setattr(material, "post_page", fake_post)
    first = material.memorize(cfg, day=DAY)
    assert first["ok"] is True
    assert first["memorized"] == 1
    assert len(posts) == 1
    assert posts[0][0].endswith("/admin/write-page")
    assert posts[0][1] == "test-token"
    index = (cfg.material_dir / "INDEX.md").read_text(encoding="utf-8")
    assert "| date | title | site | score | experiment |" in index
    assert "LoRA fine-tuning LLM" in index
    assert "| yes |" in index

    second = material.memorize(cfg, day=DAY)
    assert second["items"] == 0
    assert len(posts) == 1


def test_memorize_failure_keeps_the_note_and_exits_nonzero(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path)))
    material.distill(cfg, day=DAY)
    monkeypatch.setattr(material, "post_page", lambda *args: (500, "boom"))

    result = material.memorize(cfg, day=DAY)
    assert result["ok"] is False
    assert result["errors"] == 1
    with material.State(cfg.db_path) as state:
        item = state.items_for_date(DAY)[0]
    assert item["status"] == "error"
    note = Path(item["digest_path"]) if Path(item["digest_path"]).is_absolute() else None
    assert note is None or note.is_file()
    assert (cfg.notes_dir / DAY).is_dir()


def test_memorize_without_token_is_an_error(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path)))
    material.distill(cfg, day=DAY)
    monkeypatch.delenv("MSFT_MEMORY_TOKEN", raising=False)
    monkeypatch.setattr(material, "memory_token", lambda: None)
    with pytest.raises(material.MsftError, match="token"):
        material.memorize(cfg, day=DAY)


# --- report and status ----------------------------------------------------


def test_report_text_and_json(
    seeded: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG, WEAK]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "body about LoRA",
            },
            ("extract", "habr", "https://habr.com/ru/articles/900002"): {
                "site": "habr",
                "text": "old llm notes",
            },
        },
    )
    material.run(seeded, day=DAY, sites=["habr"], no_expand=True)
    payload = material.report(seeded, day=DAY)
    assert payload["date"] == DAY
    assert payload["queries"] == 1
    assert payload["items"] == 2
    assert payload["kept"] == 2
    assert payload["duplicates"] == 0
    assert payload["fetched"] == 2
    # the github/hf seeds were never executed (--site habr), so they are still new
    assert payload["frontier"] == 2
    assert payload["raw"] == 4
    assert payload["seconds"] >= 0

    text = material.format_report(payload)
    assert text.startswith(f"material {DAY}: 1 queries → 2 items, 2 kept (habr 2)")
    assert "frontier 2" in text and "raw 4 files" in text


def test_status_counts_offline(cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch) -> None:
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "body about LoRA",
            },
        },
    )
    material.run(cfg, day=DAY, sites=["habr"], no_expand=True)
    payload = material.status(cfg)
    assert payload["db_exists"] is True
    assert payload["candidates"]["new"] == 2
    assert payload["candidates"]["done"] == 1
    assert payload["frontier"] == 2
    assert payload["items"]["fetched"] == 1
    assert payload["dates"] == [DAY]
    assert payload["last_run"]["counters"]["queries"] == 1
    assert "test-token" not in json.dumps(payload)
    assert payload["memory_token"] is True


# --- CLI wiring ----------------------------------------------------------


def one_object(stdout: str) -> dict[str, Any]:
    assert stdout.startswith("{"), stdout
    assert stdout.rstrip("\n").count("\n") == 0, stdout
    parsed = json.loads(stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_cli_status_json(
    cfg: material.MaterialCfg, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["material", "status", "--json", "--config", str(cfg.path)]) == 0
    payload = one_object(capsys.readouterr().out)
    assert payload["config"].endswith("material.yaml")
    assert payload["frontier"] == 0


def test_cli_seed_and_report(
    cfg: material.MaterialCfg, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["material", "seed", "--config", str(cfg.path)]) == 0
    out = capsys.readouterr().out
    assert "3 seed(s)" in out and "added 3" in out

    assert main(["material", "report", "--json", "--config", str(cfg.path)]) == 0
    payload = one_object(capsys.readouterr().out)
    assert set(payload) >= {
        "date", "queries", "items", "kept", "duplicates", "fetched", "packed", "distilled",
        "memorized", "experiments", "frontier", "errors", "seconds",
    }


def test_cli_run_dry_run_is_offline(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    material.seed(cfg)
    monkeypatch.setattr(
        material, "run_webbridge", lambda *args: (_ for _ in ()).throw(AssertionError("network"))
    )
    assert main(["material", "run", "--dry-run", "--config", str(cfg.path)]) == 0
    out = capsys.readouterr().out
    assert "run plan" in out and "search habr" in out


def test_cli_run_exit_one_when_a_candidate_fails(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(monkeypatch, {})
    assert main(["material", "seed", "--config", str(cfg.path)]) == 0
    capsys.readouterr()
    code = main(["material", "run", "--limit", "1", "--site", "habr", "--config", str(cfg.path)])
    assert code == 1
    assert "1 error(s)" in capsys.readouterr().out


def test_cli_memorize_dry_run_prints_payloads(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_fetched(cfg, monkeypatch)
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path)))
    material.distill(cfg, day=DAY)
    monkeypatch.setattr(material, "post_page", lambda *args: (201, "{}"))
    argv = ["material", "memorize", "--dry-run", "--date", DAY, "--config", str(cfg.path)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "memorize plan" in out
    assert '"workspace": "default"' in out
    assert '"project": "data"' in out


# --- frontier refresh + query synthesis -----------------------------------

OLD = "2026-01-01T00:00:00+00:00"


def daily_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any):
    """A config with the shipped daily-loop defaults: refresh_days=3, synthesis on."""
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    monkeypatch.setenv("MSFT_MEMORY_TOKEN", "test-token")
    keys = {"limits__refresh_days": 3, "limits__synthesize_queries": True, **overrides}
    return material.load_material_config(write_config(tmp_path, **keys))


def finish_all(cfg: material.MaterialCfg, *, when: str | None = OLD) -> None:
    """Mark every candidate `done` and, optionally, backdate `updated_at` to `when`."""
    with material.State(cfg.db_path) as state:
        for row in state.conn.execute("SELECT id FROM candidates").fetchall():
            state.set_candidate(int(row["id"]), "done")
        if when is not None:
            state.conn.execute("UPDATE candidates SET updated_at = ?", (when,))
        state.conn.commit()


def test_refresh_makes_an_old_done_query_askable_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch)
    material.seed(cfg)
    with material.State(cfg.db_path) as state:
        for row in state.conn.execute("SELECT id FROM candidates").fetchall():
            state.set_candidate(int(row["id"]), "done")
        state.conn.execute(
            "UPDATE candidates SET updated_at = ? WHERE value = ?",
            (OLD, "LoRA fine-tuning LLM"),
        )
        state.conn.commit()
        due = state.candidates(limit=10, refresh_days=3)

    assert [cand["value"] for cand in due] == ["LoRA fine-tuning LLM"], (
        "the old node is due again, the two just-answered ones are not"
    )
    assert due[0]["status"] == "done"


def test_refresh_days_zero_is_the_old_one_shot_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch)
    material.seed(cfg)
    finish_all(cfg)
    with material.State(cfg.db_path) as state:
        assert state.candidates(limit=10) == []
        assert state.candidates(limit=10, refresh_days=0) == []
        assert state.frontier(refresh_days=0) == 0
        # and the window is what brings them back, not the status alone
        assert len(state.candidates(limit=10, refresh_days=3)) == 3


def test_done_url_candidates_never_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One explicit read of one page answers that node for ever, however old it gets."""
    cfg = daily_cfg(
        tmp_path,
        monkeypatch,
        seeds=[
            {"kind": "url", "site": "habr", "value": "https://habr.com/ru/articles/1044422/"},
            {"kind": "query", "site": "habr", "value": "LoRA fine-tuning LLM"},
        ],
    )
    material.seed(cfg)
    finish_all(cfg)
    with material.State(cfg.db_path) as state:
        due = state.candidates(limit=10, refresh_days=3)
    assert [cand["kind"] for cand in due] == ["query"]
    with material.State(cfg.db_path) as state:
        row = state.conn.execute("SELECT status FROM candidates WHERE kind = 'url'").fetchone()
    assert row["status"] == "done", "the url node stays answered; it is just never re-asked"


def test_refresh_rotation_is_oldest_updated_at_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch)
    material.seed(cfg)
    ages = {
        "hf": "2026-01-01T00:00:00+00:00",
        "github": "2026-02-01T00:00:00+00:00",
        "habr": "2026-03-01T00:00:00+00:00",
    }
    with material.State(cfg.db_path) as state:
        for row in state.conn.execute("SELECT id FROM candidates").fetchall():
            state.set_candidate(int(row["id"]), "done")
        for site, when in ages.items():
            state.conn.execute("UPDATE candidates SET updated_at = ? WHERE site = ?", (when, site))
        state.conn.commit()
        assert [cand["site"] for cand in state.candidates(limit=10, refresh_days=3)] == [
            "hf", "github", "habr"
        ]
        # A short run takes the stalest nodes, which is what makes the pool rotate.
        assert [cand["site"] for cand in state.candidates(limit=2, refresh_days=3)] == [
            "hf", "github"
        ]


def test_run_reasks_everything_after_the_refresh_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live bug: every candidate `done`, so the daily cron executed 0 queries for ever."""
    cfg = daily_cfg(tmp_path, monkeypatch, limits__synthesize_queries=False)
    material.seed(cfg)
    finish_all(cfg)
    bridge = install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([]),
            ("search", "github", "lora fine-tuning"): github_search([]),
            ("search", "hf", "lora"): hf_search([]),
        },
    )
    result = material.run(cfg, day=DAY, limit=3, no_expand=True)

    assert bridge.argv == [
        ["search", "habr", "LoRA fine-tuning LLM"],
        ["search", "github", "lora fine-tuning"],
        ["search", "hf", "lora"],
    ], "all three answered nodes are asked again, in the frontier's order"
    assert result["queries"] == 3
    assert result["refreshed"] == 3
    assert result["ok"] is True

    # Re-executing a node re-arms it: `updated_at` moved to now, so it is not due again
    # until another `refresh_days` have passed.
    with material.State(cfg.db_path) as state:
        assert state.candidates(limit=10, refresh_days=3) == []


def test_frontier_and_report_agree_with_candidates_after_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch)
    material.seed(cfg)
    finish_all(cfg)
    with material.State(cfg.db_path) as state:
        assert len(state.candidates(limit=500, refresh_days=3)) == 3
        assert state.frontier(refresh_days=3) == 3
    assert material.status(cfg)["frontier"] == 3, "status must not print 'frontier 0' with work due"
    assert material.report(cfg, day=DAY)["frontier"] == 3


def test_synthesize_queries_is_deterministic_and_skips_known_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch, product__keywords=["lora", "qlora", "peft"])
    material.seed(cfg)
    with material.State(cfg.db_path) as state:
        first = material.synthesize_queries(cfg, state, 3, day=DAY)
        assert first == [
            ("lora qlora", "synth:lora+qlora"),
            ("lora peft", "synth:lora+peft"),
            ("qlora peft", "synth:qlora+peft"),
        ]
        assert material.synthesize_queries(cfg, state, 3) == first, "no randomness, same order"
        assert len(material.synthesize_queries(cfg, state, 2)) == 2
        assert material.synthesize_queries(cfg, state, 0) == []

        # A pair already asked under *any* site and status is not asked again.
        state.add_candidate("query", "lora qlora", "hf")
        assert [value for value, _ in material.synthesize_queries(cfg, state, 3)] == [
            "lora peft", "qlora peft"
        ]


def test_synthesis_skips_hyphenated_scoring_stems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fine-tun` is a scoring stem: `lora fine-tun` is not a question any engine answers."""
    cfg = daily_cfg(
        tmp_path, monkeypatch, product__keywords=["lora", "fine-tun", "gguf"]
    )
    material.seed(cfg)
    with material.State(cfg.db_path) as state:
        assert material.synthesize_queries(cfg, state, 5) == [("lora gguf", "synth:lora+gguf")]


def test_synthesis_fills_a_short_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch, product__keywords=["lora", "qlora", "peft"])
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([]),
            ("search", "github", "lora fine-tuning"): github_search([]),
            ("search", "hf", "lora"): hf_search([]),
            ("search", "habr", "lora qlora"): habr_search([]),
            ("search", "github", "lora peft"): github_search([]),
            ("search", "hf", "qlora peft"): hf_search([]),
        },
    )
    result = material.run(cfg, day=DAY, limit=6, no_expand=True)

    assert result["queries"] == 6, "three seeds plus three synthesized questions"
    assert result["synthesized"] == 3
    with material.State(cfg.db_path) as state:
        rows = {
            str(row["value"]): dict(row) for row in state.conn.execute("SELECT * FROM candidates")
        }
    assert rows["lora qlora"]["reason"] == "synth:lora+qlora"
    assert rows["lora qlora"]["site"] == "habr", "round-robin over the seed sites, in order"
    assert rows["lora peft"]["site"] == "github"
    assert rows["qlora peft"]["site"] == "hf"
    assert all(rows[value]["status"] == "done" for value in ("lora qlora", "qlora peft"))


def test_synthesis_adds_nothing_when_the_budget_is_already_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch, product__keywords=["lora", "qlora", "peft"])
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([]),
            ("search", "github", "lora fine-tuning"): github_search([]),
            ("search", "hf", "lora"): hf_search([]),
        },
    )
    result = material.run(cfg, day=DAY, limit=3, no_expand=True)
    assert result["queries"] == 3
    assert result["synthesized"] == 0
    with material.State(cfg.db_path) as state:
        assert state.candidates_count() == 3, "a full budget must not grow the frontier"


def test_synthesis_respects_max_frontier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = daily_cfg(
        tmp_path,
        monkeypatch,
        product__keywords=["lora", "qlora", "peft"],
        limits__max_frontier=4,
    )
    material.seed(cfg)  # three fresh nodes, room for exactly one more
    result = material.run(cfg, day=DAY, limit=6, dry_run=True, no_expand=True)
    assert result["would_synthesize"] == 1
    assert result["queries_planned"] == 4


def test_synthesis_respects_a_full_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(
        tmp_path,
        monkeypatch,
        product__keywords=["lora", "qlora", "peft"],
        limits__max_frontier=3,
    )
    material.seed(cfg)
    result = material.run(cfg, day=DAY, limit=6, dry_run=True, no_expand=True)
    assert result["would_synthesize"] == 0
    assert result["queries_planned"] == 3


def test_dry_run_plan_shows_the_synthesized_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = daily_cfg(tmp_path, monkeypatch, product__keywords=["lora", "qlora", "peft"])
    material.seed(cfg)
    bridge = install(monkeypatch, {})
    result = material.run(cfg, day=DAY, dry_run=True, limit=6)

    assert bridge.calls == []
    assert result["queries_planned"] == 6
    synthesized = [step for step in result["plan"] if step.get("synthesized")]
    assert [step["value"] for step in synthesized] == ["lora qlora", "lora peft", "qlora peft"]
    assert [step["candidate"] for step in synthesized] == [
        "query/habr", "query/github", "query/hf"
    ]
    assert synthesized[0]["reason"] == "synth:lora+qlora"
    assert synthesized[0]["argv"] == ["search", "habr", "lora qlora"]
    with material.State(cfg.db_path) as state:
        assert state.candidates_count() == 3, "a dry run writes no synthesized row"


def test_config_defaults_when_the_daily_loop_keys_are_absent(tmp_path: Path) -> None:
    data = copy.deepcopy(BASE_CONFIG)
    del data["limits"]["refresh_days"]
    del data["limits"]["synthesize_queries"]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = material.load_material_config(path)
    assert cfg.limits.refresh_days == 3
    assert cfg.limits.synthesize_queries is True


def test_config_negative_refresh_days_is_a_config_error(tmp_path: Path) -> None:
    path = write_config(tmp_path, limits__refresh_days=-1)
    with pytest.raises(ConfigError, match="refresh_days"):
        material.load_material_config(path)


# --- comments, links and the tree ----------------------------------------


def comments_payload() -> dict[str, Any]:
    """A comment payload in the shape `/kek/v2/articles/<id>/comments/` returns.

    `threads` lists the root *and* the reply — that is the trap: a walker that treats `threads`
    as the root list renders the reply twice, which is what the side script already paid for.
    """
    return {
        "comments": {
            "1001": {
                "id": 1001, "parentId": None, "status": "published", "score": 7,
                "timePublished": "2026-11-09T12:30:00+00:00", "isPostAuthor": True,
                "author": {"alias": "alice", "fullname": "Alice"},
                "message": (
                    '<p>Обучал на 24 GB, batch 4. Код: '
                    '<a href="https://github.com/huggingface/peft">PEFT</a></p>'
                ),
                "children": ["1002"],
            },
            "1002": {
                "id": 1002, "parentId": 1001, "status": "published", "score": 1,
                "timePublished": "2026-11-09T13:00:00+00:00", "isPostAuthor": False,
                "author": {"alias": "bob", "fullname": ""},
                "message": "<p>Методика из https://arxiv.org/abs/2401.01234 работает лучше</p>",
                "children": [],
            },
        },
        "threads": [1001, 1002],
    }


def twitter_search(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"site": "twitter", "op": "search", "mode": "all", "items": rows, "count": len(rows)}


TWEET = {
    "author": "Nate Friedman",
    "handle": "@natfriedman",
    "text": "QLoRA adapter on one 16 GB card, dataset of 12k pairs — repo in the reply",
    "time": "2026-11-09T10:00:00.000Z",
    "link": "/natfriedman/status/1234567890",
}

TWEET_URL = "https://x.com/UnslothAI/status/2120536946774769860"
TWEET_TEXT = "Qwen3.5 8B-27B GGUF is out: https://t.co/qwen"
TWEET_RENDERED = f"@UnslothAI: {TWEET_TEXT}"
# One `extract twitter <status>` in the shape sites/twitter.yml returns: the focal tweet under
# `items`, and a page shell at the top level whose `title` is just "X".
TWEET_EXTRACT = {
    "site": "twitter",
    "op": "extract",
    "url": TWEET_URL,
    "title": "X",
    "text": TWEET_RENDERED,
    "count": 1,
    "items": [
        {
            "author": "Unsloth AI",
            "handle": "@UnslothAI",
            "text": TWEET_TEXT,
            "time": "2026-11-11T18:30:00.000Z",
            "link": "/UnslothAI/status/2120536946774769860",
        }
    ],
}


def tweet_bridge(extract: Any, evaluate: Any):
    """A webbridge stand-in for the two calls `_fetch_body` makes for a tweet."""

    def call(
        argv: list[str], timeout: float, json_flag: bool = True
    ) -> tuple[int, str, str]:
        if argv[:2] == ["extract", "twitter"]:
            return 0, json.dumps(extract), ""
        if argv[:2] == ["send", "evaluate"]:
            return 0, json.dumps(evaluate), ""
        raise AssertionError(f"unexpected webbridge call: {argv}")

    return call


def test_comments_are_appended_to_the_raw_text_and_counted(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The article body is half the page: the comment tree carries the author's own numbers."""
    material.seed(cfg)
    asked: list[str] = []
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Article body about LoRA",
            },
        },
    )
    monkeypatch.setattr(
        material,
        "_fetch_comments",
        lambda article, timeout: (asked.append(article) or comments_payload(), None),
    )
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=True)

    assert asked == ["900001"], "the comment API is keyed by the numeric article id"
    assert result["comments_fetched"] == 1
    assert result["comments_failed"] == 0
    assert result["fetched"] == 1 and result["errors"] == 0
    with material.State(cfg.db_path) as state:
        row = state.item("https://habr.com/ru/articles/900001")
    assert row is not None and row["comments"] == 2

    slug = material.slugify(str(row["url"]))
    text = (cfg.raw_dir / DAY / f"habr-{slug}.txt").read_text(encoding="utf-8")
    assert "Article body about LoRA" in text
    assert material.COMMENT_HEADING in text
    assert "24 GB" in text, "the number only the comments carried must reach the raw text"
    assert "https://arxiv.org/abs/2401.01234" in text
    assert (cfg.raw_dir / DAY / f"habr-{slug}.comments.json").is_file()


def test_comment_fetch_failure_keeps_the_item_and_increments_a_counter(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comment timeout must never lose the article a browser already rendered."""
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Article body about LoRA",
            },
        },
    )
    monkeypatch.setattr(material, "_fetch_comments", lambda article, timeout: (None, "HTTP 503"))
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=True)

    assert result["comments_failed"] == 1
    assert result["comments_fetched"] == 0
    assert result["fetched"] == 1, "the item is fetched, only its comments are missing"
    assert result["errors"] == 0
    with material.State(cfg.db_path) as state:
        row = state.item("https://habr.com/ru/articles/900001")
    assert row is not None
    assert row["status"] == "fetched"
    assert row["comments"] == 0
    text = (cfg.raw_dir / DAY / f"habr-{material.slugify(str(row['url']))}.txt").read_text(
        encoding="utf-8"
    )
    assert "Article body about LoRA" in text
    assert material.COMMENT_HEADING not in text


def test_comment_roots_come_only_from_parentless_comments() -> None:
    block, count = material.comment_block(comments_payload(), "900001")
    assert count == 2
    assert block.count("arxiv.org") == 1, "the reply is listed in `threads` too — render it once"
    assert block.count("24 GB") == 1
    assert "[автор статьи]" in block
    assert "#### 2. bob" in block, "the reply is a child of the root, not a second root"
    assert block.count("score 7") == 1


def test_links_are_extracted_from_body_and_comments(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Body naming a repo: https://github.com/huggingface/peft",
            },
        },
    )
    monkeypatch.setattr(
        material, "_fetch_comments", lambda article, timeout: (comments_payload(), None)
    )
    material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    item_url = "https://habr.com/ru/articles/900001"
    with material.State(cfg.db_path) as state:
        rows = {
            str(row["value"]): dict(row)
            for row in state.conn.execute(
                "SELECT * FROM candidates WHERE parent = ?", (item_url,)
            )
        }
    assert "https://github.com/huggingface/peft" in rows, "the body's link is a node"
    assert "https://arxiv.org/abs/2401.01234" in rows, "a comment's link is a node too"
    assert rows["https://arxiv.org/abs/2401.01234"]["kind"] == "hint"


def test_github_link_becomes_one_url_candidate_with_parent_and_reason(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Repo: https://github.com/huggingface/peft/ and again "
                        "https://github.com/huggingface/peft?utm_source=habr",
            },
        },
    )
    material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    with material.State(cfg.db_path) as state:
        rows = [
            dict(row)
            for row in state.conn.execute(
                "SELECT * FROM candidates WHERE kind = 'url' AND value LIKE 'https://github%'"
            )
        ]
    assert len(rows) == 1, "the same link twice — and with utm noise — is one node"
    row = rows[0]
    assert row["value"] == "https://github.com/huggingface/peft"
    assert row["site"] == "github"
    assert row["parent"] == "https://habr.com/ru/articles/900001"
    assert row["reason"] == "link:github.com", "the reason names the host the lead came from"
    assert row["status"] == "new"
    assert row["priority"] > 0, "the item's score is the link node's priority"


def test_arxiv_link_is_a_hint_not_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    monkeypatch.setenv("MSFT_MEMORY_TOKEN", "test-token")
    cfg = material.load_material_config(
        write_config(
            tmp_path, seeds=[{"kind": "query", "site": "habr", "value": "LoRA fine-tuning LLM"}]
        )
    )
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Paper: https://arxiv.org/abs/2401.01234",
            },
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    with material.State(cfg.db_path) as state:
        row = state.conn.execute(
            "SELECT * FROM candidates WHERE value = ?", ("https://arxiv.org/abs/2401.01234",)
        ).fetchone()
        frontier = state.frontier()
        due = state.candidates(limit=10)
    assert row is not None
    assert dict(row)["kind"] == "hint"
    assert dict(row)["status"] == "hint"
    assert dict(row)["site"] == "arxiv.org"
    assert result["links_queued"] == 0, "no recipe, so no fetch job"
    assert frontier == 2, "only the tag/author children are frontier work, not the hint"
    assert all(cand["kind"] != "hint" for cand in due), "a hint is never handed to the runner"


def test_links_per_item_caps_the_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    monkeypatch.setenv("MSFT_MEMORY_TOKEN", "test-token")
    cfg = material.load_material_config(
        write_config(
            tmp_path,
            limits__links_per_item=1,
            seeds=[{"kind": "query", "site": "habr", "value": "LoRA fine-tuning LLM"}],
        )
    )
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "https://github.com/a/one https://github.com/b/two "
                        "https://github.com/c/three",
            },
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    with material.State(cfg.db_path) as state:
        rows = state.conn.execute(
            "SELECT value FROM candidates WHERE kind = 'url' AND reason = 'link:github.com'"
        ).fetchall()
    assert [str(row["value"]) for row in rows] == ["https://github.com/a/one"]
    assert result["links_queued"] == 1


def test_config_links_per_item_defaults_to_four(tmp_path: Path) -> None:
    assert "links_per_item" not in BASE_CONFIG["limits"]
    cfg = material.load_material_config(write_config(tmp_path))
    assert cfg.limits.links_per_item == 4
    assert material.LimitsCfg().links_per_item == 4


# --- url shapes the recipe refuses ---------------------------------------
#
# `link_target` asks the recipe itself, with a resolve-only probe, whether it can read a URL's
# shape. These tests can the probe's verdict; the fail-open and canary cases are the ones that
# keep a broken or missing client from silently emptying the frontier.

HABR_QUERY = "LoRA fine-tuning LLM"
HABR_ITEM = "https://habr.com/ru/articles/900001"
GIST = "https://gist.github.com/alice/abc123"
PROFILE = "https://habr.com/ru/users/bob"
CANARY_ITEM = "https://github.com/huggingface/peft"
REFUSAL = "site config error: resolve.path: 'x' does not match '^[^/]+/[^/]+$'"


def habr_only_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> material.MaterialCfg:
    """A seeded config with one habr query, so a link test's frontier is only its children."""
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    monkeypatch.setenv("MSFT_MEMORY_TOKEN", "test-token")
    cfg = material.load_material_config(
        write_config(
            tmp_path, seeds=[{"kind": "query", "site": "habr", "value": HABR_QUERY}]
        )
    )
    material.seed(cfg)
    return cfg


def add_url_candidate(
    cfg: material.MaterialCfg,
    value: str,
    site: str,
    *,
    reason: str = "link:github.com",
    parent: str | None = HABR_ITEM,
) -> None:
    """A row as an earlier run would have left it, minted from a link (or by an operator)."""
    with material.State(cfg.db_path) as state:
        state.add_candidate("url", value, site, parent=parent, reason=reason, priority=0.5)


def add_fetched_item(cfg: material.MaterialCfg, url: str, site: str) -> None:
    """A page this site's recipe really did read — the canary `_recipe_reads_site` probes."""
    with material.State(cfg.db_path) as state:
        state.add_item(
            material._new_item_row({"url": url, "site": site, "title": "canary"}, 0.5, "fetched")
        )


def candidate_row(cfg: material.MaterialCfg, value: str) -> dict[str, Any]:
    with material.State(cfg.db_path) as state:
        row = state.conn.execute(
            "SELECT * FROM candidates WHERE value = ?", (value,)
        ).fetchone()
    assert row is not None
    return dict(row)


def test_gist_link_is_a_hint_because_the_recipe_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = habr_only_cfg(tmp_path, monkeypatch)
    install(
        monkeypatch,
        {
            ("search", "habr", HABR_QUERY): habr_search([STRONG]),
            ("extract", "habr", HABR_ITEM): {"site": "habr", "text": f"A gist: {GIST}"},
            ("extract", "github", GIST, "--dry-run"): (2, "", REFUSAL),
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    row = candidate_row(cfg, GIST)
    assert row["kind"] == "hint"
    assert row["status"] == "hint"
    assert row["site"] == "gist.github.com", "the hint keeps the host it came from"
    assert result["links_queued"] == 0, "a hint is never a fetch job"
    with material.State(cfg.db_path) as state:
        assert state.frontier() == 2, "only the tag/author children are work"
        assert all(cand["value"] != GIST for cand in state.candidates(limit=10))


def test_habr_profile_link_is_a_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = habr_only_cfg(tmp_path, monkeypatch)
    install(
        monkeypatch,
        {
            ("search", "habr", HABR_QUERY): habr_search([STRONG]),
            ("extract", "habr", HABR_ITEM): {"site": "habr", "text": f"By {PROFILE} here"},
            ("extract", "habr", PROFILE, "--dry-run"): (2, "", REFUSAL),
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    row = candidate_row(cfg, PROFILE)
    assert row["kind"] == "hint", "a profile page is not an article, even on a recipe host"
    assert result["links_queued"] == 0


def test_probe_timeout_keeps_the_link_a_fetch_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail open: an unknown probe outcome must not lose the lead."""
    cfg = habr_only_cfg(tmp_path, monkeypatch)
    install(
        monkeypatch,
        {
            ("search", "habr", HABR_QUERY): habr_search([STRONG]),
            ("extract", "habr", HABR_ITEM): {"site": "habr", "text": f"Repo: {CANARY_ITEM}"},
            ("extract", "github", CANARY_ITEM, "--dry-run"): (
                124, "", "webbridge timed out after 30s",
            ),
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    row = candidate_row(cfg, CANARY_ITEM)
    assert row["kind"] == "url"
    assert row["status"] == "new"
    assert result["links_queued"] == 1


def test_probe_success_with_an_empty_url_keeps_the_link_a_fetch_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hf` is `mode: api`: a valid model URL resolves to `"url": ""`. rc 0 is the verdict."""
    cfg = habr_only_cfg(tmp_path, monkeypatch)
    model = "https://huggingface.co/unsloth/Qwen3.5-8B-GGUF"
    install(
        monkeypatch,
        {
            ("search", "habr", HABR_QUERY): habr_search([STRONG]),
            ("extract", "habr", HABR_ITEM): {"site": "habr", "text": f"Model: {model}"},
            ("extract", "hf", model, "--dry-run"): (0, json.dumps({"url": ""}), ""),
        },
    )
    result = material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    row = candidate_row(cfg, model)
    assert row["kind"] == "url", "never test the payload's url — rc 0 is what proves the shape"
    assert result["links_queued"] == 1


def test_a_refusal_does_not_outvote_a_rotted_recipe(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rotted regex would turn every link of a site into a hint; that must be loud, not quiet."""
    link = "https://github.com/broken/shape"
    add_url_candidate(cfg, link, "github")
    add_fetched_item(cfg, CANARY_ITEM, "github")
    install(
        monkeypatch,
        {
            ("extract", "github", link): (2, "", REFUSAL),
            ("extract", "github", CANARY_ITEM, "--dry-run"): (2, "", REFUSAL),
        },
    )
    result = material.run(cfg, day=DAY, sites=["github"])

    assert result["ok"] is False, "a broken recipe is the run's problem, not the link's"
    assert result["errors"] == 1
    assert result["dropped"] == 0
    row = candidate_row(cfg, link)
    assert row["kind"] == "url", "the refusal was not trusted, so it stays a fetch job"
    assert row["status"] == "error"


def test_a_refused_link_with_a_healthy_recipe_is_dropped(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = "https://github.com/broken/shape"
    add_url_candidate(cfg, link, "github")
    add_fetched_item(cfg, CANARY_ITEM, "github")
    install(
        monkeypatch,
        {
            ("extract", "github", link): (2, "", REFUSAL),
            ("extract", "github", CANARY_ITEM, "--dry-run"): (
                0, json.dumps({"url": CANARY_ITEM}), "",
            ),
        },
    )
    result = material.run(cfg, day=DAY, sites=["github"])

    assert result["ok"] is True
    assert result["errors"] == 0
    assert result["dropped"] == 1
    assert result["links_unreadable"] == 1
    assert candidate_row(cfg, link)["status"] == "dropped", "never selected, so never retried"


def test_the_canary_tolerates_an_item_url_the_recipe_never_accepted(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`items.url` can be an API URL the recipe never accepted; an older canary still proves it."""
    api_url = "https://api.github.com/repos/ggml-org/llama.cpp"
    add_url_candidate(cfg, GIST, "github")
    add_fetched_item(cfg, CANARY_ITEM, "github")
    add_fetched_item(cfg, api_url, "github")
    bridge = install(
        monkeypatch,
        {
            ("extract", "github", GIST): (2, "", REFUSAL),
            ("extract", "github", GIST, "--dry-run"): (2, "", REFUSAL),
            ("extract", "github", CANARY_ITEM, "--dry-run"): (
                0, json.dumps({"url": CANARY_ITEM}), "",
            ),
        },
    )
    result = material.run(cfg, day=DAY, sites=["github"])

    api_probe = ["extract", "github", api_url, "--dry-run"]
    healthy_probe = ["extract", "github", CANARY_ITEM, "--dry-run"]
    assert api_probe in bridge.argv and healthy_probe in bridge.argv
    assert bridge.argv.index(api_probe) < bridge.argv.index(healthy_probe), (
        "canaries are tried newest first"
    )
    assert result["ok"] is True
    assert result["errors"] == 0
    assert result["dropped"] == 1
    assert candidate_row(cfg, GIST)["status"] == "dropped"


def test_a_manual_url_refused_by_the_recipe_stays_a_visible_error(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator's explicit URL is never silently dropped."""
    install(
        monkeypatch,
        {
            # `add` probes: the shape passed then, and the recipe rotted afterwards.
            ("extract", "habr", PROFILE, "--dry-run"): (0, json.dumps({"url": PROFILE}), ""),
            ("extract", "habr", PROFILE): (2, "", REFUSAL),
        },
    )
    material.add_urls(cfg, [PROFILE])
    result = material.run(cfg, day=DAY, sites=["habr"])

    assert result["ok"] is False
    assert result["errors"] == 1
    assert result["dropped"] == 0
    assert result["links_unreadable"] == 0
    assert candidate_row(cfg, PROFILE)["status"] == "error"


def test_a_link_minted_404_is_dropped_without_failing_the_run(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = "https://github.com/gone/repo"
    add_url_candidate(cfg, link, "github")
    install(
        monkeypatch,
        {("extract", "github", link): (1, "", "HTTP Error 404: Not Found")},
    )
    result = material.run(cfg, day=DAY, sites=["github"])

    assert result["ok"] is True
    assert result["errors"] == 0
    assert result["links_unreadable"] == 1
    assert result["dropped"] == 1, "a 404 can never work, so no retry is spent on it"
    assert candidate_row(cfg, link)["status"] == "dropped"


def test_a_link_minted_502_is_retryable_but_does_not_fail_the_run(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = "https://github.com/flaky/repo"
    add_url_candidate(cfg, link, "github")
    install(
        monkeypatch,
        {("extract", "github", link): (1, "", "HTTP Error 502: Bad Gateway")},
    )
    result = material.run(cfg, day=DAY, sites=["github"])

    assert result["ok"] is True
    assert result["errors"] == 0
    assert result["links_unreadable"] == 1
    assert result["dropped"] == 0
    assert candidate_row(cfg, link)["status"] == "error", "still visible and retryable"


def test_add_urls_rejects_a_shape_the_recipe_cannot_read(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        monkeypatch,
        {("extract", "habr", CANONICAL, "--dry-run"): (2, "", REFUSAL)},
    )
    with pytest.raises(ConfigError, match="habr cannot read"):
        material.add_urls(cfg, [ARTICLE])
    assert not cfg.db_path.exists(), "a rejected URL must not create state"


def test_a_rescored_link_item_failure_is_not_a_run_error(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link-discovered item whose body is a stale tab must not fail the day.

    It reaches `_fetch_pending` (a rescore flipped it to `new`), which must carry the item's
    `parent` so the loop knows the row came from a lead rather than from a query.
    """
    url = "https://habr.com/ru/articles/900009"
    stale = {
        "site": "habr",
        "found": False,
        "url": "https://www.google.com/search?udm=50&q=go",
        "text": '""',
    }
    install(monkeypatch, {("extract", "habr", url): stale})
    with material.State(cfg.db_path) as state:
        state.add_item(
            material._new_item_row(
                {"url": url, "site": "habr", "parent": HABR_ITEM}, 0.5, "new"
            )
        )

    result = material.run(cfg, day=DAY, limit=0)

    assert result["ok"] is True
    assert result["errors"] == 0
    assert result["links_unreadable"] == 1
    with material.State(cfg.db_path) as state:
        row = state.item(url)
    assert row is not None and row["status"] == "error", "the item row is still visible"


# --- a re-queued item is re-read, never rebuilt from its summary ----------


HFT_BODY = "Catalogue card: Qwen3.8 31B, Uncensored IQ 9/10, Reasoning 8/10.\n" * 60


def pending_item(url: str, site: str, summary: str, **fields: Any) -> dict[str, Any]:
    """The row `_fetch_pending` sees: a summary and no page payload (`source`)."""
    row = material._new_item_row({"url": url, "site": site, "summary": summary}, 0.5, "new")
    row.update(fields)
    return row


def test_a_requeued_url_site_is_read_from_the_page(cfg: material.MaterialCfg, monkeypatch) -> None:
    """`hft` has no branch of its own: re-queueing it used to store its own summary."""
    summary = "hft.md catalogue preview. " * 50
    bridge = install(
        monkeypatch,
        {
            ("extract", "hft", HFT_PAGE): {
                "site": "hft",
                "op": "extract",
                "mode": "text",
                "url": HFT_PAGE + "/",
                "title": "Uncensored AI",
                "text": HFT_BODY,
            }
        },
    )
    with material.State(cfg.db_path) as state:
        state.add_item(pending_item(HFT_PAGE, "hft", summary))

    result = material.run(cfg, day=DAY, limit=0, no_expand=True)

    assert bridge.argv == [["extract", "hft", HFT_PAGE]], "one extract, no query"
    assert result["fetched"] == 1
    assert result["ok"] is True
    with material.State(cfg.db_path) as state:
        row = state.item(HFT_PAGE)
    assert row is not None and row["status"] == "fetched"
    written = cfg.resolve(row["raw_path"]).read_text(encoding="utf-8")
    assert written == material.normalize_text(HFT_BODY) + "\n", "the page, not the summary"
    provenance = json.loads(cfg.resolve(row["raw_path"]).with_suffix(".json").read_text("utf-8"))
    assert provenance["text"] == HFT_BODY


def test_a_requeued_url_site_errors_when_the_page_cannot_be_read(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary fallback is gone: a dead daemon is an error, not a 1.2 KB body."""
    summary = "SUMMARY-MARKER " + "catalogue preview. " * 50
    install(monkeypatch, {("extract", "hft", HFT_PAGE): (1, "", "daemon down")})
    good = cfg.date_dir(DAY) / "hft-uncensored-ai.txt"
    good.parent.mkdir(parents=True, exist_ok=True)
    good.write_text("the 40 KB catalogue the last good run stored\n", encoding="utf-8")
    with material.State(cfg.db_path) as state:
        state.add_item(pending_item(HFT_PAGE, "hft", summary, raw_path=cfg.display(good)))

    result = material.run(cfg, day=DAY, limit=0, no_expand=True)

    assert result["errors"] == 1
    assert result["ok"] is False
    with material.State(cfg.db_path) as state:
        row = state.item(HFT_PAGE)
    assert row is not None and row["status"] == "error"
    assert "daemon down" in (row["error"] or "")
    assert row["raw_path"] == cfg.display(good), "the good body is not un-pointed"
    assert good.read_text(encoding="utf-8") == "the 40 KB catalogue the last good run stored\n"
    assert list(cfg.date_dir(DAY).glob("*.txt")) == [good], "nothing new was written"
    assert "SUMMARY-MARKER" not in good.read_text(encoding="utf-8")


def test_a_requeued_github_item_is_extracted_and_rendered(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending github row has no `source`; rendering the row alone produced `None` fields."""
    record = {
        "site": "github",
        "op": "extract",
        "mode": "api",
        "url": CANARY_ITEM,
        "repo": "huggingface/peft",
        "description": "Parameter-efficient fine-tuning",
        "stars": 17500,
        "language": "Python",
    }
    bridge = install(monkeypatch, {("extract", "github", CANARY_ITEM): record})
    with material.State(cfg.db_path) as state:
        state.add_item(pending_item(CANARY_ITEM, "github", "github: huggingface/peft"))

    result = material.run(cfg, day=DAY, limit=0, no_expand=True)

    assert ["extract", "github", CANARY_ITEM] in bridge.argv
    assert result["fetched"] == 1
    with material.State(cfg.db_path) as state:
        row = state.item(CANARY_ITEM)
    assert row is not None and row["status"] == "fetched"
    written = cfg.resolve(row["raw_path"]).read_text(encoding="utf-8")
    assert written == material.normalize_text(material._render_entry("github", record)) + "\n"
    assert "repo: huggingface/peft" in written
    assert "stars: 17500" in written
    assert "None" not in written


def test_a_live_payload_is_used_without_another_extract(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The query path already brought the record; re-reading it would spend a call for nothing."""
    record = {"repo": "huggingface/peft", "url": CANARY_ITEM, "stars": 17500}
    bridge = install(monkeypatch, {})
    fetched = material._fetch_body(cfg, {"site": "github", "source": record}, CANARY_ITEM, DAY)

    assert bridge.argv == []
    assert fetched.error is None
    assert fetched.raw is record
    assert fetched.body == material._render_entry("github", record)


def test_the_probe_cache_makes_two_links_one_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    twin = habr_item(
        "https://habr.com/ru/articles/900003/",
        "QLoRA adapter training notes",
        summary="<p>QLoRA adapter, VRAM, dataset of 20k samples</p>",
        published="Mon, 09 Nov 2026 12:00:00 +0000",
        tags=["machine learning"],
        author="dave",
    )
    cfg = habr_only_cfg(tmp_path, monkeypatch)
    bridge = install(
        monkeypatch,
        {
            ("search", "habr", HABR_QUERY): habr_search([STRONG, twin]),
            ("extract", "habr", HABR_ITEM): {"site": "habr", "text": f"Repo: {CANARY_ITEM}"},
            ("extract", "habr", "https://habr.com/ru/articles/900003"): {
                "site": "habr", "text": f"Same repo again: {CANARY_ITEM}",
            },
        },
    )
    material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    probe = ["extract", "github", CANARY_ITEM, "--dry-run"]
    assert bridge.argv.count(probe) == 1, "the run's cache answers the second body locally"


def test_a_link_the_db_already_knows_is_not_probed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-reading an article must not pay a probe for every link it already produced."""
    cfg = habr_only_cfg(tmp_path, monkeypatch)
    body = f"Repo: {CANARY_ITEM}"
    probe = ["extract", "github", CANARY_ITEM, "--dry-run"]
    bridge = install(monkeypatch, {tuple(probe): {"url": CANARY_ITEM}})

    with material.State(cfg.db_path) as state:
        first = material._record_links(state, cfg, HABR_ITEM, 1.0, {"body": body}, {}, 5.0)
    assert first == (1, 0)
    assert bridge.argv.count(probe) == 1

    with material.State(cfg.db_path) as state:
        second = material._record_links(state, cfg, HABR_ITEM, 1.0, {"body": body}, {}, 5.0)
    assert second == (0, 1), "the row dedupes to `known`, exactly as before"
    assert bridge.argv.count(probe) == 1, "the DB already classified it, so no probe is spent"


def test_twitter_search_rows_parse_into_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    monkeypatch.setenv("MSFT_MEMORY_TOKEN", "test-token")
    cfg = material.load_material_config(
        write_config(
            tmp_path, seeds=[{"kind": "query", "site": "twitter", "value": "lora"}]
        )
    )
    material.seed(cfg)
    bridge = install(
        monkeypatch, {("search", "twitter", "lora"): twitter_search([TWEET])}
    )
    result = material.run(cfg, day=DAY, sites=["twitter"], no_expand=True)

    assert bridge.argv[0] == ["search", "twitter", "lora"]
    assert result["items"] == 1
    assert result["fetched"] == 1
    with material.State(cfg.db_path) as state:
        row = state.item("https://x.com/natfriedman/status/1234567890")
    assert row is not None, "a relative permalink gets the x.com prefix"
    assert row["site"] == "twitter"
    assert row["author"] == "@natfriedman", "the handle is the author, the display name is not"
    assert row["published"] == "2026-11-09T10:00:00.000Z"
    assert "QLoRA adapter" in (row["summary"] or "")


def test_twitter_url_candidate_uses_the_focal_tweet(cfg: material.MaterialCfg) -> None:
    cand = {"kind": "url", "site": "twitter", "value": TWEET_URL}
    items = material.items_from_payload(TWEET_EXTRACT, cand, cfg)

    assert len(items) == 1
    item = items[0]
    assert item["url"] == TWEET_URL, "the permalink comes from the tweet, not the page shell"
    assert item["author"] == "@UnslothAI"
    assert item["published"] == "2026-11-11T18:30:00.000Z"
    assert item["summary"] == TWEET_TEXT
    assert item["title"] != "X", "the page shell's title is not the tweet"


def test_twitter_url_candidate_without_items_falls_back_to_the_page(
    cfg: material.MaterialCfg,
) -> None:
    cand = {"kind": "url", "site": "twitter", "value": TWEET_URL}
    items = material.items_from_payload({**TWEET_EXTRACT, "items": []}, cand, cfg)

    assert len(items) == 1
    assert items[0]["url"] == TWEET_URL
    assert items[0]["summary"] == TWEET_RENDERED


def test_fetch_body_twitter_appends_resolved_links_once(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluate = {
        "ok": True,
        "data": {
            "type": "object",
            "value": {
                "href": TWEET_URL,
                "links": [
                    "https://t.co/qwen",
                    "https://t.co/qwen",
                    "https://x.com/UnslothAI",
                ],
            },
        },
    }
    monkeypatch.setattr(material, "run_webbridge", tweet_bridge(TWEET_EXTRACT, evaluate))
    resolved: list[str] = []

    def fake_resolve(url: str) -> str:
        resolved.append(url)
        return "https://huggingface.co/unsloth/Qwen3.5-8B-GGUF"

    monkeypatch.setattr(material, "_resolve_short_link", fake_resolve)

    entry = {"site": "twitter", "url": TWEET_URL, "summary": TWEET_TEXT, "source": {}}
    result = material._fetch_body(cfg, entry, TWEET_URL, DAY)

    link = "https://huggingface.co/unsloth/Qwen3.5-8B-GGUF"
    assert result.error is None
    assert resolved == ["https://t.co/qwen", "https://t.co/qwen"], "resolve before dedupe"
    assert result.body == TWEET_RENDERED + f"\n\n# Links\n\n- {link}", "the link appears once"
    assert result.article == result.body, "link extraction reads `article`, not `body`"


def test_fetch_body_twitter_survives_a_failed_evaluate(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        material, "run_webbridge", tweet_bridge(TWEET_EXTRACT, {"ok": False, "error": "no tab"})
    )
    monkeypatch.setattr(
        material, "_resolve_short_link", lambda url: pytest.fail("nothing to resolve")
    )

    entry = {"site": "twitter", "url": TWEET_URL, "summary": TWEET_TEXT, "source": {}}
    result = material._fetch_body(cfg, entry, TWEET_URL, DAY)

    assert result.error is None, "the tweet text is already stored; links are best-effort"
    assert result.body == TWEET_RENDERED
    assert "# Links" not in result.body


def test_tweet_links_from_a_moved_tab_are_ignored(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluate = {
        "ok": True,
        "data": {
            "value": {
                "href": "https://x.com/someone/status/999",
                "links": ["https://t.co/other"],
            }
        },
    }
    monkeypatch.setattr(material, "run_webbridge", tweet_bridge(TWEET_EXTRACT, evaluate))
    monkeypatch.setattr(
        material,
        "_resolve_short_link",
        lambda url: pytest.fail("a moved tab's links must not be resolved"),
    )

    entry = {"site": "twitter", "url": TWEET_URL, "summary": TWEET_TEXT, "source": {}}
    result = material._fetch_body(cfg, entry, TWEET_URL, DAY)

    assert result.error is None
    assert "# Links" not in result.body, "same host is not the same status"


def test_short_link_target_is_read_from_the_redirect_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """t.co answers a script with 200 + a redirect page, not a 301 with a Location header."""

    class Response:
        def __init__(self, body: str) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def geturl(self) -> str:
            return "https://t.co/qwen"

        def read(self, size: int) -> bytes:
            assert size > 0
            return self.body.encode("utf-8")

    body = (
        '<head><noscript><META http-equiv="refresh" '
        'content="0;URL=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF"></noscript>'
        '<title>https://huggingface.co/unsloth/Qwen3.8-27B-GGUF</title></head>'
        '<script>window.opener = null; location.replace("https:\\/\\/huggingface.co'
        '\\/unsloth\\/Qwen3.8-27B-GGUF")</script>'
    )
    seen: list[Any] = []
    monkeypatch.setattr(
        material.urllib.request,
        "urlopen",
        lambda request, timeout: seen.append(request) or Response(body),
    )

    assert (
        material._resolve_short_link("https://t.co/qwen")
        == "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF"
    )
    assert seen and seen[0].get_method() == "GET", "a HEAD gets no redirect page to read"


def test_run_webbridge_appends_json_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "tunnelstack"
    script = root / "src" / "webbridge.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stand-in\n", encoding="utf-8")
    monkeypatch.setenv("MSFT_TUNNELSTACK", str(root))

    class Done:
        returncode = 0
        stdout = "{}"
        stderr = ""

    seen: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Done:
        seen.append(list(command))
        return Done()

    monkeypatch.setattr(material.subprocess, "run", fake_run)
    REAL_RUN_WEBBRIDGE(["extract", "twitter", TWEET_URL], 5)
    REAL_RUN_WEBBRIDGE(["send", "evaluate", "{}"], 5, json_flag=False)

    assert seen[0][-1] == "--json"
    assert "--json" not in seen[1], "send has no --json option and rejects it"


def test_grok_answer_is_one_item_with_a_stable_url(cfg: material.MaterialCfg) -> None:
    cand = {"kind": "note", "site": "grok", "value": "which rank for a 16 GB card?"}
    payload = {"site": "grok", "op": "ask", "text": "r=16 with alpha 32.", "answered": True}
    first = material.items_from_payload(payload, cand, cfg)
    again = material.items_from_payload(payload, cand, cfg)
    other = material.items_from_payload(
        payload, {**cand, "value": "another question"}, cfg
    )

    assert len(first) == 1
    assert first[0]["site"] == "grok"
    assert first[0]["url"].startswith("https://grok.com/")
    assert first[0]["url"] == again[0]["url"], "the same question dedupes onto one item"
    assert other[0]["url"] != first[0]["url"], "a different question is a different item"
    assert material.items_from_payload({"site": "grok", "text": ""}, cand, cfg) == []
    assert material.items_from_payload(None, cand, cfg) == []


def test_grok_node_argv_is_ask(cfg: material.MaterialCfg) -> None:
    assert material.candidate_argv(
        {"kind": "note", "site": "grok", "value": "which rank?"}
    ) == ["ask", "grok", "which rank?"]
    assert "grok" in material.SITES and "twitter" in material.SITES


def test_twitter_expands_a_good_account_and_grok_expands_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    cfg = material.load_material_config(
        write_config(
            tmp_path,
            expansion__per_site={"habr": ["tag", "author"], "twitter": ["author"]},
            expansion__allow_kinds=["tag", "author", "repo", "model"],
        )
    )
    tweet = {"site": "twitter", "author": "@natfriedman", "tags": []}
    children = material._children(tweet, 0.9, cfg)
    assert [child.kind for child in children] == ["author"]
    assert material.candidate_argv(
        {"kind": "author", "site": "twitter", "value": "@natfriedman"}
    ) == ["search", "twitter", "from:natfriedman"]

    assert material._children({"site": "twitter", "author": "@x", "tags": []}, 0.5, cfg) == []
    assert material._children({"site": "grok", "author": None, "tags": []}, 1.0, cfg) == []


def chain_two_items(cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run twice so one fetched page has a child page of its own; returns the child URL."""
    child = habr_item(
        "https://habr.com/ru/articles/900010/",
        "QLoRA adapter follow-up",
        summary="<p>llm adapter, 12k dataset</p>",
        published="Tue, 10 Nov 2026 09:00:00 +0000",
        tags=["ml"],
        author="alice",
    )
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Body naming a repo: https://github.com/huggingface/peft",
            },
            ("search", "habr", "machine learning"): habr_search([child]),
            ("extract", "habr", "https://habr.com/ru/articles/900010"): {
                "site": "habr",
                "text": "follow-up body about QLoRA",
            },
        },
    )
    material.seed(cfg)
    material.run(cfg, day=DAY, sites=["habr"], no_expand=False)
    material.run(cfg, day=DAY, sites=["habr"], limit=1, no_expand=False)
    return "https://habr.com/ru/articles/900010"


def test_tree_contains_the_parent_child_relationship(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_url = chain_two_items(cfg, monkeypatch)
    payload = material.tree(cfg, day=DAY)

    parents = {root["url"]: root for root in payload["roots"]}
    parent_url = "https://habr.com/ru/articles/900001"
    assert parent_url in parents, "the article whose parent is outside the day is a root"
    nested = [kid["url"] for kid in parents[parent_url]["children"]]
    assert child_url in nested, "the tag child's page hangs off the article that named the tag"
    assert {"tag", "author"} <= {cand["kind"] for cand in parents[parent_url]["candidates"]}
    assert [link["url"] for link in parents[parent_url]["links"]] == [
        "https://github.com/huggingface/peft"
    ], "a link row is drawn under `links`, not twice as a candidate"
    assert any(link["url"] == "https://github.com/huggingface/peft"
               for link in parents[parent_url]["links"])

    text = material.format_tree(payload)
    lines = text.splitlines()
    parent_at = next(
        i for i, line in enumerate(lines) if "LoRA fine-tuning LLM на практике" in line
    )
    child_at = next(i for i, line in enumerate(lines) if "QLoRA adapter follow-up" in line)
    assert parent_at < child_at
    assert lines[child_at].startswith(("│  ", "   ")), "the child is indented under its parent"
    assert "link github  https://github.com/huggingface/peft" in text


def test_tree_on_a_day_without_items_is_one_line(cfg: material.MaterialCfg) -> None:
    material.seed(cfg)
    payload = material.tree(cfg, day="2026-01-01")
    assert payload["items"] == 0 and payload["roots"] == []
    text = material.format_tree(payload)
    assert text.count("\n") == 0
    assert "no items" in text


def test_cli_material_tree_json(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chain_two_items(cfg, monkeypatch)
    argv = ["material", "tree", "--date", DAY, "--json", "--config", str(cfg.path)]
    assert main(argv) == 0
    payload = one_object(capsys.readouterr().out)
    assert payload["date"] == DAY
    assert payload["items"] == 2
    assert payload["roots"]

    assert main(["material", "tree", "--date", DAY, "--config", str(cfg.path)]) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"material tree {DAY} ·")


def test_state_migrates_an_old_database_in_place(tmp_path: Path) -> None:
    """`CREATE TABLE IF NOT EXISTS` never adds a column, so an existing DB is altered, not lost."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE items (url TEXT PRIMARY KEY, site TEXT NOT NULL, title TEXT, "
        "author TEXT, published TEXT, tags TEXT, summary TEXT, content_sha TEXT, "
        "score REAL NOT NULL DEFAULT 0, status TEXT NOT NULL, raw_path TEXT, "
        "digest_path TEXT, parent TEXT, error TEXT, first_seen_at TEXT NOT NULL, "
        "fetched_at TEXT, distilled_at TEXT, memorized_at TEXT);"
        "INSERT INTO items (url, site, status, first_seen_at) "
        "VALUES ('https://habr.com/ru/articles/1', 'habr', 'fetched', '2026-11-01');"
    )
    conn.commit()
    conn.close()

    with material.State(db) as state:
        columns = {str(row["name"]) for row in state.conn.execute("PRAGMA table_info(items)")}
        kept = state.item("https://habr.com/ru/articles/1")
        state.update_item("https://habr.com/ru/articles/1", comments=12)
    assert "comments" in columns
    assert kept is not None, "the rows an older database holds survive the migration"
    assert kept["comments"] == 0

    with material.State(db) as state:  # idempotent: a second open is not a second ALTER
        assert state.item("https://habr.com/ru/articles/1")["comments"] == 12


def test_pack_shows_the_links_and_the_comment_count(
    cfg: material.MaterialCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    material.seed(cfg)
    install(
        monkeypatch,
        {
            ("search", "habr", "LoRA fine-tuning LLM"): habr_search([STRONG]),
            ("extract", "habr", "https://habr.com/ru/articles/900001"): {
                "site": "habr",
                "text": "Body naming a repo: https://github.com/huggingface/peft "
                        "and a paper: https://arxiv.org/abs/2401.01234",
            },
        },
    )
    monkeypatch.setattr(
        material, "_fetch_comments", lambda article, timeout: (comments_payload(), None)
    )
    material.run(cfg, day=DAY, sites=["habr"], no_expand=False)

    with material.State(cfg.db_path) as state:
        items = state.items_for_date(DAY)
    text = material._pack_text(cfg, DAY, items)
    assert "- comments: 2" in text
    assert "- links: 1 fetchable, 1 hint" in text
    assert "  - queued · github · https://github.com/huggingface/peft" in text
    assert "  - hint · arxiv.org · https://arxiv.org/abs/2401.01234" in text
    assert "## Links" in text
    assert "Комментарии" in text, "the rules tell the agent what the comment delimiter means"
