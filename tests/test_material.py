"""Tests for `msft material` — the daily research state machine. Offline: no network, no docker.

Every remote call goes through two module-level seams (`run_webbridge`, `post_page`) and the
distill step goes through `MSFT_AT_BIN`, which points at a fake script here. Nothing in this
file opens a socket.
"""

from __future__ import annotations

import copy
import json
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from msft import material
from msft.cli import main
from msft.material import ConfigError

DAY = "2026-11-10"

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

    def __call__(self, argv: list[str], timeout: float) -> tuple[int, str, str]:
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
    assert len(cfg.seeds) == 6
    assert cfg.limits.items_per_run == 12
    assert cfg.scoring.expand_min == 0.6
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


def test_normalize_text_caps_at_40k() -> None:
    out = material.normalize_text("x" * 50_000)
    assert len(out) == material.TEXT_LIMIT == 40_000


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


def fake_at(tmp_path: Path, *, status: str = "ok", exit_code: int = 0) -> Path:
    """A stand-in for `at`: reads the pack, writes one note per slug, prints a JSON summary."""
    script = tmp_path / "fake-at"
    script.write_text(
        f"""#!/usr/bin/env python3
import json, pathlib, sys
pack = pathlib.Path(sys.argv[sys.argv.index("-f") + 1])
notes = pathlib.Path({str(tmp_path / "material" / "notes")!r}) / pack.stem
notes.mkdir(parents=True, exist_ok=True)
slugs = [line.split(": ", 1)[1] for line in pack.read_text().splitlines()
         if line.startswith("- slug: ")]
for slug in slugs:
    (notes / f"{{slug}}.md").write_text(
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
print("Now using node v22.20.0 (npm v10.9.3)")
print(json.dumps({{"status": {status!r}, "files": len(slugs)}}))
sys.exit({exit_code})
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_pack_contains_brief_rules_schema_and_truncated_body(
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
    # the section body is truncated to 8 000 chars: "Body " plus z's, so the run is 7 995 long
    assert text.count("z") == 8_000 - len("Body ")
    assert "z" * 8_000 not in text


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
    monkeypatch.setenv("MSFT_AT_BIN", str(fake_at(tmp_path, status="failed", exit_code=1)))

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
