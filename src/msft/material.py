"""`msft material` — the daily research state machine.

collect → score → dedupe → fetch → expand → pack → distill → memorize → report.

The loop exists so this product keeps seeing current LoRA/SFT/dataset/eval material without a
human reading seven sites a day. Everything expensive or remote is a seam:

- ``run_webbridge(argv, timeout)`` drives tunnelstack's Kimi WebBridge client (the seven sites);
- ``post_page(url, token, payload)`` writes a distilled note into the ai-memory wiki;
- distill shells out to ``at`` instead of running an agent in-process.

Tests monkeypatch the first two and drop a fake ``at`` on PATH; nothing here touches the network
at import time, and every ``--dry-run`` is read-only.

State is SQLite under ``<material_dir>/.state/material.db``: the row, never the files on disk, is
what says where an item is. Files under ``docs/material/`` are the payloads the rows point at.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import yaml

from msft.config import MsftError, redact

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TUNNELSTACK = Path("/home/boris/tunnelstack")
DEFAULT_MEMORY_URL = "http://127.0.0.1:49374"
DEFAULT_CONFIG_NAME = "configs/material.yaml"

SITES = ("habr", "github", "hf", "google-ai", "deepseek", "qwen")
CANDIDATE_KINDS = ("query", "tag", "author", "repo", "model", "note")
ITEM_STATUSES = (
    "seen",
    "scored_out",
    "duplicate",
    "fetched",
    "packed",
    "distilled",
    "memorized",
    "error",
)

TEXT_LIMIT = 40_000
PACK_TEXT_LIMIT = 8_000
NOTE_TAG_ALLOW = ("lora", "qlora", "peft", "sft", "dpo", "grpo", "unsloth", "trl", "axolotl",
                  "dataset")

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  value TEXT NOT NULL,
  site TEXT NOT NULL,
  parent TEXT,
  reason TEXT,
  priority REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'new',
  attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(kind, value, site)
);
CREATE TABLE IF NOT EXISTS items (
  url TEXT PRIMARY KEY,
  site TEXT NOT NULL, title TEXT, author TEXT, published TEXT,
  tags TEXT,
  summary TEXT, content_sha TEXT,
  score REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  raw_path TEXT, digest_path TEXT, parent TEXT, error TEXT,
  first_seen_at TEXT NOT NULL,
  fetched_at TEXT, distilled_at TEXT, memorized_at TEXT
);
CREATE INDEX IF NOT EXISTS items_sha ON items(content_sha);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT,
  argv TEXT, counters TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

_DATE_DIR_RE = re.compile(r"/raw/(\d{4}-\d{2}-\d{2})/")
_LORA_RE = re.compile(r"(?i)\blora(?:wan)?\b")
_BLOCK_RE = re.compile(r"(?is)<(script|style|nav|aside|footer)\b[^>]*>.*?</\1\s*>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")
_SPACES_RE = re.compile(r"[ \t\ufeff]{2,}")
_BLANKS_RE = re.compile(r"\n{3,}")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ConfigError(MsftError):
    """`configs/material.yaml` is missing, unreadable or incomplete (exit code 2)."""


# --- environment ---------------------------------------------------------


def _env(name: str, default: str) -> str:
    """Environment override, else the default. Read at call time so tests can patch it."""
    value = os.environ.get(name)
    return value if value else default


def config_path() -> Path:
    """`--config`, else `MSFT_MATERIAL_CONFIG`, else `<repo>/configs/material.yaml`."""
    return Path(_env("MSFT_MATERIAL_CONFIG", str(REPO_ROOT / DEFAULT_CONFIG_NAME)))


def material_dir() -> Path:
    """`MSFT_MATERIAL_DIR`, else `<repo>/docs/material`."""
    return Path(_env("MSFT_MATERIAL_DIR", str(REPO_ROOT / "docs" / "material")))


def tunnelstack_root() -> Path:
    """Where `src/webbridge.py` lives: `MSFT_TUNNELSTACK`, else `/home/boris/tunnelstack`."""
    return Path(_env("MSFT_TUNNELSTACK", str(DEFAULT_TUNNELSTACK)))


def memory_url() -> str:
    """ai-memory base URL; loopback by default."""
    return _env("MSFT_MEMORY_URL", DEFAULT_MEMORY_URL).rstrip("/")


def memory_token() -> str | None:
    """`MSFT_MEMORY_TOKEN`, else the first `AI_MEMORY_AUTH_TOKEN=` line of the memory env file.

    The value is never logged, printed or put into a payload; callers only ask whether it is set.
    """
    explicit = os.environ.get("MSFT_MEMORY_TOKEN")
    if explicit:
        return explicit
    env_file = REPO_ROOT / "deploy" / "memory" / ".env"
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("AI_MEMORY_AUTH_TOKEN="):
                return line.split("=", 1)[1].strip() or None
    except OSError:
        return None
    return None


def at_bin() -> Path:
    """The `at` CLI used for the distill step."""
    return Path(_env("MSFT_AT_BIN", str(Path.home() / ".local" / "bin" / "at")))


def today() -> str:
    """Local calendar date, the default for `--date`."""
    return date_cls.today().isoformat()


def _now() -> str:
    """UTC timestamp, seconds precision: enough to order rows, stable to compare."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    """Progress on stderr, so `--json` keeps stdout to one object."""
    print(redact(message), file=sys.stderr)


# --- config --------------------------------------------------------------


@dataclass
class ProductCfg:
    name: str
    brief: str
    keywords: list[str]
    negative: list[str] = field(default_factory=list)
    domain_terms: list[str] = field(default_factory=list)


@dataclass
class SeedCfg:
    kind: str
    value: str
    site: str
    reason: str | None = None


@dataclass
class LimitsCfg:
    queries_per_run: int = 6
    items_per_run: int = 12
    items_per_site: int = 6
    max_frontier: int = 200
    children_per_item: int = 3
    fetch_timeout_s: int = 120
    ask_timeout_s: int = 240


@dataclass
class ScoringCfg:
    keep_min: float = 0.35
    expand_min: float = 0.60
    recency_days: list[int] = field(default_factory=lambda: [30, 180])
    site_prior: dict[str, float] = field(default_factory=dict)


@dataclass
class ExpansionCfg:
    allow_kinds: list[str] = field(default_factory=list)
    per_site: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DistillCfg:
    agent: str = "ds-code"
    timeout_ms: int = 900_000
    notes_dir: str = "notes"


@dataclass
class MemoryCfg:
    workspace: str = "default"
    project: str = "data"
    path_prefix: str = "projects/machinum-sft/material"


@dataclass
class MaterialCfg:
    """Everything one material run needs, plus the paths resolved from the environment."""

    path: Path
    product: ProductCfg
    seeds: list[SeedCfg]
    limits: LimitsCfg
    scoring: ScoringCfg
    expansion: ExpansionCfg
    distill: DistillCfg
    memory: MemoryCfg
    material_dir: Path
    repo: Path

    # --- derived paths ---------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.material_dir / "raw"

    @property
    def pack_dir(self) -> Path:
        return self.material_dir / "pack"

    @property
    def notes_dir(self) -> Path:
        return self.material_dir / self.distill.notes_dir

    @property
    def db_path(self) -> Path:
        return self.material_dir / ".state" / "material.db"

    def date_dir(self, day: str) -> Path:
        """`raw/<date>` — the folder every item of one day lands in."""
        return self.raw_dir / day

    def display(self, path: Path) -> str:
        """Repo-relative path when possible, absolute otherwise (tests use tmp dirs)."""
        try:
            return path.relative_to(self.repo).as_posix()
        except ValueError:
            return str(path)

    def resolve(self, path: str | None) -> Path | None:
        """Turn a stored `raw_path` / `digest_path` back into a real path."""
        if not path:
            return None
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.repo / candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": str(self.path),
            "material_dir": str(self.material_dir),
            "product": self.product.name,
            "seeds": len(self.seeds),
            "limits": self.limits.__dict__,
            "scoring": self.scoring.__dict__,
            "distill_agent": self.distill.agent,
        }


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    """A mandatory key, or a ConfigError naming the section."""
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"material config: '{where}' is missing required key '{key}'")
    return mapping[key]


def _as_str(value: Any, where: str, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"material config: '{where}.{key}' must be a non-empty string")
    return value


def _as_int(value: Any, where: str, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"material config: '{where}.{key}' must be a number")
    return int(value)


def _as_float(value: Any, where: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"material config: '{where}.{key}' must be a number")
    return float(value)


def _as_list(value: Any, where: str, key: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"material config: '{where}.{key}' must be a non-empty list")
    return value


def _as_str_list(value: Any, where: str, key: str) -> list[str]:
    items = _as_list(value, where, key)
    if not all(isinstance(item, str) and item for item in items):
        raise ConfigError(f"material config: '{where}.{key}' must contain only non-empty strings")
    return list(items)


def _as_mapping(value: Any, where: str, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"material config: '{where}.{key}' must be a mapping")
    return value


_TOP_SECTIONS = ("product", "seeds", "limits", "scoring", "expansion", "distill", "memory")


def _product_cfg(raw: dict[str, Any]) -> ProductCfg:
    return ProductCfg(
        name=_as_str(_require(raw, "name", "product"), "product", "name"),
        brief=_as_str(_require(raw, "brief", "product"), "product", "brief"),
        keywords=_as_str_list(_require(raw, "keywords", "product"), "product", "keywords"),
        negative=_as_str_list(_require(raw, "negative", "product"), "product", "negative"),
        domain_terms=_as_str_list(
            _require(raw, "domain_terms", "product"), "product", "domain_terms"
        ),
    )


def _seed_cfg(raw: Any, index: int) -> SeedCfg:
    where = f"seeds[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"material config: '{where}' must be a mapping")
    kind = _as_str(_require(raw, "kind", where), where, "kind")
    if kind not in CANDIDATE_KINDS:
        raise ConfigError(
            f"material config: '{where}.kind' is '{kind}', expected one of "
            f"{', '.join(CANDIDATE_KINDS)}"
        )
    site = _as_str(_require(raw, "site", where), where, "site")
    if site not in SITES:
        raise ConfigError(
            f"material config: '{where}.site' is '{site}', expected one of {', '.join(SITES)}"
        )
    reason = raw.get("reason")
    return SeedCfg(
        kind=kind,
        value=_as_str(_require(raw, "value", where), where, "value"),
        site=site,
        reason=str(reason) if reason is not None else None,
    )


def _limits_cfg(raw: dict[str, Any]) -> LimitsCfg:
    return LimitsCfg(
        queries_per_run=_as_int(_require(raw, "queries_per_run", "limits"), "limits",
                                "queries_per_run"),
        items_per_run=_as_int(_require(raw, "items_per_run", "limits"), "limits",
                              "items_per_run"),
        items_per_site=_as_int(_require(raw, "items_per_site", "limits"), "limits",
                               "items_per_site"),
        max_frontier=_as_int(_require(raw, "max_frontier", "limits"), "limits", "max_frontier"),
        children_per_item=_as_int(_require(raw, "children_per_item", "limits"), "limits",
                                  "children_per_item"),
        fetch_timeout_s=_as_int(_require(raw, "fetch_timeout_s", "limits"), "limits",
                                "fetch_timeout_s"),
        ask_timeout_s=_as_int(_require(raw, "ask_timeout_s", "limits"), "limits",
                              "ask_timeout_s"),
    )


def _scoring_cfg(raw: dict[str, Any]) -> ScoringCfg:
    days = _as_list(_require(raw, "recency_days", "scoring"), "scoring", "recency_days")
    if len(days) != 2:
        raise ConfigError("material config: 'scoring.recency_days' must hold exactly two numbers")
    priors = _as_mapping(_require(raw, "site_prior", "scoring"), "scoring", "site_prior")
    cfg = ScoringCfg(
        keep_min=_as_float(_require(raw, "keep_min", "scoring"), "scoring", "keep_min"),
        expand_min=_as_float(_require(raw, "expand_min", "scoring"), "scoring", "expand_min"),
        recency_days=[_as_int(days[0], "scoring", "recency_days"),
                      _as_int(days[1], "scoring", "recency_days")],
        site_prior={str(k): _as_float(v, "scoring", "site_prior") for k, v in priors.items()},
    )
    if not 0.0 <= cfg.keep_min <= 1.0 or not 0.0 <= cfg.expand_min <= 1.0:
        raise ConfigError("material config: keep_min/expand_min must be fractions in [0, 1]")
    if cfg.expand_min < cfg.keep_min:
        raise ConfigError("material config: expand_min below keep_min can never be reached")
    return cfg


def _expansion_cfg(raw: dict[str, Any]) -> ExpansionCfg:
    per_site_raw = _as_mapping(
        _require(raw, "per_site", "expansion"), "expansion", "per_site"
    )
    allow = _as_str_list(_require(raw, "allow_kinds", "expansion"), "expansion", "allow_kinds")
    unknown = sorted(set(allow) - set(CANDIDATE_KINDS))
    if unknown:
        raise ConfigError(f"material config: unknown expansion kind(s): {', '.join(unknown)}")
    per_site: dict[str, list[str]] = {}
    for site, kinds in per_site_raw.items():
        if site not in SITES:
            raise ConfigError(f"material config: 'expansion.per_site' names unknown site '{site}'")
        per_site[str(site)] = _as_str_list(kinds, "expansion.per_site", str(site))
    return ExpansionCfg(allow_kinds=allow, per_site=per_site)


def load_material_config(path: str | Path | None = None) -> MaterialCfg:
    """Load and validate `configs/material.yaml`; any hole is a ConfigError (exit 2)."""
    p = Path(path) if path else config_path()
    if not p.is_file():
        raise ConfigError(f"material config not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level")
    unknown = sorted(set(data) - set(_TOP_SECTIONS))
    if unknown:
        raise ConfigError(f"material config: unknown top-level key(s): {', '.join(unknown)}")
    missing = [name for name in _TOP_SECTIONS if name not in data]
    if missing:
        raise ConfigError(f"material config: missing section(s): {', '.join(missing)}")

    seeds_raw = data["seeds"]
    if not isinstance(seeds_raw, list):
        raise ConfigError("material config: 'seeds' must be a list")
    dist = data["distill"]
    if not isinstance(dist, dict):
        raise ConfigError("material config: 'distill' must be a mapping")
    memory = data["memory"]
    if not isinstance(memory, dict):
        raise ConfigError("material config: 'memory' must be a mapping")

    cfg = MaterialCfg(
        path=p,
        product=_product_cfg(_as_mapping(data["product"], "product", "product")),
        seeds=[_seed_cfg(raw, index) for index, raw in enumerate(seeds_raw)],
        limits=_limits_cfg(_as_mapping(data["limits"], "limits", "limits")),
        scoring=_scoring_cfg(_as_mapping(data["scoring"], "scoring", "scoring")),
        expansion=_expansion_cfg(_as_mapping(data["expansion"], "expansion", "expansion")),
        distill=DistillCfg(
            agent=_as_str(_require(dist, "agent", "distill"), "distill", "agent"),
            timeout_ms=_as_int(_require(dist, "timeout_ms", "distill"), "distill", "timeout_ms"),
            notes_dir=_as_str(_require(dist, "notes_dir", "distill"), "distill", "notes_dir"),
        ),
        memory=MemoryCfg(
            workspace=_as_str(_require(memory, "workspace", "memory"), "memory", "workspace"),
            project=_as_str(_require(memory, "project", "memory"), "memory", "project"),
            path_prefix=_as_str(
                _require(memory, "path_prefix", "memory"), "memory", "path_prefix"
            ).rstrip("/"),
        ),
        material_dir=material_dir(),
        repo=REPO_ROOT,
    )
    return cfg


# --- state ---------------------------------------------------------------


class State:
    """The SQLite state machine. Rows are the truth; files are payloads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def __enter__(self) -> State:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # --- candidates ------------------------------------------------------
    def add_candidate(
        self,
        kind: str,
        value: str,
        site: str,
        *,
        parent: str | None = None,
        reason: str | None = None,
        priority: float = 0.0,
    ) -> bool:
        """Insert a frontier node; False when UNIQUE(kind, value, site) already had it."""
        now = _now()
        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO candidates "
            "(kind, value, site, parent, reason, priority, status, attempts, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, 'new', 0, ?, ?)",
            (kind, value, site, parent, reason, float(priority), now, now),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def candidates(self, limit: int = 10, sites: list[str] | None = None) -> list[dict[str, Any]]:
        """New frontier nodes, best first; ties by insertion order."""
        sql = "SELECT * FROM candidates WHERE status = 'new'"
        params: list[Any] = []
        if sites:
            sql += f" AND site IN ({', '.join('?' for _ in sites)})"
            params.extend(sites)
        sql += " ORDER BY priority DESC, id ASC LIMIT ?"
        params.append(int(limit))
        return [dict(row) for row in self.conn.execute(sql, params)]

    def candidates_count(self, status: str | None = None) -> int:
        if status is None:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM candidates WHERE status = ?", (status,)
            ).fetchone()
        return int(row["n"])

    def set_candidate(self, cand_id: int, status: str, error: str | None = None) -> None:
        """Every candidate transition is one row update (attempts always increments)."""
        self.conn.execute(
            "UPDATE candidates SET status = ?, error = ?, attempts = attempts + 1, "
            "updated_at = ? WHERE id = ?",
            (status, error, _now(), int(cand_id)),
        )
        self.conn.commit()

    def frontier(self) -> int:
        return self.candidates_count("new")

    # --- items -----------------------------------------------------------
    def item(self, url: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM items WHERE url = ?", (url,)).fetchone()
        return dict(row) if row else None

    def sha_owner(self, sha: str, url: str) -> str | None:
        """URL of a different item with the same content hash (cross-post detection)."""
        row = self.conn.execute(
            "SELECT url FROM items WHERE content_sha = ? AND url != ? LIMIT 1", (sha, url)
        ).fetchone()
        return str(row["url"]) if row else None

    def add_item(self, row: dict[str, Any]) -> None:
        """Insert one item row; the caller has already checked the URL is new."""
        self.conn.execute(
            "INSERT INTO items (url, site, title, author, published, tags, summary, "
            "content_sha, score, status, raw_path, digest_path, parent, error, first_seen_at, "
            "fetched_at, distilled_at, memorized_at) "
            "VALUES (:url, :site, :title, :author, :published, :tags, :summary, :content_sha, "
            ":score, :status, :raw_path, :digest_path, :parent, :error, :first_seen_at, "
            ":fetched_at, :distilled_at, :memorized_at)",
            row,
        )
        self.conn.commit()

    def update_item(self, url: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "title", "author", "published", "tags", "summary", "content_sha", "score",
            "status", "raw_path", "digest_path", "parent", "error", "fetched_at",
            "distilled_at", "memorized_at",
        }
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise MsftError(f"cannot update unknown item field(s): {', '.join(unknown)}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [*fields.values(), url]
        self.conn.execute(f"UPDATE items SET {assignments} WHERE url = ?", values)
        self.conn.commit()

    def items_for_date(self, day: str, statuses: tuple[str, ...] | None = None) -> list[dict]:
        """Items whose raw payload landed in `raw/<day>/`; status filters are applied in SQL."""
        sql = "SELECT * FROM items WHERE raw_path LIKE ?"
        params: list[Any] = [f"%/raw/{day}/%"]
        if statuses:
            sql += f" AND status IN ({', '.join('?' for _ in statuses)})"
            params.extend(statuses)
        sql += " ORDER BY score DESC, url ASC"
        return [dict(row) for row in self.conn.execute(sql, params)]

    def items_by_status(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM items GROUP BY status ORDER BY status"
        )
        return {str(row["status"]): int(row["n"]) for row in rows}

    def items_total(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"])

    def memorized(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM items WHERE status = 'memorized' ORDER BY first_seen_at DESC, url ASC"
        )
        return [dict(row) for row in rows]

    def dates(self) -> list[str]:
        found: set[str] = set()
        for row in self.conn.execute("SELECT raw_path FROM items WHERE raw_path IS NOT NULL"):
            match = _DATE_DIR_RE.search(str(row["raw_path"]))
            if match:
                found.add(match.group(1))
        return sorted(found, reverse=True)

    # --- runs ------------------------------------------------------------
    def start_run(self, argv: list[str]) -> int:
        cursor = self.conn.execute(
            "INSERT INTO runs (started_at, argv) VALUES (?, ?)",
            (_now(), json.dumps(argv, ensure_ascii=False)),
        )
        self.conn.commit()
        if cursor.lastrowid is None:  # pragma: no cover - sqlite always reports one here
            raise MsftError("sqlite did not return a row id for the new run")
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, counters: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, counters = ? WHERE id = ?",
            (_now(), json.dumps(counters, ensure_ascii=False), int(run_id)),
        )
        self.conn.commit()

    def runs_for_date(self, day: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE started_at LIKE ? ORDER BY id ASC", (f"{day}%",)
        )
        return [dict(row) for row in rows]

    def last_run(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    # --- kv --------------------------------------------------------------
    def put(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None


# --- text helpers --------------------------------------------------------


def _fold(text: str) -> str:
    """Lowercase and normalise the hyphens sites mix (Habr uses U+2011 inside names)."""
    return (text or "").replace("\u2011", "-").replace("\u2010", "-").lower()


def normalize_text(text: str) -> str:
    """Strip script/style/nav/aside/footer blocks and tags, tidy whitespace, cap at 40 000."""
    if not text:
        return ""
    out = _BLOCK_RE.sub(" ", text)
    out = _TAG_RE.sub(" ", out)
    out = html.unescape(out)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in out.splitlines()]
    out = "\n".join(lines)
    out = _BLANKS_RE.sub("\n\n", out)
    return out.strip()[:TEXT_LIMIT].strip()


def strip_html(text: str) -> str:
    """Habr's RSS `description` is a raw HTML excerpt; one line of prose is what we store."""
    return normalize_text(text).replace("\n", " ").strip()


def canonical_url(url: str) -> str:
    """Lowercase the host, drop `utm_*` params, the fragment and a trailing slash."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urllib.parse.urlsplit(raw)
    scheme = parts.scheme.lower()
    host = parts.netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if (scheme == "http" and host.endswith(":80")) or (
        scheme == "https" and host.endswith(":443")
    ):
        host = host.rsplit(":", 1)[0]
    pairs = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    path = parts.path.rstrip("/") if parts.path not in ("", "/") else ""
    return urllib.parse.urlunsplit((scheme, host, path, urllib.parse.urlencode(pairs), ""))


def slugify(url: str) -> str:
    """60 chars of the URL path in `[a-z0-9-]`, plus 8 chars of its sha1 (stable and unique)."""
    parts = urllib.parse.urlsplit(url)
    raw = urllib.parse.unquote(parts.path).strip("/") or parts.netloc or "item"
    base = _SLUG_RE.sub("-", raw.lower()).strip("-")[:60].strip("-") or "item"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def content_sha(title: str, body: str) -> str:
    """Hash of the normalised `title + body`; equal hashes mean the same page in two URLs."""
    payload = f"{title or ''}\n{body or ''}"
    return hashlib.sha256(normalize_text(payload).encode("utf-8")).hexdigest()


def _parse_date(value: str | None) -> date_cls | None:
    """ISO-8601 or RFC-822, else None (an unparsable date earns no recency bonus)."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    iso = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError):
        return None


def _recency_bonus(published: str | None, days: list[int], now: date_cls) -> float:
    """0.20 inside the first window, 0.10 inside the second, 0 for older or unparsable."""
    when = _parse_date(published)
    if when is None:
        return 0.0
    age = max(0, (now - when).days)
    if age <= days[0]:
        return 0.20
    if age <= days[1]:
        return 0.10
    return 0.0


def score_item(item: dict[str, Any], cfg: MaterialCfg, now: date_cls | None = None) -> float:
    """Score one item in [0, 1], rounded to 3 places. 0 means 'drop it', never 'unknown'."""
    today_date = now or date_cls.today()
    product = cfg.product
    title = _fold(str(item.get("title") or ""))
    summary = _fold(str(item.get("summary") or ""))
    blob = f"{title}\n{summary}"

    for word in product.negative:
        if _fold(word) and _fold(word) in blob:
            return 0.0
    # "LoRa" is a radio standard, not an adapter: without a domain word the item is noise.
    if _LORA_RE.search(blob) and not any(_fold(term) in blob for term in product.domain_terms):
        return 0.0

    hits = 0
    for keyword in product.keywords:
        needle = _fold(keyword)
        if not needle:
            continue
        if needle in title:
            hits += 3
        elif needle in summary:
            hits += 1
    denominator = max(1, min(len(product.keywords), 5))
    score = 0.6 * hits / denominator
    score += float(cfg.scoring.site_prior.get(str(item.get("site") or ""), 0.0))
    score += _recency_bonus(item.get("published"), cfg.scoring.recency_days, today_date)
    if (item.get("stars") or 0) >= 100 or (item.get("downloads") or 0) >= 1000:
        score += 0.15
    return round(min(1.0, max(0.0, score)), 3)


# --- the two network seams ------------------------------------------------


def run_webbridge(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """Run tunnelstack's WebBridge client and return `(rc, stdout, stderr)`.

    Tests replace this; nothing else in the module shells out to the network. A missing client
    or a timeout is a `(rc != 0, ...)` result, not an exception: one dead site must not stop
    the run.
    """
    script = tunnelstack_root() / "src" / "webbridge.py"
    if not script.is_file():
        return 127, "", f"webbridge client not found at {script}"
    command = [sys.executable, str(script), *argv, "--json"]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script.parent.parent),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"webbridge timed out after {timeout:g}s: {' '.join(argv)}"
    except OSError as exc:
        return 127, "", f"cannot run webbridge: {exc}"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def post_page(url: str, token: str, payload: dict[str, Any]) -> tuple[int, str]:
    """POST one page to ai-memory; returns `(status, body)` with status 0 for transport errors."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return int(exc.code), body
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc)


# --- webbridge payload handling -----------------------------------------


def _loads(text: str) -> Any:
    """Parse the client's stdout; unparsable JSON is a None, not a crash."""
    body = (text or "").strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        # The client prints one JSON document; keep the last line that looks like one.
        for line in reversed(body.splitlines()):
            line = line.strip()
            if line.startswith("{") or line.startswith("["):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
    return None


def _tail(text: str, limit: int = 400) -> str:
    """Last few hundred characters of a failed command's output, one line."""
    collapsed = " ".join((text or "").split())
    return collapsed[-limit:]


def candidate_argv(cand: dict[str, Any]) -> list[str]:
    """Map one frontier node onto the WebBridge client's `op site arg` surface.

    A child that names a concrete object (`owner/name`, `topic:` for a repo search) is a
    direct lookup; a bare tag or author is a search. That keeps `kind=repo|model` usable both
    for seeds that name a repo and for children minted from a topic or a technique tag.
    """
    kind = str(cand["kind"])
    site = str(cand["site"])
    value = str(cand["value"])
    if kind == "note":
        return ["ask", site, value]
    if kind == "repo":
        if value.startswith("topic:"):
            return ["search", "github", value.split(":", 1)[1]]
        return ["extract", "github", value]
    if kind == "model":
        return ["extract", "hf", value] if "/" in value else ["search", "hf", value]
    return ["search", site, value]


def _summary_line(site: str, entry: dict[str, Any]) -> str:
    """A one-paragraph summary from whichever fields the site supplied."""
    if site == "github":
        description = str(entry.get("description") or "").strip()
        stats = [f"{entry.get('stars', 0)} stars"]
        if entry.get("language"):
            stats.append(str(entry["language"]))
        if entry.get("forks"):
            stats.append(f"{entry['forks']} forks")
        if entry.get("license"):
            stats.append(str(entry["license"]))
        return " · ".join([description, ", ".join(stats)]).strip(" ·")
    if site == "hf":
        bits = [
            str(entry.get("pipeline") or "model"),
            f"{entry.get('downloads', 0)} downloads",
            f"{entry.get('likes', 0)} likes",
        ]
        if entry.get("library"):
            bits.append(str(entry["library"]))
        if entry.get("params"):
            bits.append(f"{int(entry['params']):,} params")
        description = str(entry.get("description") or "").strip()
        return " · ".join([description, ", ".join(bits)]).strip(" ·")
    return strip_html(str(entry.get("description") or entry.get("summary") or ""))


def _normalise_entry(site: str, entry: Any) -> dict[str, Any] | None:
    """One site-shaped mapping → the fields the state machine stores and scores."""
    if not isinstance(entry, dict):
        return None
    if site == "habr":
        url = entry.get("guid") or entry.get("link") or entry.get("url")
        if not url:
            return None
        category = entry.get("category")
        tags = [str(category)] if category else []
        return {
            "url": str(url),
            "site": site,
            "title": str(entry.get("title") or "").strip(),
            "author": (str(entry.get("creator")).strip() if entry.get("creator") else None),
            "published": str(entry.get("pubDate")) if entry.get("pubDate") else None,
            "tags": tags,
            "summary": _summary_line(site, entry),
            "stars": 0,
            "downloads": 0,
            "source": entry,
        }
    if site == "github":
        repo = entry.get("repo") or entry.get("full_name")
        url = entry.get("url") or (f"https://github.com/{repo}" if repo else None)
        if not url:
            return None
        topics = entry.get("topics") or []
        return {
            "url": str(url),
            "site": site,
            "title": str(repo or entry.get("title") or url).strip(),
            "author": (str(repo).split("/")[0] if repo and "/" in str(repo) else None),
            "published": str(entry.get("pushed")) if entry.get("pushed") else None,
            "tags": [str(topic) for topic in topics] if isinstance(topics, list) else [],
            "summary": _summary_line(site, entry),
            "stars": int(entry.get("stars") or 0) if str(entry.get("stars") or 0).isdigit()
            else 0,
            "downloads": 0,
            "source": entry,
        }
    if site == "hf":
        model_id = entry.get("id") or entry.get("modelId")
        if not model_id:
            return None
        tags = entry.get("tags") or []
        return {
            "url": str(entry.get("url") or f"https://huggingface.co/{model_id}"),
            "site": site,
            "title": str(model_id),
            "author": (str(model_id).split("/")[0] if "/" in str(model_id) else None),
            "published": (str(entry.get("created") or entry.get("modified") or "") or None),
            "tags": [str(tag) for tag in tags] if isinstance(tags, list) else [],
            "summary": _summary_line(site, entry),
            "stars": 0,
            "downloads": int(entry.get("downloads") or 0)
            if str(entry.get("downloads") or 0).isdigit()
            else 0,
            "source": entry,
        }
    return None


def items_from_payload(
    payload: Any, cand: dict[str, Any], cfg: MaterialCfg
) -> list[dict[str, Any]]:
    """Turn one CLI payload into normalised items; a `note` candidate yields one answer item."""
    site = str(cand["site"])
    if cand["kind"] == "note":
        text = str(payload.get("text") or "") if isinstance(payload, dict) else ""
        prompt = str(cand["value"])
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        return [
            {
                "url": f"ask://{site}/{digest}",
                "site": site,
                "title": prompt[:120],
                "author": None,
                "published": None,
                "tags": [],
                "summary": text,
                "stars": 0,
                "downloads": 0,
                "source": payload if isinstance(payload, dict) else {"text": text},
            }
        ]
    if not isinstance(payload, dict):
        return []
    raw_items = payload.get("items")
    if isinstance(raw_items, list):
        out = []
        for entry in raw_items:
            normalised = _normalise_entry(site, entry)
            if normalised:
                out.append(normalised)
        return out
    data = payload.get("data")
    if isinstance(data, dict):
        normalised = _normalise_entry(site, data)
        return [normalised] if normalised else []
    return []


def _render_entry(site: str, entry: dict[str, Any]) -> str:
    """The "body" of a github/hf item: a rendered key: value block, since there is no article."""
    order = {
        "github": ("repo", "url", "description", "stars", "forks", "language", "open_issues",
                   "pushed", "license", "topics", "archived"),
        "hf": ("id", "url", "pipeline", "library", "downloads", "likes", "params", "license",
               "tags", "architecture", "gated"),
    }.get(site, tuple(entry))
    lines = []
    for key in order:
        value = entry.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {value}")
    for key in sorted(entry):
        if key in order or entry[key] in (None, "", []):
            continue
        value = entry[key]
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


# --- files ---------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    """Write through a sibling temp file so a crash never leaves half a note or pack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


# --- seed ----------------------------------------------------------------


def seed(
    cfg: MaterialCfg, *, dry_run: bool = False, argv: list[str] | None = None
) -> dict[str, Any]:
    """Insert the configured seeds into the frontier once; re-running adds nothing."""
    if dry_run and not cfg.db_path.exists():
        return {
            "dry_run": True,
            "config": str(cfg.path),
            "db": str(cfg.db_path),
            "seeds": len(cfg.seeds),
            "added": len(cfg.seeds),
            "existing": 0,
            "frontier": len(cfg.seeds),
        }
    with State(cfg.db_path) as state:
        added: list[str] = []
        existing = 0
        for item in cfg.seeds:
            known = state.conn.execute(
                "SELECT 1 FROM candidates WHERE kind = ? AND value = ? AND site = ?",
                (item.kind, item.value, item.site),
            ).fetchone()
            if known:
                existing += 1
                continue
            if not dry_run:
                state.add_candidate(
                    item.kind, item.value, item.site, reason=item.reason or "seed"
                )
            added.append(f"{item.kind}/{item.site}: {item.value}")
        return {
            "dry_run": dry_run,
            "config": str(cfg.path),
            "db": str(cfg.db_path),
            "seeds": len(cfg.seeds),
            "added": len(added),
            "existing": existing,
            "would_add": added if dry_run else [],
            "frontier": state.frontier() + (len(added) if dry_run else 0),
        }


# --- fetch ---------------------------------------------------------------


def _fetch_body(
    cfg: MaterialCfg, entry: dict[str, Any], url: str, day: str
) -> tuple[str, Any, str | None]:
    """Article body plus the raw payload to keep. Returns `(body, raw, error)`."""
    site = str(entry["site"])
    if site == "habr":
        rc, stdout, stderr = run_webbridge(
            ["extract", "habr", url], cfg.limits.fetch_timeout_s
        )
        payload = _loads(stdout)
        if not isinstance(payload, dict):
            return "", None, f"extract failed (rc={rc}): {_tail(stderr or stdout)}"
        body = str(payload.get("text") or "")
        return body, payload, None
    if site in ("github", "hf"):
        return _render_entry(site, entry["source"]), entry["source"], None
    return str(entry.get("summary") or ""), entry.get("source"), None


def _store_raw(cfg: MaterialCfg, day: str, entry: dict[str, Any], body: str) -> tuple[str, str]:
    """Write `<site>-<slug>.json` (exact CLI payload) and `.txt` (normalised text).

    Returns `(text_path, json_path)`: the state row points at the text, because that is what
    the pack and a human read; the JSON sits beside it as the provenance record.
    """
    slug = slugify(str(entry["url"]))
    folder = cfg.date_dir(day)
    json_path = folder / f"{entry['site']}-{slug}.json"
    text_path = folder / f"{entry['site']}-{slug}.txt"
    _write_json(json_path, entry["source"])
    text = normalize_text(body)
    _atomic_write(text_path, text + "\n" if text else "")
    return cfg.display(text_path), cfg.display(json_path)


def _new_item_row(entry: dict[str, Any], score: float, status: str) -> dict[str, Any]:
    """The column set for one freshly discovered item."""
    tags = entry.get("tags") or []
    return {
        "url": entry["url"],
        "site": entry["site"],
        "title": entry.get("title"),
        "author": entry.get("author"),
        "published": entry.get("published"),
        "tags": json.dumps(tags, ensure_ascii=False),
        "score": score,
        "status": status,
        "content_sha": entry.get("content_sha"),
        "summary": (entry.get("summary") or "")[:4000],
        "raw_path": None,
        "digest_path": None,
        "parent": entry.get("parent"),
        "error": None,
        "first_seen_at": _now(),
        "fetched_at": None,
        "distilled_at": None,
        "memorized_at": None,
    }


def _children(entry: dict[str, Any], score: float, cfg: MaterialCfg) -> list[SeedCfg]:
    """Children worth enqueuing for one fetched item: tags, author, topics, technique tags."""
    site = str(entry["site"])
    allowed = set(cfg.expansion.per_site.get(site, []))
    allowed &= set(cfg.expansion.allow_kinds)
    if not allowed:
        return []
    tags = entry.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    out: list[SeedCfg] = []
    if site == "habr":
        if "tag" in allowed:
            for tag in tags:
                text = str(tag).strip()
                if len(text) >= 3:
                    out.append(SeedCfg(kind="tag", value=text, site="habr",
                                        reason=f"tag:{text}"))
        if "author" in allowed and score >= cfg.scoring.expand_min + 0.15:
            author = str(entry.get("author") or "").strip()
            if author:
                out.append(SeedCfg(kind="author", value=author, site="habr",
                                    reason=f"author:{author}"))
    elif site == "github":
        if "repo" in allowed:
            for topic in tags:
                text = str(topic).strip()
                if len(text) >= 3:
                    out.append(SeedCfg(kind="repo", value=f"topic:{text}", site="github",
                                        reason=f"topic:{text}"))
    elif site == "hf" and "model" in allowed:
        for tag in tags:
            text = _fold(str(tag)).strip()
            if text in NOTE_TAG_ALLOW:
                out.append(SeedCfg(kind="model", value=text, site="hf", reason=f"tag:{text}"))
    return out


# --- run -----------------------------------------------------------------


def _empty_counters() -> dict[str, Any]:
    return {
        "queries": 0,
        "items": 0,
        "kept": 0,
        "kept_by_site": {},
        "duplicates": 0,
        "fetched": 0,
        "errors": 0,
        "children": 0,
        "frontier_full": 0,
        "seen_before": 0,
    }


def run(
    cfg: MaterialCfg,
    *,
    limit: int | None = None,
    day: str | None = None,
    dry_run: bool = False,
    no_expand: bool = False,
    max_items: int | None = None,
    sites: list[str] | None = None,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Collect → score → dedupe → fetch → expand, one row per transition."""
    day = day or today()
    budget = cfg.limits.items_per_run if max_items is None else max(0, int(max_items))
    budget = min(budget, cfg.limits.items_per_run)
    query_budget = cfg.limits.queries_per_run if limit is None else max(0, int(limit))
    counters = _empty_counters()

    if dry_run and not cfg.db_path.is_file():
        # A plan is a read: with no state yet there is no frontier and nothing to create.
        return {
            "dry_run": True,
            "date": day,
            "config": str(cfg.path),
            "queries_planned": 0,
            "items_budget": budget,
            "frontier": 0,
            "plan": [],
            "counters": counters,
        }

    with State(cfg.db_path) as state:
        pending = state.candidates(limit=query_budget, sites=sites)
        if dry_run:
            plan = [
                {"candidate": f"{cand['kind']}/{cand['site']}", "value": cand["value"],
                 "argv": candidate_argv(cand)}
                for cand in pending
            ]
            return {
                "dry_run": True,
                "date": day,
                "config": str(cfg.path),
                "queries_planned": len(plan),
                "items_budget": budget,
                "frontier": state.frontier(),
                "plan": plan,
                "counters": counters,
            }

        run_id = state.start_run(argv if argv is not None else sys.argv[1:])
        started = time.monotonic()
        per_site: dict[str, int] = {}
        try:
            for cand in pending:
                counters["queries"] += 1
                if not str(cand["value"]).strip():
                    # A seed with an empty value can never be executed; drop it, do not fail it.
                    state.set_candidate(int(cand["id"]), "dropped", "empty value")
                    counters["dropped"] = counters.get("dropped", 0) + 1
                    continue
                argv_c = candidate_argv(cand)
                timeout = (
                    cfg.limits.ask_timeout_s if cand["kind"] == "note"
                    else cfg.limits.fetch_timeout_s
                )
                rc, stdout, stderr = run_webbridge(argv_c, timeout)
                payload = _loads(stdout)
                # The tunnelstack client exits 1 when a search matches nothing — its contract
                # is "print no text, fail" — but the JSON on stdout is still a valid answer.
                # A query that legitimately found zero results must not fail the whole run:
                # it is one frontier node that has been explored and is now empty. Only an
                # unparsable/absent stdout is a real failure (bad argv, dead site, no JSON).
                if payload is None:
                    counters["errors"] += 1
                    state.set_candidate(
                        int(cand["id"]), "error",
                        f"rc={rc}: {_tail(stderr or stdout)}",
                    )
                    _log(f"{cand['kind']}/{cand['site']} failed: rc={rc}")
                    continue
                if rc != 0:
                    _log(
                        f"{cand['kind']}/{cand['site']}: rc={rc} with valid JSON "
                        "(empty result, not a failure)"
                    )
                state.set_candidate(int(cand["id"]), "done")
                for entry in items_from_payload(payload, cand, cfg):
                    url = canonical_url(str(entry["url"]))
                    entry["url"] = url
                    entry["parent"] = cand.get("parent")
                    if not url:
                        continue
                    if state.item(url):
                        counters["seen_before"] += 1
                        continue
                    if counters["items"] >= budget:
                        continue
                    site = str(entry["site"])
                    if per_site.get(site, 0) >= cfg.limits.items_per_site:
                        continue
                    score = score_item(entry, cfg)
                    entry["content_sha"] = content_sha(
                        str(entry.get("title") or ""), str(entry.get("summary") or "")
                    )
                    owner = state.sha_owner(entry["content_sha"], url)
                    counters["items"] += 1
                    per_site[site] = per_site.get(site, 0) + 1
                    kept_by_site = counters["kept_by_site"]
                    if owner:
                        counters["duplicates"] += 1
                        state.add_item(_new_item_row(entry, score, "duplicate"))
                        state.update_item(url, error=f"same content as {owner}")
                        continue
                    if score < cfg.scoring.keep_min:
                        state.add_item(_new_item_row(entry, score, "scored_out"))
                        continue
                    counters["kept"] += 1
                    kept_by_site[site] = kept_by_site.get(site, 0) + 1
                    state.add_item(_new_item_row(entry, score, "seen"))
                    body, raw, error = _fetch_body(cfg, entry, url, day)
                    text = normalize_text(body)
                    if error or not text:
                        counters["errors"] += 1
                        state.update_item(
                            url, status="error",
                            error=error or "empty body after normalisation",
                        )
                        continue
                    if raw is not None:
                        entry["source"] = raw
                    raw_path, _json_path = _store_raw(cfg, day, entry, body)
                    state.update_item(
                        url, status="fetched", raw_path=raw_path, fetched_at=_now()
                    )
                    counters["fetched"] += 1
                    if no_expand or score < cfg.scoring.expand_min:
                        continue
                    for child in _children(entry, score, cfg)[: cfg.limits.children_per_item]:
                        if state.frontier() >= cfg.limits.max_frontier:
                            counters["frontier_full"] += 1
                            break
                        if state.add_candidate(
                            child.kind, child.value, child.site,
                            parent=url, reason=child.reason, priority=score,
                        ):
                            counters["children"] += 1
        finally:
            counters["seconds"] = round(time.monotonic() - started, 1)
            state.finish_run(run_id, counters)
        counters["frontier"] = state.frontier()
        counters["date"] = day
        counters["ok"] = counters["errors"] == 0
        return counters


# --- pack and distill ----------------------------------------------------

PACK_RULES = """\
## Rules

- Write only under `docs/material/notes/{day}/`. Do not touch any other directory.
- No network access. Do not install anything, do not run docker.
- Keep numbers, model names, versions and links verbatim; do not round, translate or
  transliterate identifiers.
- One file per item, named after the slug in the item heading below.
"""

PACK_SCHEMA = """\
## Output schema

`notes/{day}/<slug>.md`, YAML frontmatter followed by the body:

```yaml
---
title: <the item title>
site: <habr|github|hf|google-ai|deepseek|qwen>
url: <the item url>
tags: [<tags, comma separated>]
published: <date or null>
score: <the score from this pack>
experiment: <one line, or none>
---
```

Body sections, in this order and with these headings:

- `## What it is`
- `## Why it matters to machinum-sft`
- `## Techniques / numbers`
- `## Links`
- `## Experiment` — one concrete, cheap, runnable idea, or the single word `none`.
"""


def _pack_text(cfg: MaterialCfg, day: str, items: list[dict[str, Any]]) -> str:
    """The whole distill task pack: brief, rules, schema, one section per item."""
    lines = [
        f"# Material pack — {day}",
        "",
        f"Product: {cfg.product.name}",
        "",
        cfg.product.brief.strip(),
        "",
        PACK_RULES.format(day=day),
        PACK_SCHEMA.format(day=day),
        "---",
        "",
    ]
    for index, item in enumerate(items, 1):
        slug = slugify(str(item["url"]))
        tags = item.get("tags") or "[]"
        try:
            tags = ", ".join(json.loads(tags)) if isinstance(tags, str) else ", ".join(tags)
        except (ValueError, TypeError):
            tags = str(tags)
        text_path = cfg.resolve(item.get("raw_path"))
        body = ""
        if text_path and text_path.is_file():
            body = text_path.read_text(encoding="utf-8", errors="replace")
        lines.extend(
            [
                f"## {index}. {item.get('title') or item['url']}",
                "",
                f"- slug: {slug}",
                f"- url: {item['url']}",
                f"- site: {item['site']}",
                f"- title: {item.get('title') or '-'}",
                f"- author: {item.get('author') or '-'}",
                f"- published: {item.get('published') or '-'}",
                f"- tags: {tags or '-'}",
                f"- score: {item.get('score', 0)}",
                f"- raw: {item.get('raw_path') or '-'}",
                "",
                "```text",
                body[:PACK_TEXT_LIMIT],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _last_json_line(text: str) -> dict[str, Any] | None:
    """`at` prints an nvm banner before its summary; take the last line that starts with `{`."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _parse_note(path: Path) -> tuple[dict[str, Any], str]:
    """Split a note into (frontmatter, body); a note without frontmatter still yields a body."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    header = text[3:end].strip("\n")
    body = text[end + 4 :].lstrip("\n")
    try:
        parsed = yaml.safe_load(header)
    except yaml.YAMLError:
        parsed = None
    return (parsed if isinstance(parsed, dict) else {}), body


def _note_tags(front: dict[str, Any]) -> list[str]:
    raw = front.get("tags")
    if isinstance(raw, list):
        return [str(tag).strip() for tag in raw if str(tag).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in raw.strip("[]").split(",") if part.strip()]
    return []


def _experiment(front: dict[str, Any]) -> str:
    raw = front.get("experiment")
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if text.lower() in ("", "none", "null", "-") else text


def _append_experiments(cfg: MaterialCfg, day: str, entries: list[tuple[str, str]]) -> int:
    """Append one section per experiment, deduped by slug; the file only ever grows."""
    path = cfg.material_dir / "experiments.md"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not existing:
        existing = (
            "# Experiment proposals\n\n"
            "One section per distilled note that proposed a concrete, cheap experiment.\n"
            "Append-only: re-running distill never duplicates a slug.\n\n"
        )
    written = 0
    chunk = []
    for slug, text in entries:
        if not text or f"`{slug}`" in existing:
            continue
        chunk.append(f"## {day} · `{slug}`\n\n{text}\n")
        written += 1
    if chunk:
        _atomic_write(path, existing.rstrip("\n") + "\n\n" + "\n".join(chunk))
    return written


def distill(
    cfg: MaterialCfg,
    *,
    day: str | None = None,
    agent: str | None = None,
    timeout_ms: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Pack every fetched item of the day, hand the pack to `at`, then mark what came back."""
    day = day or today()
    agent = agent or cfg.distill.agent
    timeout = int(timeout_ms if timeout_ms is not None else cfg.distill.timeout_ms)
    pack_path = cfg.pack_dir / f"{day}.md"
    command = [
        str(at_bin()),
        "-a",
        agent,
        "-f",
        str(pack_path),
        "--wait",
        "--json-summary",
        "--timeout",
        str(timeout),
        "--scope",
        "docs/material/**",
    ]
    empty = {
        "dry_run": dry_run,
        "date": day,
        "ok": True,
        "items": 0,
        "packed": 0,
        "distilled": 0,
        "missing_notes": 0,
        "experiments": 0,
        "pack": cfg.display(pack_path),
        "note": "nothing to pack",
    }
    if dry_run and not cfg.db_path.is_file():
        return empty
    with State(cfg.db_path) as state:
        items = state.items_for_date(day, statuses=("fetched", "packed"))
        if not items:
            return empty
        pack_text = _pack_text(cfg, day, items)
        if dry_run:
            return {
                "dry_run": True,
                "date": day,
                "ok": True,
                "items": len(items),
                "packed": len(items),
                "distilled": 0,
                "missing_notes": 0,
                "experiments": 0,
                "pack": cfg.display(pack_path),
                "command": command,
                "cwd": str(cfg.repo),
                "chars": len(pack_text),
                "items_list": [
                    {"slug": slugify(str(item["url"])), "url": item["url"],
                     "score": item.get("score", 0)}
                    for item in items
                ],
            }
        _atomic_write(pack_path, pack_text)
        for item in items:
            state.update_item(item["url"], status="packed")
        _log(f"packed {len(items)} item(s) into {cfg.display(pack_path)}")

        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                cwd=str(cfg.repo),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return {
                "dry_run": False,
                "date": day,
                "ok": False,
                "items": len(items),
                "packed": len(items),
                "distilled": 0,
                "missing_notes": len(items),
                "experiments": 0,
                "pack": cfg.display(pack_path),
                "command": command,
                "error": f"cannot run {command[0]}: {exc}",
                "seconds": round(time.monotonic() - started, 1),
            }
        seconds = round(time.monotonic() - started, 1)
        summary = _last_json_line(proc.stdout or "")
        status = str((summary or {}).get("status", "")).lower()
        ok = proc.returncode == 0 and summary is not None and status not in ("failed", "error")
        if not ok:
            return {
                "dry_run": False,
                "date": day,
                "ok": False,
                "items": len(items),
                "packed": len(items),
                "distilled": 0,
                "missing_notes": len(items),
                "experiments": 0,
                "pack": cfg.display(pack_path),
                "command": command,
                "returncode": proc.returncode,
                "summary": summary,
                "error": _tail((proc.stderr or "") + " " + (proc.stdout or "")),
                "seconds": seconds,
            }

        distilled = 0
        missing = 0
        experiments: list[tuple[str, str]] = []
        for item in items:
            slug = slugify(str(item["url"]))
            note = cfg.notes_dir / day / f"{slug}.md"
            if not note.is_file():
                missing += 1
                continue
            front, _body = _parse_note(note)
            state.update_item(
                item["url"],
                status="distilled",
                distilled_at=_now(),
                digest_path=cfg.display(note),
            )
            distilled += 1
            proposal = _experiment(front)
            if proposal:
                experiments.append((slug, proposal))
        written = _append_experiments(cfg, day, experiments)
        return {
            "dry_run": False,
            "date": day,
            "ok": missing == 0,
            "items": len(items),
            "packed": len(items),
            "distilled": distilled,
            "missing_notes": missing,
            "experiments": written,
            "pack": cfg.display(pack_path),
            "command": command,
            "returncode": proc.returncode,
            "summary": summary,
            "seconds": seconds,
        }


# --- memorize ------------------------------------------------------------


def _note_body(front: dict[str, Any], body: str) -> str:
    """The stored markdown, with the H1 the wiki page needs."""
    if re.search(r"(?m)^#\s+\S", body):
        return body if body.endswith("\n") else body + "\n"
    title = str(front.get("title") or "").strip() or "Material note"
    text = f"# {title}\n\n{body.lstrip()}"
    return text if text.endswith("\n") else text + "\n"


def memory_payload(
    cfg: MaterialCfg, day: str, item: dict[str, Any], *, workspace: str | None = None,
    project: str | None = None,
) -> dict[str, Any] | None:
    """Build the `admin/write-page` payload for one distilled item (None when the note is gone)."""
    note = cfg.resolve(item.get("digest_path"))
    if note is None or not note.is_file():
        return None
    slug = slugify(str(item["url"]))
    front, body = _parse_note(note)
    tags = _note_tags(front) or []
    for extra in ("material", str(item["site"])):
        if extra not in tags:
            tags.append(extra)
    return {
        "workspace": workspace or cfg.memory.workspace,
        "project": project or cfg.memory.project,
        "path": f"{cfg.memory.path_prefix}/{day}-{slug}.md",
        "kind": "note",
        "tags": tags,
        "body": _note_body(front, body),
        "_url": str(item["url"]),
        "_slug": slug,
        "_title": str(front.get("title") or item.get("title") or slug),
        "_score": item.get("score", 0),
        "_experiment": _experiment(front),
    }


def _strip_private(payload: dict[str, Any]) -> dict[str, Any]:
    """The wire payload: the `_`-prefixed fields are ours, not ai-memory's."""
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def write_index(cfg: MaterialCfg) -> str:
    """Rewrite `INDEX.md` from the DB: date, title, site, score, experiment flag."""
    with State(cfg.db_path) as state:
        rows = state.memorized()
    lines = [
        "# Material index",
        "",
        "Every note this machine memorised into the ai-memory wiki. Generated from the state",
        "database; re-run `msft material memorize` to refresh it.",
        "",
        "| date | title | site | score | experiment |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        match = _DATE_DIR_RE.search(str(row.get("raw_path") or ""))
        day = match.group(1) if match else "-"
        note = cfg.resolve(row.get("digest_path"))
        experiment = ""
        if note and note.is_file():
            front, _body = _parse_note(note)
            experiment = "yes" if _experiment(front) else ""
        title = str(row.get("title") or row["url"]).replace("|", "\\|")
        lines.append(
            f"| {day} | {title} | {row['site']} | {row.get('score', 0)} | {experiment} |"
        )
    text = "\n".join(lines) + "\n"
    _atomic_write(cfg.material_dir / "INDEX.md", text)
    return cfg.display(cfg.material_dir / "INDEX.md")


def memorize(
    cfg: MaterialCfg,
    *,
    day: str | None = None,
    dry_run: bool = False,
    workspace: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """POST every distilled note of the day to ai-memory, then refresh `INDEX.md`."""
    day = day or today()
    if dry_run and not cfg.db_path.is_file():
        # Nothing has been collected yet, so there is nothing to preview and nothing to create.
        return {
            "dry_run": True,
            "date": day,
            "items": 0,
            "memorized": 0,
            "errors": 0,
            "skipped": 0,
            "url": f"{memory_url()}/admin/write-page",
            "payloads": [],
        }
    with State(cfg.db_path) as state:
        items = state.items_for_date(day, statuses=("distilled",))
        payloads: list[dict[str, Any]] = []
        skipped = 0
        for item in items:
            payload = memory_payload(cfg, day, item, workspace=workspace, project=project)
            if payload is None:
                skipped += 1
                continue
            payloads.append(payload)
        if dry_run:
            return {
                "dry_run": True,
                "date": day,
                "items": len(payloads),
                "memorized": 0,
                "errors": 0,
                "skipped": skipped,
                "url": f"{memory_url()}/admin/write-page",
                "payloads": [_strip_private(payload) for payload in payloads],
            }
        if not payloads:
            return {
                "dry_run": False,
                "date": day,
                "items": 0,
                "memorized": 0,
                "errors": 0,
                "skipped": skipped,
                "note": "nothing to memorize",
            }
        token = memory_token()
        if not token:
            raise MsftError(
                "no memory token: set MSFT_MEMORY_TOKEN or put AI_MEMORY_AUTH_TOKEN=... in "
                "deploy/memory/.env"
            )
        url = f"{memory_url()}/admin/write-page"
        memorized = 0
        errors = 0
        for payload in payloads:
            status, body = post_page(url, token, _strip_private(payload))
            if status in (200, 201):
                memorized += 1
                state.update_item(
                    payload["_url"],
                    status="memorized",
                    memorized_at=_now(),
                    digest_path=cfg.display(cfg.notes_dir / day / f"{payload['_slug']}.md"),
                )
                continue
            errors += 1
            state.update_item(
                payload["_url"],
                status="error",
                error=f"memory write failed: HTTP {status}: {_tail(body, 200)}",
            )
        index = write_index(cfg) if memorized else None
        return {
            "dry_run": False,
            "date": day,
            "items": len(payloads),
            "memorized": memorized,
            "errors": errors,
            "skipped": skipped,
            "index": index,
            "url": url,
            "ok": errors == 0,
        }


def _url_for_slug(state: State, day: str, slug: str) -> str | None:
    """Map a note slug back to its item URL (the slug is derived from the URL)."""
    for item in state.items_for_date(day):
        if slugify(str(item["url"])) == slug:
            return str(item["url"])
    return None


# --- report and status ---------------------------------------------------


def _raw_files(cfg: MaterialCfg, day: str) -> int:
    folder = cfg.date_dir(day)
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() and not path.name.startswith("."))


def _experiment_count(cfg: MaterialCfg, day: str) -> int:
    path = cfg.material_dir / "experiments.md"
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(rf"(?m)^##\s+{re.escape(day)}\s", text))


def report(cfg: MaterialCfg, *, day: str | None = None) -> dict[str, Any]:
    """One line (or one object) about what the loop did on `--date` (default: today)."""
    day = day or today()
    empty = {
        "date": day,
        "queries": 0,
        "items": 0,
        "kept": 0,
        "kept_by_site": {},
        "duplicates": 0,
        "fetched": 0,
        "packed": 0,
        "distilled": 0,
        "memorized": 0,
        "experiments": 0,
        "frontier": 0,
        "raw": _raw_files(cfg, day),
        "errors": 0,
        "seconds": 0.0,
    }
    if not cfg.db_path.is_file():
        return empty
    with State(cfg.db_path) as state:
        runs = state.runs_for_date(day)
        items = state.items_for_date(day)
        frontier = state.frontier()
    queries = items_count = kept = duplicates = fetched = errors = 0
    kept_by_site: dict[str, int] = {}
    seconds = 0.0
    for row in runs:
        counters = json.loads(row.get("counters") or "{}")
        queries += int(counters.get("queries", 0))
        items_count += int(counters.get("items", 0))
        kept += int(counters.get("kept", 0))
        duplicates += int(counters.get("duplicates", 0))
        fetched += int(counters.get("fetched", 0))
        errors += int(counters.get("errors", 0))
        for site, count in (counters.get("kept_by_site") or {}).items():
            kept_by_site[site] = kept_by_site.get(site, 0) + int(count)
        seconds += float(counters.get("seconds", 0) or 0)
    by_status: dict[str, int] = {}
    for item in items:
        by_status[str(item["status"])] = by_status.get(str(item["status"]), 0) + 1
    return {
        "date": day,
        "queries": queries,
        "items": items_count,
        "kept": kept,
        "kept_by_site": kept_by_site,
        "duplicates": duplicates,
        "fetched": fetched,
        "packed": by_status.get("packed", 0),
        "distilled": by_status.get("distilled", 0) + by_status.get("memorized", 0),
        "memorized": by_status.get("memorized", 0),
        "experiments": _experiment_count(cfg, day),
        "frontier": frontier,
        "raw": _raw_files(cfg, day),
        "errors": errors,
        "seconds": round(seconds, 1),
    }


def format_report(payload: dict[str, Any]) -> str:
    """The one-line summary an operator actually reads."""
    by_site = payload.get("kept_by_site") or {}
    breakdown = ""
    if by_site:
        ordered = sorted(by_site.items(), key=lambda pair: (-pair[1], pair[0]))
        breakdown = " (" + ", ".join(f"{site} {count}" for site, count in ordered) + ")"
    return (
        f"material {payload['date']}: {payload.get('queries', 0)} queries → "
        f"{payload.get('items', 0)} items, {payload.get('kept', 0)} kept{breakdown}, "
        f"{payload.get('duplicates', 0)} duplicates, {payload.get('distilled', 0)} distilled, "
        f"{payload.get('memorized', 0)} memorized, {payload.get('experiments', 0)} experiment "
        f"· frontier {payload.get('frontier', 0)} · raw {payload.get('raw', 0)} files · "
        f"{payload.get('seconds', 0):g} s"
    )


def status(cfg: MaterialCfg) -> dict[str, Any]:
    """Offline health snapshot: counts, dates, last run and the paths in use (never the token)."""
    exists = cfg.db_path.is_file()
    payload: dict[str, Any] = {
        "config": str(cfg.path),
        "config_exists": cfg.path.is_file(),
        "material_dir": str(cfg.material_dir),
        "db": str(cfg.db_path),
        "db_exists": exists,
        "repo": str(cfg.repo),
        "tunnelstack": str(tunnelstack_root()),
        "tunnelstack_client": (tunnelstack_root() / "src" / "webbridge.py").is_file(),
        "at_bin": str(at_bin()),
        "at_exists": at_bin().is_file(),
        "memory_url": memory_url(),
        "memory_token": memory_token() is not None,
        "dates": [],
        "candidates": {},
        "candidates_total": 0,
        "frontier": 0,
        "items": {},
        "items_total": 0,
        "last_run": None,
    }
    if not exists:
        return payload
    with State(cfg.db_path) as state:
        by_status = state.conn.execute(
            "SELECT status, COUNT(*) AS n FROM candidates GROUP BY status ORDER BY status"
        )
        payload["candidates"] = {str(row["status"]): int(row["n"]) for row in by_status}
        payload["candidates_total"] = state.candidates_count()
        payload["frontier"] = state.frontier()
        payload["items"] = state.items_by_status()
        payload["items_total"] = state.items_total()
        payload["dates"] = state.dates()
        last = state.last_run()
        if last:
            payload["last_run"] = {
                "id": last["id"],
                "started_at": last["started_at"],
                "finished_at": last["finished_at"],
                "counters": json.loads(last.get("counters") or "{}"),
            }
    return payload


def format_status(payload: dict[str, Any]) -> str:
    """Indented snapshot: frontier, items per status, last run."""
    lines = [
        f"material  {payload['config']}",
        f"dir       {payload['material_dir']}",
        f"db        {payload['db']}" + ("" if payload["db_exists"] else "  (not created yet)"),
        f"frontier  {payload['frontier']} of {payload.get('candidates_total', 0)} candidate(s)",
        f"sites     webbridge {'ok' if payload['tunnelstack_client'] else 'MISSING'}  "
        f"at {'ok' if payload['at_exists'] else 'MISSING'}",
    ]
    if payload["candidates"]:
        lines.append(
            "candidates "
            + " ".join(f"{key}={value}" for key, value in sorted(payload["candidates"].items()))
        )
    if payload["items"]:
        lines.append(
            "items     " + " ".join(f"{key}={value}" for key, value in payload["items"].items())
        )
    if payload["dates"]:
        lines.append(f"dates     {', '.join(payload['dates'][:7])}")
    last = payload.get("last_run")
    if last:
        counters = last.get("counters") or {}
        lines.append(
            f"last run  #{last['id']} {last['started_at']} "
            f"queries={counters.get('queries', 0)} items={counters.get('items', 0)} "
            f"errors={counters.get('errors', 0)}"
        )
    lines.append(
        f"memory    {payload['memory_url']} token "
        f"{'set' if payload['memory_token'] else 'MISSING'}"
    )
    return "\n".join(lines)


# --- one-line (or one-plan) renderers for the other commands -------------


def format_seed(payload: dict[str, Any]) -> str:
    """What `seed` added, and what the frontier now holds."""
    verb = "would add" if payload.get("dry_run") else "added"
    lines = [
        f"seed {payload['config']}: {payload.get('seeds', 0)} seed(s), "
        f"{verb} {payload.get('added', 0)}, already present {payload.get('existing', 0)}, "
        f"frontier {payload.get('frontier', 0)}"
    ]
    for entry in payload.get("would_add") or []:
        lines.append(f"  + {entry}")
    return "\n".join(lines)


def format_run(payload: dict[str, Any]) -> str:
    """Either the plan (dry run) or the counters one real run left behind."""
    if payload.get("dry_run"):
        lines = [
            f"run plan {payload['date']}: {payload.get('queries_planned', 0)} query(s), "
            f"item budget {payload.get('items_budget', 0)}, frontier {payload.get('frontier', 0)}"
        ]
        for step in payload.get("plan") or []:
            lines.append(f"  {step['candidate']:<14} {' '.join(step['argv'])}")
        if not payload.get("plan"):
            lines.append("  (nothing new in the frontier — run `msft material seed` first)")
        return "\n".join(lines)
    by_site = payload.get("kept_by_site") or {}
    breakdown = ""
    if by_site:
        ordered = sorted(by_site.items(), key=lambda pair: (-pair[1], pair[0]))
        breakdown = " (" + ", ".join(f"{site} {count}" for site, count in ordered) + ")"
    return (
        f"run {payload.get('date')}: {payload.get('queries', 0)} queries, "
        f"{payload.get('items', 0)} items, {payload.get('kept', 0)} kept{breakdown}, "
        f"{payload.get('duplicates', 0)} duplicates, {payload.get('fetched', 0)} fetched, "
        f"{payload.get('children', 0)} children, {payload.get('errors', 0)} error(s) "
        f"· frontier {payload.get('frontier', 0)} · {payload.get('seconds', 0):g} s"
    )


def format_distill(payload: dict[str, Any]) -> str:
    """Pack size, the exact `at` command, and how many notes came back."""
    if payload.get("dry_run"):
        lines = [
            f"distill plan {payload['date']}: {payload.get('items', 0)} item(s) → "
            f"{payload.get('pack')} ({payload.get('chars', 0)} chars)"
        ]
        for entry in payload.get("items_list") or []:
            lines.append(f"  {entry['slug']}  score={entry['score']}  {entry['url']}")
        if payload.get("items"):
            lines.append("  would run: " + " ".join(payload.get("command") or []))
        return "\n".join(lines)
    if payload.get("items") == 0:
        return f"distill {payload['date']}: nothing to pack"
    head = (
        f"distill {payload['date']}: {'ok' if payload.get('ok') else 'FAILED'} · "
        f"{payload.get('packed', 0)} packed, {payload.get('distilled', 0)} distilled, "
        f"{payload.get('missing_notes', 0)} missing note(s), "
        f"{payload.get('experiments', 0)} experiment(s) · {payload.get('seconds', 0):g} s"
    )
    if payload.get("error"):
        head += f"\n  error: {payload['error']}"
    return head


def format_memorize(payload: dict[str, Any]) -> str:
    """Dry run prints the exact payloads; a real run prints the counters."""
    if payload.get("dry_run"):
        lines = [
            f"memorize plan {payload['date']}: {payload.get('items', 0)} page(s) → "
            f"{payload.get('url')}",
            f"  workspace={payload['payloads'][0]['workspace']} "
            f"project={payload['payloads'][0]['project']}" if payload.get("payloads") else "",
        ]
        for entry in payload.get("payloads") or []:
            lines.append(json.dumps(entry, ensure_ascii=False, indent=2))
        return "\n".join(line for line in lines if line)
    if payload.get("items") == 0:
        return f"memorize {payload['date']}: nothing to memorize"
    return (
        f"memorize {payload['date']}: {payload.get('memorized', 0)}/{payload.get('items', 0)} "
        f"page(s) written, {payload.get('errors', 0)} error(s)"
        + (f" · index {payload['index']}" if payload.get("index") else "")
    )
