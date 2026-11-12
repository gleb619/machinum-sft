"""`msft experiments` — a durable, costed queue between the daily notes and the GPU box.

The research loop already writes one prose `experiment:` per note, and nothing ran it: the
proposals piled up in `docs/material/experiments.md` while the box sat idle. This module is
the missing arrow — note → validated job → one run — with two rules that make it safe:

- **A recipe is the unit of permission.** The distiller may only name a recipe that
  `configs/experiments/catalog.yaml` already costs, with `est_minutes` taken from a measured
  run. A model writing a note therefore cannot invent a 12B job: `sync` drops the name and
  counts it instead of queueing it.
- **One GPU job at a time, cheapest first.** Two trainings on one 16 GB card is an OOM, not
  a speed-up, so `tick` starts nothing while the box has a job in `running/`, and it takes
  the lowest `est_minutes` queued row before any dearer one. A weekday gets one job; the
  weekend drains the backlog.

State is SQLite at `<material dir>/.state/experiments.db` — the row, not the box's log, is
what says where a proposal stands. Every remote call sits behind a seam (`_signal`,
`_job_state`, `_box_busy`, `_sleep`), so the tests run offline: no ssh, no docker, no net.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from msft import material, worker
from msft.config import MsftError, redact

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_NAME = "configs/experiments/catalog.yaml"

STATUSES = ("queued", "running", "done", "failed", "skipped")

# A weekday is one job and 90 minutes: enough for the cheapest recipe and for a smoke train,
# and short enough that a cron slot can wait for the whole thing. Saturday and Sunday have no
# count limit (`None` = drain) and six hours, which is what draining a backlog costs.
WEEKDAY_BUDGET = 1
WEEKDAY_SECONDS = 5_400
WEEKEND_SECONDS = 21_600

POLL_SECONDS = 20.0
MIN_TIME_BUDGET_S = 60
TAIL_LINES = 40
HYPOTHESIS_LIMIT = 300
ERROR_LIMIT = 600
# Exit code for "the box could not be reached", kept distinct from 2 (config) so the cron
# script can report it and exit 0: an unreachable box is not a failed experiment.
EXIT_UNREACHABLE = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  origin TEXT NOT NULL,
  recipe TEXT NOT NULL,
  kind TEXT NOT NULL,
  config TEXT,
  args TEXT NOT NULL DEFAULT '[]',
  est_minutes REAL NOT NULL DEFAULT 0,
  hypothesis TEXT,
  status TEXT NOT NULL DEFAULT 'queued',
  remote_id TEXT,
  started_at TEXT,
  finished_at TEXT,
  rc INTEGER,
  log_path TEXT,
  error TEXT,
  metrics TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, est_minutes, created_at);
"""

_TOP_KEYS = ("recipes", "fallback")
_RECIPE_KEYS = ("kind", "config", "args", "est_minutes", "what")
_RELATIVE_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
_SECTION_RE = re.compile(r"(?m)^##\s+\S+\s+·\s+`([^`]+)`\s*$")


class CatalogError(MsftError):
    """`configs/experiments/catalog.yaml` is missing, unreadable or incomplete (exit 2)."""


# --- environment ----------------------------------------------------------


def _env(name: str, default: str) -> str:
    """Environment override, else the default. Read at call time so tests can patch it."""
    return os.environ.get(name) or default


def catalog_path() -> Path:
    """`MSFT_EXPERIMENTS_CATALOG`, else `<repo>/configs/experiments/catalog.yaml`."""
    return Path(_env("MSFT_EXPERIMENTS_CATALOG", str(REPO_ROOT / DEFAULT_CATALOG_NAME)))


def material_dir() -> Path:
    """The material tree, from the same `MSFT_MATERIAL_DIR` the research loop honours."""
    return material.material_dir()


def db_path() -> Path:
    """The queue's SQLite file, next to `material.db` and inside the git-ignored tree."""
    return material_dir() / ".state" / "experiments.db"


def log_dir() -> Path:
    """Where the tail of each finished job is kept locally, so the report survives the box."""
    return material_dir() / ".state" / "logs"


def today() -> str:
    """Local calendar date: the default `--date` and the input to the budget rule."""
    return material.today()


def _now() -> str:
    """UTC timestamp, seconds precision — the shape `material.db` writes, so days compare."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    """Progress on stderr; stdout stays one object or one line."""
    print(redact(message), file=sys.stderr)


def _trim(text: str | None, limit: int = ERROR_LIMIT) -> str:
    """Keep the end of a remote message, where the reason usually is."""
    value = (text or "").strip()
    return value if len(value) <= limit else "..." + value[-limit:]


# --- catalog --------------------------------------------------------------


def _as_str(raw: dict[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


def _as_args(raw: dict[str, Any], where: str) -> list[str]:
    value = raw.get("args")
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CatalogError(f"{where}: 'args' must be a list of strings (one CLI token each)")
    return [str(item) for item in value]


@dataclass
class Recipe:
    """One KNOWN-COST run: the only thing a proposal is allowed to ask for."""

    name: str
    kind: str
    config: str
    est_minutes: float
    what: str
    args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "config": self.config,
            "args": list(self.args),
            "est_minutes": self.est_minutes,
            "what": self.what,
        }


@dataclass
class ExperimentsCfg:
    """The catalog plus the paths a queue needs; `repo` is where configs must be committed."""

    path: Path
    recipes: dict[str, Recipe]
    material_dir: Path
    repo: Path
    #: The cheapest known-good recipe used on a day whose notes proposed nothing runnable.
    fallback: str | None = None

    @property
    def db_path(self) -> Path:
        return self.material_dir / ".state" / "experiments.db"

    @property
    def notes_dir(self) -> Path:
        return self.material_dir / "notes"

    @property
    def log_dir(self) -> Path:
        return self.material_dir / ".state" / "logs"

    def recipe(self, name: str) -> Recipe:
        """Look one up, naming the alternatives — an unknown recipe is a user error."""
        chosen = self.recipes.get(name)
        if chosen is None:
            known = ", ".join(sorted(self.recipes)) or "none"
            raise CatalogError(f"unknown recipe {name!r} in {self.path}; pick one of: {known}")
        return chosen

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog": str(self.path),
            "repo": str(self.repo),
            "material_dir": str(self.material_dir),
            "recipes": len(self.recipes),
        }


def load_catalog(
    path: str | Path | None = None, *, repo: str | Path | None = None
) -> ExperimentsCfg:
    """Load and validate the recipe catalog; any hole is a CatalogError (exit 2).

    `repo` exists so a test can point at a temporary git checkout: the guard that refuses an
    uncommitted config has to run against the tree the box would actually clone.
    """
    p = Path(path) if path else catalog_path()
    if not p.is_file():
        raise CatalogError(
            f"experiments catalog not found: {p} "
            "(write configs/experiments/catalog.yaml or set MSFT_EXPERIMENTS_CATALOG)"
        )
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"cannot read {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"{p} must contain a mapping at the top level")
    unknown = sorted(set(data) - set(_TOP_KEYS))
    if unknown:
        raise CatalogError(f"{p}: unknown top-level key(s): {', '.join(unknown)}")
    raw_recipes = data.get("recipes")
    if not isinstance(raw_recipes, dict) or not raw_recipes:
        raise CatalogError(f"{p}: 'recipes' must be a non-empty mapping of name -> recipe")

    recipes: dict[str, Recipe] = {}
    for name, raw in raw_recipes.items():
        where = f"{p}: recipe {name!r}"
        if not isinstance(raw, dict):
            raise CatalogError(f"{where} must be a mapping")
        unknown = sorted(set(raw) - set(_RECIPE_KEYS))
        if unknown:
            raise CatalogError(f"{where}: unknown key(s): {', '.join(unknown)}")
        kind = _as_str(raw, "kind", where)
        if kind not in worker.JOB_KINDS:
            raise CatalogError(
                f"{where}: kind {kind!r} is not a worker job kind "
                f"({', '.join(worker.JOB_KINDS)})"
            )
        config = _as_str(raw, "config", where)
        if not _RELATIVE_RE.match(config) or not config.endswith(".yaml"):
            raise CatalogError(
                f"{where}: config {config!r} must be a repo-relative path ending in .yaml"
            )
        est = raw.get("est_minutes")
        if isinstance(est, bool) or not isinstance(est, (int, float)) or float(est) <= 0:
            raise CatalogError(f"{where}: 'est_minutes' must be a positive number")
        recipes[str(name)] = Recipe(
            name=str(name),
            kind=kind,
            config=config,
            est_minutes=round(float(est), 2),
            what=_as_str(raw, "what", where),
            args=_as_args(raw, where),
        )
    fallback = data.get("fallback")
    if fallback is not None:
        if not isinstance(fallback, str) or fallback not in recipes:
            raise CatalogError(
                f"{p}: 'fallback' must name one of the recipes above, got {fallback!r}"
            )
    return ExperimentsCfg(
        path=p,
        recipes=recipes,
        material_dir=material_dir(),
        repo=Path(repo) if repo is not None else REPO_ROOT,
        fallback=str(fallback) if fallback is not None else None,
    )


def catalog(cfg: ExperimentsCfg) -> dict[str, Any]:
    """The menu as data: what a proposal may name, and what each name costs."""
    ordered = sorted(cfg.recipes.values(), key=lambda item: (item.est_minutes, item.name))
    return {
        "catalog": str(cfg.path),
        "count": len(ordered),
        "recipes": [item.to_dict() for item in ordered],
    }


def format_catalog(payload: dict[str, Any]) -> str:
    """One line per recipe, cheapest first — the order the scheduler uses."""
    lines = [f"experiment menu — {payload.get('catalog')} ({payload.get('count', 0)} recipes)"]
    for item in payload.get("recipes") or []:
        args = " ".join(item.get("args") or []) or "-"
        lines.append(
            f"  {item.get('est_minutes', 0):>5} min  {item.get('name', ''):<20}"
            f"{item.get('kind', ''):<9}{item.get('config', '')}  args: {args}"
        )
        lines.append(f"             {item.get('what', '')}")
    return "\n".join(lines)


# --- the budget rule ------------------------------------------------------


def schedule_budget(day: str | None = None) -> dict[str, Any]:
    """The weekday/weekend policy in one place: Mon-Fri one job, Sat-Sun the whole backlog.

    It lives in Python, not in the shell wrapper, because it is the rule that decides how
    much GPU time a day may spend — one implementation, one unit test, and `--budget auto`
    cannot drift from the script that calls it.
    """
    chosen = day or today()
    weekday = date_cls.fromisoformat(chosen).isoweekday() <= 5
    return {
        "date": chosen,
        "weekday": weekday,
        "budget": WEEKDAY_BUDGET if weekday else None,
        "time_budget_s": WEEKDAY_SECONDS if weekday else WEEKEND_SECONDS,
    }


def resolve_budget(
    spec: str | int | None = "auto", day: str | None = None, time_budget_s: int | None = None
) -> tuple[int | None, int]:
    """Turn `--budget 1|0|auto` into (jobs, seconds). `0` means drain, like `None`."""
    policy = schedule_budget(day)
    text = "auto" if spec is None else str(spec).strip().lower()
    if text == "auto":
        budget = policy["budget"]
    elif text in ("0", "drain", "all"):
        budget = None
    else:
        try:
            budget = int(text)
        except ValueError as exc:
            raise MsftError(f"invalid --budget {spec!r}: use a number, 0 (drain) or auto") from exc
        if budget < 0:
            raise MsftError(f"invalid --budget {spec!r}: use a number, 0 (drain) or auto")
    seconds = int(time_budget_s if time_budget_s is not None else policy["time_budget_s"])
    if seconds < MIN_TIME_BUDGET_S:
        raise MsftError(f"invalid --time-budget {seconds}: use at least {MIN_TIME_BUDGET_S} s")
    return budget, seconds


# --- sqlite ---------------------------------------------------------------


@contextlib.contextmanager
def _writer(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the queue for writing, autocommit, and always close.

    Autocommit matters: a tick can watch a job for hours, and holding an implicit write
    transaction open would lock `report` and `list` out of the database for that whole time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
    finally:
        conn.close()


@contextlib.contextmanager
def _reader(path: Path) -> Iterator[sqlite3.Connection | None]:
    """Read-only connection, or None when the queue has never been written (a plan is fine)."""
    if not path.is_file():
        yield None
        return
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _job_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    """A jobs row as a plain dict, with `args` and `metrics` decoded from JSON."""
    job = dict(row)
    for key, fallback in (("args", []), ("metrics", {})):
        value = job.get(key)
        try:
            job[key] = json.loads(value) if value else fallback
        except (TypeError, ValueError):
            job[key] = fallback
    return job


def _next_id(conn: sqlite3.Connection, day: str) -> str:
    """`exp-<YYYYMMDD>-<n>`: readable in a log line, unique without a uuid."""
    prefix = f"exp-{day.replace('-', '')}-"
    highest = 0
    for row in conn.execute("SELECT id FROM jobs WHERE id LIKE ?", (f"{prefix}%",)):
        suffix = str(row["id"]).rsplit("-", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"{prefix}{highest + 1}"


def _pairs(conn: sqlite3.Connection | None) -> set[tuple[str, str]]:
    """The (origin, recipe) pairs already queued — what `sync` dedupes against."""
    if conn is None:
        return set()
    return {
        (str(row["origin"]), str(row["recipe"]))
        for row in conn.execute("SELECT origin, recipe FROM jobs")
    }


def _queued(conn: sqlite3.Connection | None, limit: int | None) -> list[dict[str, Any]]:
    """Queued rows, cheapest first; ties by age, then id, so a plan is reproducible."""
    if conn is None:
        return []
    sql = (
        "SELECT * FROM jobs WHERE status = 'queued' "
        "ORDER BY est_minutes ASC, created_at ASC, id ASC"
    )
    rows = [_job_row(row) for row in conn.execute(sql)]
    return rows if limit is None else rows[:limit]


def _count(conn: sqlite3.Connection | None, status: str, day: str | None = None) -> int:
    """Rows in one status, optionally only those created on `day`."""
    if conn is None:
        return 0
    if day is None:
        row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (status,)).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = ? AND substr(created_at, 1, 10) = ?",
            (status, day),
        ).fetchone()
    return int(row["n"]) if row else 0


# --- the note side --------------------------------------------------------


def _prose_hypotheses(path: Path) -> dict[str, str]:
    """`experiments.md` sections by note slug: the human sentence, used as a fallback.

    The block's own `hypothesis` wins; this exists so a note that names a recipe but writes
    no hypothesis still carries the sentence a reader will recognise.
    """
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    found: dict[str, str] = {}
    for match in _SECTION_RE.finditer(text):
        start = match.end()
        following = text.find("\n## ", start)
        body = text[start : following if following >= 0 else len(text)].strip()
        first = body.splitlines()[0].strip() if body else ""
        if first:
            found[match.group(1)] = first[:HYPOTHESIS_LIMIT]
    return found


def _hypothesis(
    block: dict[str, str], prose: dict[str, str], slug: str, front: dict[str, Any]
) -> str:
    """Block first, then `experiments.md`, then the note's prose `experiment:` line."""
    for candidate in (block.get("hypothesis"), prose.get(slug), material._experiment(front)):
        text = " ".join(str(candidate or "").split())
        if text:
            return text[:HYPOTHESIS_LIMIT]
    return ""


def sync(cfg: ExperimentsCfg, day: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Queue one job per note that names a catalog recipe; counters, never exceptions.

    The note's `experiment_job.recipe` is the only structured input, and the catalog decides
    whether it exists. A missing block, an unknown recipe or a malformed block is a counter
    and a stderr line: one bad frontmatter must not stop a day's queue. `est_minutes` comes
    from the catalog, never from the note — the note's number is a guess, the catalog's is a
    measurement.
    """
    chosen = day or today()
    counters: dict[str, Any] = {
        "date": chosen,
        "scanned": 0,
        "queued": 0,
        "duplicates": 0,
        "unknown_recipe": 0,
        "dry_run": bool(dry_run),
    }
    folder = cfg.notes_dir / chosen
    notes = sorted(folder.glob("*.md")) if folder.is_dir() else []
    if not notes:
        _log(f"sync: no notes under {cfg.notes_dir / chosen}")
    prose = _prose_hypotheses(cfg.material_dir / "experiments.md")
    pending: list[dict[str, Any]] = []
    for note in notes:
        counters["scanned"] += 1
        front, _body = material._parse_note(note)  # one frontmatter reader, not a second copy
        block = material.parse_experiment_job(front)
        if block is None:
            continue
        slug = note.stem
        recipe = cfg.recipes.get(block.get("recipe", ""))
        if recipe is None:
            counters["unknown_recipe"] += 1
            _log(
                f"sync: {slug}: recipe {block.get('recipe')!r} is not in {cfg.path.name}; "
                "dropped (the catalog is the allow-list)"
            )
            continue
        pending.append(
            {
                "origin": f"note:{slug}",
                "recipe": recipe.name,
                "kind": recipe.kind,
                "config": recipe.config,
                "args": list(recipe.args),
                "est_minutes": recipe.est_minutes,
                "hypothesis": _hypothesis(block, prose, slug, front),
            }
        )

    if not pending and cfg.fallback:
        # A day whose notes proposed nothing runnable must not become a silent no-op — that is
        # the failure this queue exists to end. The fallback is the cheapest known-good recipe,
        # queued once per day (`fallback:<day>`), and its origin says plainly that no note asked
        # for it, so the report cannot be mistaken for a research finding.
        recipe = cfg.recipes[cfg.fallback]
        pending.append(
            {
                "origin": f"fallback:{chosen}",
                "recipe": recipe.name,
                "kind": recipe.kind,
                "config": recipe.config,
                "args": list(recipe.args),
                "est_minutes": recipe.est_minutes,
                "hypothesis": "no note proposed a runnable recipe; the cheapest known-good run",
            }
        )
        counters["fallback"] = 1
        _log(f"sync: nothing runnable in {chosen}; falling back to {recipe.name}")

    if dry_run:
        seen = _pairs_readonly(cfg.db_path)
        for row in pending:
            key = (row["origin"], row["recipe"])
            if key in seen:
                counters["duplicates"] += 1
            else:
                seen.add(key)
                counters["queued"] += 1
        return counters

    with _writer(cfg.db_path) as conn:
        seen = _pairs(conn)
        for row in pending:
            key = (row["origin"], row["recipe"])
            if key in seen:
                counters["duplicates"] += 1
                continue
            seen.add(key)
            _insert(conn, row, chosen)
            counters["queued"] += 1
    return counters


def _pairs_readonly(path: Path) -> set[tuple[str, str]]:
    """`_pairs` for a run that must not write (a `--dry-run` sync creates no database)."""
    with _reader(path) as conn:
        if conn is None:
            return set()
        try:
            return _pairs(conn)
        except sqlite3.Error:
            return set()


def _insert(conn: sqlite3.Connection, row: dict[str, Any], day: str) -> str:
    """Insert one queued job and return its id."""
    ident = _next_id(conn, day)
    conn.execute(
        "INSERT INTO jobs (id, created_at, origin, recipe, kind, config, args, est_minutes, "
        "hypothesis, status, metrics) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', '{}')",
        (
            ident,
            _now(),
            row["origin"],
            row["recipe"],
            row["kind"],
            row["config"],
            json.dumps(row["args"]),
            float(row["est_minutes"]),
            row.get("hypothesis") or None,
        ),
    )
    return ident


def add_job(
    recipe: str,
    hypothesis: str | None = None,
    origin: str = "manual",
    *,
    cfg: ExperimentsCfg | None = None,
) -> dict[str, Any]:
    """Queue a recipe by hand — the operator's own experiment, costed from the catalog."""
    resolved = cfg or load_catalog()
    chosen = resolved.recipe(recipe)
    row = {
        "origin": origin,
        "recipe": chosen.name,
        "kind": chosen.kind,
        "config": chosen.config,
        "args": list(chosen.args),
        "est_minutes": chosen.est_minutes,
        "hypothesis": " ".join(str(hypothesis or "").split())[:HYPOTHESIS_LIMIT] or None,
    }
    with _writer(resolved.db_path) as conn:
        ident = _insert(conn, row, today())
        stored = conn.execute("SELECT * FROM jobs WHERE id = ?", (ident,)).fetchone()
    return _job_row(stored) if stored is not None else {**row, "id": ident}


def list_jobs(
    status: str | None = None,
    *,
    cfg: ExperimentsCfg | None = None,
    db: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Rows, newest first, optionally filtered by status; [] when the queue is empty."""
    if status is not None and status not in STATUSES:
        raise MsftError(f"unknown status {status!r}; pick one of {', '.join(STATUSES)}")
    path = db or (cfg.db_path if cfg is not None else db_path())
    sql = "SELECT * FROM jobs"
    params: list[Any] = []
    if status is not None:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _reader(path) as conn:
        if conn is None:
            return []
        return [_job_row(row) for row in conn.execute(sql, params)]


# --- the remote side (the seams) ------------------------------------------


def _config_guard(cfg: ExperimentsCfg, config_rel: str) -> str | None:
    """Why this job must not be signalled, or None. Cheaper than a failed job ten minutes in.

    The box runs a `git clone` of this repository, so a config that is untracked, modified or
    absent locally simply does not exist there: the worker starts, the stage cannot read its
    YAML, and the failure lands in `done/<id>.json` long after the queue said "running".
    Refusing here costs one `git` call and turns that into a skipped row with a reason.
    """
    if not config_rel:
        return "job has no config path; a catalog recipe always names one"
    path = (cfg.repo / config_rel).resolve()
    try:
        path.relative_to(cfg.repo.resolve())
    except ValueError:
        return f"config {config_rel} escapes the repository; the box clones only this repo"
    if not path.is_file():
        return f"config {config_rel} does not exist in {cfg.repo}"
    code, _out = _git(cfg.repo, ["rev-parse", "--is-inside-work-tree"])
    if code != 0:
        return (
            f"{cfg.repo} is not a git checkout, so nothing proves {config_rel} reaches the "
            "box: the box runs a git clone, so this job would fail ten minutes later"
        )
    code, _out = _git(cfg.repo, ["cat-file", "-e", f"HEAD:{config_rel}"])
    if code != 0:
        return (
            f"config {config_rel} is not committed: the box runs a git clone, so this job "
            "would fail ten minutes later. Commit AND push it, then sync."
        )
    code, out = _git(cfg.repo, ["status", "--porcelain", "--", config_rel])
    if code == 0 and out.strip():
        return (
            f"config {config_rel} has uncommitted changes: the box runs a git clone, so this "
            "job would run the old file and fail ten minutes later. Commit and push it first."
        )
    return None


def _git(repo: Path, args: list[str]) -> tuple[int, str]:
    """One short git call, never raising: a missing binary is "cannot prove", not a crash."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _signal(cfg: ExperimentsCfg, job: dict[str, Any]) -> dict[str, Any]:
    """Queue one job on the box. The seam the tests patch instead of opening an ssh session.

    Returns `{ok, unreachable, error, returncode}` and never raises: a dropped link is a state
    of the world, not an exception a scheduler should turn into a traceback.
    """
    from msft import deploy  # imported here so `catalog`/`list` need no worker config

    try:
        spec = deploy.load_spec()
        # Fast-forward the box FIRST: the worker executes the container's `python -m msft`
        # against the box's clone, so a job queued against a stale clone quietly produces
        # results from old code. `pull --ff-only` failing means the box cannot reach the
        # commit we are about to ask for (unpushed local commit, or a dirty checkout there),
        # and that is a reason to refuse, not a warning: a ten-minute run on the wrong code
        # costs more than a skipped row.
        pulled = deploy.pull(spec, execute=True)
        if not pulled.get("ok"):
            rc = pulled.get("returncode")
            detail = pulled.get("stderr") or pulled.get("stdout") or "git pull failed on the box"
            return {
                "ok": False,
                "unreachable": rc in (127, 255),
                "error": _trim(f"box could not fast-forward (push the commit first): {detail}"),
                "returncode": rc,
            }
        result = deploy.signal(
            spec,
            str(job["kind"]),
            config=str(job["config"]),
            job_id=str(job["id"]),
            extra_args=list(job["args"]),
            execute=True,
        )
    except MsftError as exc:
        return {"ok": False, "unreachable": True, "error": _trim(str(exc)), "returncode": None}
    ok = bool(result.get("ok"))
    rc = result.get("returncode")
    detail = result.get("stderr") or result.get("stdout") or "signal failed"
    return {
        "ok": ok,
        "unreachable": rc in (127, 255),
        "error": None if ok else _trim(detail),
        "returncode": rc,
    }


def _job_state(cfg: ExperimentsCfg, remote_id: str) -> dict[str, Any]:
    """One look at the box's `done/<id>.json`: {state: pending|done|failed, returncode, tail}.

    Reuses `remote.logs_job`, which already knows that a missing sentinel is a transport
    failure and not a green run; a second copy of that ssh command would drift from it.
    """
    from msft import remote as remote_mod

    try:
        host = remote_mod.remote_host()
    except MsftError as exc:
        return {"state": "unknown", "transport": True, "error": _trim(str(exc))}
    return remote_mod.logs_job(host, remote_id, lines=TAIL_LINES)


def _box_busy(cfg: ExperimentsCfg) -> bool:
    """True when the box already has a job in `running/`; an unreachable box answers False.

    This is the same check `remote run --tmux` refuses on, deliberately reused: one GPU job
    at a time is a property of the card, and re-implementing the probe would let the two
    paths disagree about what "busy" means.
    """
    from msft import remote as remote_mod

    try:
        host = remote_mod.remote_host()
        repo = remote_mod.remote_repo_dir()
        return bool(remote_mod._running_jobs(host, repo))
    except MsftError:
        return False


def _sleep(seconds: float) -> None:
    """The polling sleep, a seam so a test does not spend a minute waiting."""
    time.sleep(seconds)


def _store_log(cfg: ExperimentsCfg, ident: str, tail: str) -> str | None:
    """Keep the job's log tail locally; the box's `.worker/logs` is not in git either."""
    if not tail.strip():
        return None
    path = cfg.log_dir / f"{ident}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tail.rstrip("\n") + "\n", encoding="utf-8")
    return str(path)


def _finish(
    conn: sqlite3.Connection,
    ident: str,
    status: str,
    rc: int | None = None,
    *,
    error: str | None = None,
    log_path: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Move a row to a terminal status (or back to the queue), always with `finished_at`."""
    conn.execute(
        "UPDATE jobs SET status = ?, rc = ?, finished_at = ?, error = ?, "
        "log_path = COALESCE(?, log_path), metrics = ? WHERE id = ?",
        (
            status,
            rc,
            _now(),
            _trim(error) if error else None,
            log_path,
            json.dumps(metrics or {}),
            ident,
        ),
    )


def _watch(cfg: ExperimentsCfg, job: dict[str, Any], deadline: float) -> dict[str, Any]:
    """Poll `done/<id>.json` until the job has a verdict or the tick's clock runs out."""
    started = time.monotonic()
    while True:
        state = _job_state(cfg, str(job["remote_id"]))
        status = str(state.get("state") or "pending")
        if status in ("done", "failed"):
            elapsed = round(time.monotonic() - started, 1)
            ok = status == "done" and state.get("returncode") in (0, None)
            tail = str(state.get("tail") or "")
            return {
                "status": "done" if ok else "failed",
                "rc": state.get("returncode"),
                "error": None
                if ok
                else _trim(state.get("error") or f"remote exited {state.get('returncode')}"),
                "log_path": _store_log(cfg, str(job["id"]), tail),
                "metrics": {
                    "watched_s": elapsed,
                    "timed_out": bool(state.get("timed_out")),
                    "est_minutes": job.get("est_minutes"),
                    "actual_minutes": round(elapsed / 60.0, 1),
                },
            }
        if time.monotonic() >= deadline:
            return {
                "status": "failed",
                "rc": None,
                "error": (
                    "watch timed out: the tick's time budget ran out while the job was still "
                    f"running on the box (remote id {job['remote_id']}); it may still finish — "
                    f"check `msft remote logs --job {job['remote_id']}`"
                ),
                "log_path": None,
                "metrics": {
                    "watched_s": round(time.monotonic() - started, 1),
                    "timed_out": True,
                    "est_minutes": job.get("est_minutes"),
                },
            }
        _log(f"tick: {job['id']} ({job['recipe']}) still {status} on the box")
        _sleep(POLL_SECONDS)


def _reap(cfg: ExperimentsCfg, conn: sqlite3.Connection) -> int:
    """Book rows a dead tick left `running`; the process that returns must not leave one.

    A `running` row whose job already has a verdict on the box is simply booked. One with no
    verdict is marked failed with the reason, because the local watcher is gone and the queue
    must not stay locked behind a row nothing is watching.
    """
    reaped = 0
    for row in conn.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall():
        job = _job_row(row)
        remote_id = job.get("remote_id")
        status = "unknown"
        if remote_id:
            status = str(_job_state(cfg, str(remote_id)).get("state") or "unknown")
        if status in ("done", "failed"):
            _finish(
                conn,
                str(job["id"]),
                "done" if status == "done" else "failed",
                None if status == "done" else 1,
                error=None if status == "done" else "the box reported the job failed",
            )
        else:
            _finish(
                conn,
                str(job["id"]),
                "failed",
                None,
                error=(
                    "interrupted: a previous tick left this row running and the box has no "
                    f"verdict for {remote_id or 'it'}; nothing is watching it. Re-queue with "
                    "`msft experiments add <recipe>` if it still matters."
                ),
            )
        reaped += 1
        _log(f"tick: reaped {job['id']} (left running by an earlier process)")
    return reaped


# --- the scheduler --------------------------------------------------------


def tick(
    cfg: ExperimentsCfg,
    budget: int | None = 1,
    execute: bool = False,
    time_budget_s: int = 21_600,
    day: str | None = None,
) -> dict[str, Any]:
    """Start at most `budget` of the cheapest queued jobs on the box (`None` = drain).

    Without `execute` this is a plan: it reads the queue read-only, lists what would run in
    what order, and writes nothing — not even the database file. With `execute` it refuses to
    start anything while the box (or this queue) already has a running job, signals each job
    through the worker, polls its verdict, and records rc/status/log on the row.
    """
    chosen = day or today()
    limit = None if budget in (0, None) else max(1, int(budget))
    payload: dict[str, Any] = {
        "date": chosen,
        "budget": "drain" if limit is None else limit,
        "execute": bool(execute),
        "time_budget_s": int(time_budget_s),
        "selected": [],
        "started": [],
        "finished": [],
        "refused": [],
        "failed": 0,
        "reaped": 0,
        "box_busy": False,
        "box_unreachable": False,
        "note": None,
        "error": None,
    }
    if not execute:
        with _reader(cfg.db_path) as conn:
            payload["selected"] = [
                {
                    "id": job["id"],
                    "recipe": job["recipe"],
                    "kind": job["kind"],
                    "est_minutes": job["est_minutes"],
                }
                for job in _queued(conn, limit)
            ]
        payload["today"] = _summary(cfg, chosen)
        payload["queued"] = payload["today"]["queued"]
        return payload

    deadline = time.monotonic() + max(MIN_TIME_BUDGET_S, int(time_budget_s))
    with _writer(cfg.db_path) as conn:
        payload["reaped"] = _reap(cfg, conn)
        if _box_busy(cfg) or _count(conn, "running"):
            payload["box_busy"] = True
            payload["note"] = "box busy: a GPU job is already running, nothing started"
            payload["today"] = _summary(cfg, chosen)
            payload["queued"] = payload["today"]["queued"]
            return payload
        for job in _queued(conn, limit):
            if time.monotonic() >= deadline:
                payload["note"] = "time budget exhausted; nothing further started"
                break
            reason = _config_guard(cfg, str(job["config"] or ""))
            if reason:
                payload["refused"].append(job["id"])
                _finish(conn, str(job["id"]), "skipped", None, error=reason)
                _log(f"tick: refused {job['id']} ({job['recipe']}): {reason}")
                continue
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, remote_id = ?, "
                "error = NULL WHERE id = ?",
                (_now(), job["id"], job["id"]),
            )
            payload["started"].append(job["id"])
            payload["selected"].append(
                {
                    "id": job["id"],
                    "recipe": job["recipe"],
                    "kind": job["kind"],
                    "est_minutes": job["est_minutes"],
                }
            )
            job["remote_id"] = job["id"]
            _log(f"tick: starting {job['id']} ({job['recipe']}, {job['est_minutes']} min)")
            signal = _signal(cfg, job)
            if not signal.get("ok"):
                error = str(signal.get("error") or "signal failed")
                if signal.get("unreachable"):
                    # Nothing was queued; the row goes back so tomorrow's tick retries it.
                    conn.execute(
                        "UPDATE jobs SET status = 'queued', started_at = NULL, remote_id = "
                        "NULL, error = ? WHERE id = ?",
                        (_trim(error), job["id"]),
                    )
                    payload["started"].remove(job["id"])
                    payload["box_unreachable"] = True
                    payload["error"] = error
                    _log(f"tick: box unreachable ({error}); stopping")
                    break
                _finish(conn, str(job["id"]), "failed", None, error=error)
                payload["failed"] += 1
                payload["finished"].append({"id": job["id"], "status": "failed", "rc": None})
                continue
            outcome = _watch(cfg, job, deadline)
            _finish(
                conn,
                str(job["id"]),
                outcome["status"],
                outcome["rc"],
                error=outcome["error"],
                log_path=outcome["log_path"],
                metrics=outcome["metrics"],
            )
            payload["finished"].append(
                {"id": job["id"], "status": outcome["status"], "rc": outcome["rc"]}
            )
            if outcome["status"] == "failed":
                payload["failed"] += 1
            _log(
                f"tick: {job['id']} -> {outcome['status']} rc={outcome['rc']} "
                f"({(outcome['metrics'] or {}).get('actual_minutes', 0)} min)"
            )
        payload["today"] = _summary(cfg, chosen)
        payload["queued"] = payload["today"]["queued"]
    return payload


# --- reporting ------------------------------------------------------------


def _summary(cfg: ExperimentsCfg, day: str) -> dict[str, Any]:
    """Counts for the report line: the backlog as queued/running, today's work as done/failed.

    The split is deliberate. A row queued three days ago is still queue depth, not yesterday's
    news, while `done`/`failed` answer "what did today's cron actually do".
    """
    summary: dict[str, Any] = {
        "date": day,
        "queued": 0,
        "running": 0,
        "done": 0,
        "failed": 0,
        "skipped": 0,
        "last": None,
        "est_minutes": 0.0,
        "actual_minutes": 0.0,
    }
    with _reader(cfg.db_path) as conn:
        if conn is None:
            return summary
        summary["queued"] = _count(conn, "queued")
        summary["running"] = _count(conn, "running")
        for status in ("done", "failed", "skipped"):
            summary[status] = _count(conn, status, day)
        rows = conn.execute(
            "SELECT * FROM jobs WHERE started_at IS NOT NULL "
            "ORDER BY finished_at DESC, started_at DESC LIMIT 1"
        ).fetchall()
        for row in rows:
            job = _job_row(row)
            metrics = job.get("metrics") or {}
            summary["last"] = {
                "id": job["id"],
                "recipe": job["recipe"],
                "status": job["status"],
                "rc": job["rc"],
                "est_minutes": job["est_minutes"],
                "actual_minutes": metrics.get("actual_minutes"),
            }
        today_rows = conn.execute(
            "SELECT est_minutes, metrics FROM jobs "
            "WHERE substr(created_at, 1, 10) = ? AND status IN ('done', 'failed')",
            (day,),
        ).fetchall()
        summary["est_minutes"] = round(sum(float(row["est_minutes"] or 0) for row in today_rows), 1)
        summary["actual_minutes"] = round(
            sum(
                float((json.loads(row["metrics"] or "{}") or {}).get("actual_minutes") or 0)
                for row in today_rows
            ),
            1,
        )
    return summary


def report(cfg: ExperimentsCfg, day: str | None = None) -> dict[str, Any]:
    """What the queue did on `day`: backlog, today's verdicts, and est versus actual cost."""
    return _summary(cfg, day or today())


def _job_line(job: dict[str, Any]) -> str:
    """One job as a row: id, recipe, kind, cost, and where it came from."""
    return (
        f"{job.get('status', '?'):<7} {job.get('id', ''):<20} {job.get('recipe', ''):<20}"
        f"{job.get('kind', ''):<9}{job.get('est_minutes', 0):>5} min  {job.get('origin', '')}"
    )


def format_sync(payload: dict[str, Any]) -> str:
    """Counters of one sync, in the voice of `material report`."""
    suffix = " (dry run, nothing written)" if payload.get("dry_run") else ""
    return (
        f"experiments sync {payload.get('date')}: {payload.get('scanned', 0)} notes → "
        f"{payload.get('queued', 0)} queued, {payload.get('duplicates', 0)} duplicates, "
        f"{payload.get('unknown_recipe', 0)} unknown recipe{suffix}"
    )


def format_add(payload: dict[str, Any]) -> str:
    """One line after a manual add."""
    return (
        f"queued {payload.get('id')} {payload.get('recipe')} "
        f"({payload.get('est_minutes', 0)} min, {payload.get('kind')}) "
        f"— origin {payload.get('origin')}"
    )


def format_list(payload: list[dict[str, Any]]) -> str:
    """The queue as a table; an empty queue says so instead of printing a bare header."""
    if not payload:
        return "experiments queue: empty (msft experiments sync queues today's notes)"
    lines = [f"experiments queue ({len(payload)} rows):"]
    lines.extend("  " + _job_line(job) for job in payload)
    return "\n".join(lines)


def format_tick(payload: dict[str, Any]) -> str:
    """The one line a cron run shows: what ran, what it cost, what is left."""
    day = payload.get("date")
    today = payload.get("today") or {}
    tail = f"{today.get('done', 0)} done, {today.get('failed', 0)} failed today"
    if payload.get("box_unreachable"):
        return (
            f"experiments {day}: box unreachable: {payload.get('queued', 0)} queued, "
            f"nothing started · {tail}"
        )
    if payload.get("box_busy"):
        return (
            f"experiments {day}: box busy, nothing started · {payload.get('queued', 0)} queued"
            f" · {tail}"
        )
    if not payload.get("execute"):
        selected = payload.get("selected") or []
        left = f"{payload.get('queued', 0)} queued · {tail}"
        if not selected:
            return f"experiments plan {day}: nothing due · {left}"
        plan = ", ".join(
            f"{job['id']} {job['recipe']} ({job['est_minutes']} min)" for job in selected
        )
        return f"experiments plan {day}: would run {plan} · {left}"
    finished = payload.get("finished") or []
    left = f"{payload.get('queued', 0)} queued · {tail}"
    if not finished and not payload.get("refused"):
        reason = payload.get("note") or "nothing due"
        return f"experiments {day}: {reason} · {left}"
    parts = []
    for item in finished:
        est = next(
            (
                job.get("est_minutes")
                for job in payload.get("selected") or []
                if job.get("id") == item["id"]
            ),
            None,
        )
        cost = f" rc={item.get('rc')}" + (f" {est} min est" if est is not None else "")
        parts.append(f"{item['id']} {item['status']}{cost}")
    if payload.get("refused"):
        parts.append(f"{len(payload['refused'])} refused (config not committed)")
    return f"experiments {day}: " + " · ".join(parts) + f" · {left}"


def format_report(payload: dict[str, Any]) -> str:
    """The report line: backlog, today's verdicts, and the last job's cost against its estimate."""
    last = payload.get("last") or {}
    last_text = "last: none"
    if last:
        actual = last.get("actual_minutes")
        cost = f"{actual}/{last.get('est_minutes')} min" if actual is not None else "watch lost"
        last_text = (
            f"last {last.get('recipe')} {last.get('status')} rc={last.get('rc')} {cost}"
        )
    return (
        f"experiments {payload.get('date')}: {payload.get('queued', 0)} queued · "
        f"{payload.get('running', 0)} running · {payload.get('done', 0)} done · "
        f"{payload.get('failed', 0)} failed · {last_text} · est "
        f"{payload.get('est_minutes', 0):g} vs actual {payload.get('actual_minutes', 0):g} min"
    )
