"""Tests for `msft experiments` — the queue between the daily notes and the GPU box.

Offline: no ssh, no docker, no network. Every remote call sits behind a seam (`_signal`,
`_job_state`, `_box_busy`) that the fake box below patches, `_sleep` is a no-op, and each
test gets a real SQLite file under `tmp_path`. The commit guard is the one thing that needs
a git binary, so its tests skip rather than pretend.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from msft import experiments, worker
from msft.cli import main
from msft.config import MsftError
from msft.experiments import CatalogError

DAY = "2026-11-12"  # a Thursday
SATURDAY = "2026-11-14"
GIT = shutil.which("git")

CATALOG: dict[str, Any] = {
    "recipes": {
        "cheap": {
            "kind": "data",
            "config": "configs/cheap.yaml",
            "est_minutes": 0.1,
            "what": "rebuild the train/val split",
        },
        "mid": {
            "kind": "eval",
            "config": "configs/mid.yaml",
            "est_minutes": 0.6,
            "what": "replay the protocol gates",
        },
        "dear": {
            "kind": "train",
            "config": "configs/dear.yaml",
            "est_minutes": 3.0,
            "what": "one smoke epoch",
        },
    }
}


# --- fixtures and helpers -------------------------------------------------


def write_catalog(tmp_path: Path, data: dict[str, Any] | None = None) -> Path:
    """A catalog YAML under tmp_path."""
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(data if data is not None else CATALOG, sort_keys=False), encoding="utf-8"
    )
    return path


def write_note(
    folder: Path,
    slug: str,
    *,
    recipe: str | None = None,
    hypothesis: str | None = None,
    prose: str = "try something cheap",
) -> Path:
    """One note in the shape `distill` writes: frontmatter, optional experiment_job block."""
    block = ""
    if recipe is not None or hypothesis is not None:
        block = "experiment_job:\n"
        if recipe is not None:
            block += f"  recipe: {recipe}\n"
        if hypothesis is not None:
            block += f"  hypothesis: {hypothesis}\n"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}.md"
    path.write_text(
        "---\n"
        f"title: {slug}\n"
        "site: habr\n"
        f"url: https://habr.com/ru/articles/{slug}/\n"
        "score: 0.5\n"
        f"experiment: {prose}\n"
        f"{block}"
        "---\n\n## What it is\n\nprose.\n",
        encoding="utf-8",
    )
    return path


def git(repo: Path, *args: str) -> None:
    """Run one git command against `repo` with no user/system config in the way."""
    assert GIT is not None
    proc = subprocess.run(
        [GIT, "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-C", str(repo)]
        + list(args),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    assert proc.returncode == 0, proc.stderr


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway checkout with the catalog's configs committed — the thing the box clones."""
    if GIT is None:
        pytest.skip("git is required for the commit guard")
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    for name in ("cheap", "mid", "dear"):
        (root / "configs" / f"{name}.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "configs")
    return root


@pytest.fixture
def cfg(tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch) -> experiments.ExperimentsCfg:
    """A catalog whose material tree and repo both live under tmp_path."""
    monkeypatch.setenv("MSFT_MATERIAL_DIR", str(tmp_path / "material"))
    monkeypatch.setenv("MSFT_EXPERIMENTS_CATALOG", str(write_catalog(tmp_path)))
    return experiments.load_catalog(repo=repo)


class Box:
    """Stand-in for the GPU box: records signals, replays verdicts, opens no socket.

    A verdict that is a list is consumed one state at a time, which is how a test proves the
    scheduler polls until the job finishes instead of reading the first answer.
    """

    def __init__(
        self,
        verdicts: dict[str, Any] | None = None,
        *,
        busy: bool = False,
        unreachable: bool = False,
    ) -> None:
        self.signalled: list[tuple[str, str]] = []
        self.verdicts = verdicts or {}
        self.busy = busy
        self.unreachable = unreachable
        self.polls = 0

    def signal(self, _cfg: experiments.ExperimentsCfg, job: dict[str, Any]) -> dict[str, Any]:
        if self.unreachable:
            return {
                "ok": False,
                "unreachable": True,
                "error": "ssh exited 255",
                "returncode": 255,
            }
        self.signalled.append((str(job["id"]), str(job["recipe"])))
        return {"ok": True, "unreachable": False, "error": None, "returncode": 0}

    def state(self, _cfg: experiments.ExperimentsCfg, remote_id: str) -> dict[str, Any]:
        self.polls += 1
        verdict = self.verdicts.get(remote_id, {"state": "done", "returncode": 0, "tail": "ok"})
        if isinstance(verdict, list):
            verdict = verdict.pop(0) if len(verdict) > 1 else verdict[0]
        return dict(verdict)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> Box:
        monkeypatch.setattr(experiments, "_signal", self.signal)
        monkeypatch.setattr(experiments, "_job_state", self.state)
        monkeypatch.setattr(experiments, "_box_busy", lambda _cfg: self.busy)
        monkeypatch.setattr(experiments, "_sleep", lambda _seconds: None)
        return self


def queue(cfg: experiments.ExperimentsCfg, *recipes: str) -> list[dict[str, Any]]:
    """Add recipes by hand, cheapest-first order irrelevant — the scheduler sorts."""
    return [experiments.add_job(recipe, cfg=cfg) for recipe in recipes]


@pytest.fixture
def cli_cfg(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> experiments.ExperimentsCfg:
    """The CLI resolves the catalog from the repo, so point that lookup at the fixture."""
    original = experiments.load_catalog
    monkeypatch.setattr(
        experiments,
        "load_catalog",
        lambda path=None, **kwargs: cfg if path is None else original(path, **kwargs),
    )
    return cfg


# --- the catalog ----------------------------------------------------------


def test_shipped_catalog_is_valid() -> None:
    """Every shipped recipe names a real kind, a real config and a positive cost."""
    cfg = experiments.load_catalog()
    assert len(cfg.recipes) >= 5
    for recipe in cfg.recipes.values():
        assert recipe.kind in worker.JOB_KINDS
        assert (cfg.repo / recipe.config).is_file(), recipe.config
        assert recipe.est_minutes > 0
        assert recipe.what


def test_catalog_is_ordered_by_cost() -> None:
    """The menu sorts cheapest first, which is the order the scheduler uses."""
    cfg = experiments.load_catalog(repo=Path("/"))
    payload = experiments.catalog(cfg)
    costs = [item["est_minutes"] for item in payload["recipes"]]
    assert costs == sorted(costs)


def test_catalog_rejects_an_unknown_kind(tmp_path: Path) -> None:
    data = copy.deepcopy(CATALOG)
    data["recipes"]["cheap"]["kind"] = "arena"
    path = write_catalog(tmp_path, data)
    with pytest.raises(CatalogError, match="not a worker job kind"):
        experiments.load_catalog(path, repo=tmp_path)


def test_catalog_rejects_a_missing_field_and_a_bad_cost(tmp_path: Path) -> None:
    data = copy.deepcopy(CATALOG)
    del data["recipes"]["cheap"]["config"]
    with pytest.raises(CatalogError, match="config"):
        experiments.load_catalog(write_catalog(tmp_path, data), repo=tmp_path)
    data = copy.deepcopy(CATALOG)
    data["recipes"]["cheap"]["est_minutes"] = 0
    with pytest.raises(CatalogError, match="est_minutes"):
        experiments.load_catalog(write_catalog(tmp_path, data), repo=tmp_path)


def test_missing_catalog_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="not found"):
        experiments.load_catalog(tmp_path / "nope.yaml")


def test_unknown_recipe_names_the_alternatives(cfg: experiments.ExperimentsCfg) -> None:
    with pytest.raises(CatalogError, match="cheap, dear, mid"):
        cfg.recipe("invented")
    with pytest.raises(CatalogError, match="unknown recipe"):
        experiments.add_job("invented", cfg=cfg)


# --- sync: notes -> queue -------------------------------------------------


def test_sync_queues_once_then_dedupes(cfg: experiments.ExperimentsCfg) -> None:
    notes = cfg.notes_dir / DAY
    write_note(notes, "alpha", recipe="cheap", hypothesis="the split shrinks")
    write_note(notes, "beta", recipe="mid")
    write_note(notes, "gamma", prose="no structured block here")
    write_note(notes, "delta", recipe="invented")

    first = experiments.sync(cfg, day=DAY)
    assert first == {
        "date": DAY,
        "scanned": 4,
        "queued": 2,
        "duplicates": 0,
        "unknown_recipe": 1,
        "dry_run": False,
    }
    rows = experiments.list_jobs(cfg=cfg)
    assert {row["recipe"] for row in rows} == {"cheap", "mid"}
    origins = {row["origin"] for row in rows}
    assert origins == {"note:alpha", "note:beta"}

    second = experiments.sync(cfg, day=DAY)
    assert second["queued"] == 0
    assert second["duplicates"] == 2
    assert second["unknown_recipe"] == 1
    assert len(experiments.list_jobs(cfg=cfg)) == 2


def test_sync_takes_the_cost_from_the_catalog(cfg: experiments.ExperimentsCfg) -> None:
    """The note's own guess is ignored: `est_minutes` is what the scheduler sorts by."""
    write_note(cfg.notes_dir / DAY, "alpha", recipe="cheap", prose="45 minutes of training")
    experiments.sync(cfg, day=DAY)
    row = experiments.list_jobs(cfg=cfg)[0]
    assert row["est_minutes"] == pytest.approx(0.1)
    assert row["kind"] == "data"
    assert row["config"] == "configs/cheap.yaml"
    assert row["args"] == []
    assert row["status"] == "queued"
    assert row["hypothesis"]


def test_sync_uses_the_prose_section_as_a_fallback_hypothesis(
    cfg: experiments.ExperimentsCfg,
) -> None:
    write_note(cfg.notes_dir / DAY, "alpha", recipe="cheap")
    experiments.sync(cfg, day=DAY)
    row = experiments.list_jobs(cfg=cfg)[0]
    assert "try something cheap" in row["hypothesis"]


def test_sync_counts_a_malformed_block_instead_of_raising(
    cfg: experiments.ExperimentsCfg,
) -> None:
    folder = cfg.notes_dir / DAY
    write_note(folder, "alpha", recipe="cheap")
    (folder / "broken.md").write_text(
        "---\ntitle: broken\nexperiment_job: 7\n---\n\n## What it is\n\nx\n", encoding="utf-8"
    )
    counters = experiments.sync(cfg, day=DAY)
    assert counters["scanned"] == 2
    assert counters["queued"] == 1
    assert counters["unknown_recipe"] == 1


def test_sync_dry_run_writes_nothing(cfg: experiments.ExperimentsCfg) -> None:
    write_note(cfg.notes_dir / DAY, "alpha", recipe="cheap")
    counters = experiments.sync(cfg, day=DAY, dry_run=True)
    assert counters["queued"] == 1
    assert counters["dry_run"] is True
    assert not cfg.db_path.exists()
    assert experiments.list_jobs(cfg=cfg) == []


def test_sync_with_no_notes_is_empty_not_an_error(cfg: experiments.ExperimentsCfg) -> None:
    counters = experiments.sync(cfg, day="2026-01-01")
    assert counters["scanned"] == 0
    assert counters["queued"] == 0


# --- the budget rule ------------------------------------------------------


def test_schedule_budget_splits_weekday_from_weekend() -> None:
    weekday = experiments.schedule_budget(DAY)
    assert weekday == {
        "date": DAY,
        "weekday": True,
        "budget": 1,
        "time_budget_s": 5_400,
    }
    weekend = experiments.schedule_budget(SATURDAY)
    assert weekend["weekday"] is False
    assert weekend["budget"] is None  # drain
    assert weekend["time_budget_s"] == 21_600


def test_resolve_budget_understands_auto_zero_and_a_number() -> None:
    assert experiments.resolve_budget("auto", day=DAY) == (1, 5_400)
    assert experiments.resolve_budget("auto", day=SATURDAY) == (None, 21_600)
    assert experiments.resolve_budget("0", day=DAY) == (None, 5_400)
    assert experiments.resolve_budget("3", day=DAY) == (3, 5_400)
    assert experiments.resolve_budget("auto", day=DAY, time_budget_s=900) == (1, 900)
    with pytest.raises(MsftError, match="budget"):
        experiments.resolve_budget("many", day=DAY)
    with pytest.raises(MsftError, match="time-budget"):
        experiments.resolve_budget("1", day=DAY, time_budget_s=5)


# --- tick: the scheduler --------------------------------------------------


def test_tick_plan_is_read_only(cfg: experiments.ExperimentsCfg) -> None:
    queue(cfg, "dear", "cheap", "mid")
    plan = experiments.tick(cfg, budget=1, day=DAY)
    assert [item["recipe"] for item in plan["selected"]] == ["cheap"]
    assert plan["started"] == []
    assert plan["today"]["queued"] == 3
    assert {row["status"] for row in experiments.list_jobs(cfg=cfg)} == {"queued"}
    assert all(row["started_at"] is None for row in experiments.list_jobs(cfg=cfg))


def test_tick_starts_the_cheapest_job_first(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue(cfg, "dear", "cheap", "mid")
    box = Box().install(monkeypatch)

    result = experiments.tick(cfg, budget=1, execute=True, day=DAY)

    assert box.signalled == [(result["started"][0], "cheap")]
    assert result["failed"] == 0
    rows = {row["recipe"]: row for row in experiments.list_jobs(cfg=cfg)}
    assert rows["cheap"]["status"] == "done"
    assert rows["cheap"]["rc"] == 0
    assert rows["cheap"]["finished_at"]
    assert rows["cheap"]["metrics"]["actual_minutes"] is not None
    assert rows["mid"]["status"] == "queued"
    assert rows["dear"]["status"] == "queued"


def test_tick_drains_the_backlog_when_the_budget_is_none(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue(cfg, "dear", "mid", "cheap")
    box = Box().install(monkeypatch)

    result = experiments.tick(cfg, budget=None, execute=True, day=SATURDAY)

    assert [recipe for _id, recipe in box.signalled] == ["cheap", "mid", "dear"]
    assert all(row["status"] == "done" for row in experiments.list_jobs(cfg=cfg))
    assert result["queued"] == 0


def test_tick_polls_until_the_job_has_a_verdict(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = experiments.add_job("cheap", cfg=cfg)
    verdicts = {
        job["id"]: [
            {"state": "pending", "returncode": None, "tail": ""},
            {"state": "pending", "returncode": None, "tail": ""},
            {"state": "done", "returncode": 0, "tail": "rc=0"},
        ]
    }
    box = Box(verdicts).install(monkeypatch)

    result = experiments.tick(cfg, budget=1, execute=True, day=DAY)

    assert box.polls == 3
    assert result["finished"] == [{"id": job["id"], "status": "done", "rc": 0}]
    row = experiments.list_jobs(cfg=cfg)[0]
    assert row["log_path"] and Path(row["log_path"]).read_text(encoding="utf-8").strip() == "rc=0"


def test_tick_records_a_failure_and_never_leaves_a_row_running(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = experiments.add_job("cheap", cfg=cfg)
    verdicts = {job["id"]: {"state": "failed", "returncode": 1, "tail": "boom", "error": "rc=1"}}
    Box(verdicts).install(monkeypatch)

    result = experiments.tick(cfg, budget=1, execute=True, day=DAY)

    assert result["failed"] == 1
    row = experiments.list_jobs(cfg=cfg)[0]
    assert row["status"] == "failed"
    assert row["rc"] == 1
    assert "rc=1" in (row["error"] or "")
    assert not [item for item in experiments.list_jobs(cfg=cfg) if item["status"] == "running"]


def test_tick_refuses_a_config_that_is_not_committed(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The box runs a git clone, so an untracked config is a job that fails ten minutes later."""
    data = copy.deepcopy(CATALOG)
    data["recipes"]["draft"] = {
        "kind": "train",
        "config": "configs/draft.yaml",
        "est_minutes": 0.2,
        "what": "an uncommitted config",
    }
    draft = cfg.repo / "configs" / "draft.yaml"
    draft.write_text("name: draft\n", encoding="utf-8")  # written, never `git add`ed
    monkeypatch.setenv("MSFT_EXPERIMENTS_CATALOG", str(write_catalog(tmp_path, data)))
    fresh = experiments.load_catalog(repo=cfg.repo)
    job = experiments.add_job("draft", cfg=fresh)
    box = Box().install(monkeypatch)

    result = experiments.tick(fresh, budget=1, execute=True, day=DAY)

    assert box.signalled == []
    assert result["refused"] == [job["id"]]
    row = experiments.list_jobs(cfg=fresh)[0]
    assert row["status"] == "skipped"
    assert "git clone" in row["error"]
    assert "ten minutes later" in row["error"]


def test_tick_refuses_a_config_that_does_not_exist(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recipe pointing at a file nobody wrote is a typo, not a job for the box."""
    data = copy.deepcopy(CATALOG)
    data["recipes"]["ghost"] = {
        "kind": "data",
        "config": "configs/ghost.yaml",
        "est_minutes": 0.1,
        "what": "a config that was never written",
    }
    monkeypatch.setenv("MSFT_EXPERIMENTS_CATALOG", str(write_catalog(tmp_path, data)))
    fresh = experiments.load_catalog(repo=cfg.repo)
    experiments.add_job("ghost", cfg=fresh)
    box = Box().install(monkeypatch)

    result = experiments.tick(fresh, budget=1, execute=True, day=DAY)

    assert box.signalled == []
    assert result["refused"]
    row = experiments.list_jobs(cfg=fresh)[0]
    assert row["status"] == "skipped"
    assert "does not exist" in row["error"]


def test_tick_refuses_when_the_box_is_busy(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue(cfg, "cheap")
    box = Box(busy=True).install(monkeypatch)

    result = experiments.tick(cfg, budget=1, execute=True, day=DAY)

    assert box.signalled == []
    assert result["box_busy"] is True
    assert experiments.list_jobs(cfg=cfg)[0]["status"] == "queued"
    line = experiments.format_tick(result)
    assert "box busy" in line


def test_tick_leaves_the_row_queued_when_the_box_is_unreachable(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue(cfg, "cheap")
    Box(unreachable=True).install(monkeypatch)

    result = experiments.tick(cfg, budget=1, execute=True, day=DAY)

    assert result["box_unreachable"] is True
    row = experiments.list_jobs(cfg=cfg)[0]
    assert row["status"] == "queued"
    assert row["remote_id"] is None
    assert "ssh exited 255" in (row["error"] or "")
    assert "box unreachable: 1 queued" in experiments.format_tick(result)


def test_tick_reaps_a_row_a_dead_process_left_running(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = experiments.add_job("cheap", cfg=cfg)
    with experiments._writer(cfg.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, remote_id = ? WHERE id = ?",
            ("2026-11-12T09:00:00+00:00", job["id"], job["id"]),
        )
    Box({job["id"]: {"state": "unknown", "returncode": None, "tail": ""}}).install(monkeypatch)

    result = experiments.tick(cfg, budget=1, execute=True, day=DAY)

    assert result["reaped"] == 1
    row = experiments.list_jobs(cfg=cfg)[0]
    assert row["status"] == "failed"
    assert "interrupted" in (row["error"] or "")


def test_tick_books_a_running_row_whose_job_actually_finished(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = experiments.add_job("cheap", cfg=cfg)
    with experiments._writer(cfg.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, remote_id = ? WHERE id = ?",
            ("2026-11-12T09:00:00+00:00", job["id"], job["id"]),
        )
    Box({job["id"]: {"state": "done", "returncode": 0, "tail": "rc=0"}}).install(monkeypatch)

    result = experiments.tick(cfg, budget=1, execute=True, day=DAY)

    assert result["reaped"] == 1
    assert experiments.list_jobs(cfg=cfg)[0]["status"] == "done"


# --- report and the CLI ---------------------------------------------------


def test_report_counts_the_backlog_and_todays_verdicts(
    cfg: experiments.ExperimentsCfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue(cfg, "cheap", "mid")
    Box().install(monkeypatch)
    experiments.tick(cfg, budget=1, execute=True, day=experiments.today())

    payload = experiments.report(cfg)
    assert payload["queued"] == 1
    assert payload["done"] == 1
    assert payload["failed"] == 0
    assert payload["last"]["recipe"] == "cheap"
    assert payload["last"]["est_minutes"] == pytest.approx(0.1)
    line = experiments.format_report(payload)
    assert "1 queued" in line and "1 done" in line and "last cheap" in line


def test_ids_are_readable_and_unique(cfg: experiments.ExperimentsCfg) -> None:
    rows = queue(cfg, "cheap", "cheap")
    stamp = experiments.today().replace("-", "")
    assert [row["id"] for row in rows] == [f"exp-{stamp}-1", f"exp-{stamp}-2"]


def test_cli_catalog_and_list(
    cli_cfg: experiments.ExperimentsCfg, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["experiments", "catalog"]) == 0
    assert "cheap" in capsys.readouterr().out
    queue(cli_cfg, "mid")
    assert main(["experiments", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["recipe"] == "mid"


def test_cli_tick_plan_exits_zero_and_writes_nothing(
    cli_cfg: experiments.ExperimentsCfg, capsys: pytest.CaptureFixture[str]
) -> None:
    queue(cli_cfg, "cheap")
    before = cli_cfg.db_path.read_bytes()
    assert main(["experiments", "tick", "--budget", "1", "--date", DAY]) == 0
    assert "would run" in capsys.readouterr().out
    assert cli_cfg.db_path.read_bytes() == before


def test_cli_tick_reports_an_unreachable_box_with_its_own_code(
    cli_cfg: experiments.ExperimentsCfg,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue(cli_cfg, "cheap")
    Box(unreachable=True).install(monkeypatch)
    code = main(["experiments", "tick", "--budget", "1", "--execute"])
    assert code == experiments.EXIT_UNREACHABLE
    assert "box unreachable: 1 queued" in capsys.readouterr().out
