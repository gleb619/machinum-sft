# The experiment queue: from a note's proposal to one run on the box

`msft experiments` is the arrow the research loop was missing. `msft material` ends with a
distilled note whose frontmatter carries one `experiment:` sentence; that sentence used to
land in `docs/material/experiments.md` and stop there, because running it is a GPU decision
and `AGENTS.md` governs the GPU. This queue makes the last step mechanical without making it
unsupervised: a note may ask for one **recipe** — a run whose cost the repository has already
measured — and a daily `tick` starts the cheapest queued recipe, one at a time, and records
the verdict.

```
note frontmatter            catalog                    box
experiment_job:  ──sync──▶  validate + price  ──▶  jobs row (queued)
                                    │                     │
                            unknown recipe →          tick (cheapest first,
                            dropped + counted          nothing while busy)
                                                          │
                                              msft remote signal <kind>
                                                          │
                                              done/<id>.json → done | failed
```

Four files carry it: `configs/experiments/catalog.yaml` (the menu),
`<material dir>/.state/experiments.db` (the queue), `src/msft/experiments.py` (validation,
scheduling, reporting) and `scripts/experiments-daily.sh` (the cron entrypoint).

## Recipe vs proposal

A **proposal** is prose. It is written by the distiller, it may be vague, and it costs
nothing:

```yaml
experiment: On smoke-qwen3.5-0.8b, train one epoch with load_in_4bit: true and one with
  the bf16 baseline, then diff the five protocol gates.
```

A **recipe** is a job: a kind the worker serves, a committed config, optional extra argv, an
`est_minutes` measured from a real run, and one line saying what question it answers. The
note may also carry a structured block — optional, because most notes have nothing that is
worth a GPU:

```yaml
experiment_job:
  recipe: smoke-train
  hypothesis: a 4-bit first epoch loses less than 0.1 eval loss against bf16
```

`sync` reads the notes of one day, validates each block against the catalog, dedupes on
`(origin, recipe)` — `origin` is `note:<slug>`, so re-running `sync` is free — and inserts a
`queued` row. The prose `experiment:` line is untouched and still reaches `experiments.md`:
the note stays readable, and the queue only ever holds things the repository already knows
how to run.

## Why the catalog is an allow-list

A queue that accepts free-form jobs is a queue a model can point at a 12B training run. The
distiller is a language model reading blog posts; it will happily propose "fine-tune
Qwen3.5-9B for three epochs" as a Tuesday idea. So the permission is the **recipe name**, not
the sentence:

- `sync` drops any recipe that is not a key in `configs/experiments/catalog.yaml` and counts
  it as `unknown_recipe` — no exception, no job.
- `est_minutes` comes from the catalog, never from the note. The note's "~45 minutes" is a
  guess; the catalog's number is a measurement with its arithmetic in a comment.
- The catalog is data in git, so widening the menu is a reviewable diff: a new recipe means a
  new measured number, not a new code path.

`kind` must be one of `worker.JOB_KINDS` (`train`, `merge`, `quantize`, `eval`, `data`,
`doctor`, `fetch`, `verify`), `config` must be a repo-relative `configs/*.yaml`, and `args`
may only use flags the real CLI accepts — checked against `msft train|data|eval|verify --help`
when the recipe was written. There is deliberately no `arena` recipe: `JOB_KINDS` has no
`arena` kind, so `remote signal arena` cannot be queued at all.

## Cost decides the order

`tick` sorts `queued` rows by `est_minutes` ascending, then by age. That single number is the
whole scheduler, which is why every recipe carries its derivation:

| recipe | kind | est | where the number comes from |
| --- | --- | --- | --- |
| `data-build` | data | 0.1 | `msft data build` under 5 s in the 2026-11-08 run |
| `smoke-quantize` | quantize | 0.2 | 9.4 s measured (f16 1.56 GB → 516 MB Q4_K_S) |
| `eval-4b-gates` | eval | 0.2 | pure-Python gates over a JSONL; container start is the cost |
| `smoke-merge` | merge | 0.6 | 31.6 s measured (adapter → 1.7 GB bf16) |
| `verify-4b` | verify | 0.8 | `--deep` re-hashes 2.63 GB at the ~55 MB/s the quantize wrote |
| `smoke-train` | train | 3.0 | 167 s end to end (18 steps, 3 epochs) |
| `short-4b-train` | train | 3.0 | 165.8 s for 4 steps, 9.4 GB of weights already in `/hf` |

A wrong `est_minutes` silently starves the cheap jobs (or the dear ones), which is the failure
mode to watch when adding a recipe: measure, divide by 60, round up to 0.1 min, write the
arithmetic in the comment. `short-4b-train` carries a caveat for the same reason — its 3.0 min
assumes a warm HF cache, and a cold one adds a download no estimate here covers.

## One job a weekday, the backlog on the weekend

```
Mon-Fri   budget 1   watch up to 5 400 s (90 min)   one job, then stop
Sat-Sun   budget None (drain)  21 600 s (6 h)       start the next as each finishes
```

The rule lives in `msft.experiments.schedule_budget` and the script reaches it with
`tick --budget auto`, so there is exactly one implementation and a test can inject the day
(`2026-11-14` is a Saturday). `--budget 0` also means drain; `--budget 3` is three jobs.

**Single-GPU serialisation is not a flag.** The box serves one job at a time by design (two
trainings on one 16 GB card is an OOM, not a speed-up), and `tick` refuses to start anything
while a job sits in `.worker/running/` — the same check `remote run --tmux` refuses on. A row
that is `running` locally is the same answer: the queue is locked behind its own job.

`est_minutes` orders the queue; `time_budget_s` bounds the watch. If the watch runs out while
a job is still running on the box, the row is marked `failed` with an error that says the job
may still finish and names `msft remote logs --job <id>` — the local record stops honestly
rather than claiming a verdict it does not have.

## State

One SQLite file, next to `material.db` and inside the git-ignored tree:

```
docs/material/.state/experiments.db    the queue — rows are the truth
docs/material/.state/logs/<id>.log     the tail of each finished job's log
docs/material/.state/experiments.lock  the script's flock target
```

```sql
CREATE TABLE jobs (
  id TEXT PRIMARY KEY,          -- exp-20261112-1: readable in a log line, unique per day
  created_at TEXT NOT NULL,
  origin TEXT NOT NULL,         -- note:<slug> | manual | recipe:<name>
  recipe TEXT NOT NULL, kind TEXT NOT NULL, config TEXT, args TEXT NOT NULL,  -- args: JSON
  est_minutes REAL NOT NULL,    -- from the catalog, never from the note
  hypothesis TEXT,
  status TEXT NOT NULL,         -- queued | running | done | failed | skipped
  remote_id TEXT,               -- the worker's job id (equal to `id`)
  started_at TEXT, finished_at TEXT, rc INTEGER, log_path TEXT, error TEXT,
  metrics TEXT                  -- JSON: watched_s, timed_out, est/actual minutes
);
```

`remote_id` equals `id`, so `done/<id>.json` on the box maps one-to-one onto a row and
`msft remote logs --job exp-20261112-1` works without a lookup.

The statuses are the whole state machine:

```
queued ──tick──▶ running ──done────▶ done
   ▲                │  └──failed────▶ failed
   │                └──watch timeout─▶ failed (error says the box may still be running it)
   └──signal failed (box unreachable): back to queued
queued ──config not committed──▶ skipped
```

`skipped` is reserved for "we refused to signal it": an uncommitted or missing config, which
the box would only discover ten minutes into the container run.

## The commands

```bash
msft experiments catalog                 # the menu, cheapest first
msft experiments sync                    # today's notes → queued jobs (counters on stderr)
msft experiments sync --date 2026-11-10  # a backlog day
msft experiments sync --dry-run          # counts only: no database file is created
msft experiments list                    # the queue
msft experiments list --status failed
msft experiments add smoke-train --hypothesis "does 4-bit merge cleanly?"
msft experiments tick                    # PLAN: what would run, in what order
msft experiments tick --execute          # start the cheapest, watch it, record the verdict
msft experiments tick --budget 0 --execute      # ignore the weekday limit and drain
msft experiments report                  # backlog, today's verdicts, est vs actual minutes
```

Everything is a plan until `--execute`. `tick` without it reads the queue read-only and
writes nothing — not even the database file. Exit codes: `0` plan/ok, `1` a job failed, `2`
config error (bad catalog, unreadable notes), `4` the box could not be reached — distinct
from `2` because an unreachable box is not a broken repository, and the cron wrapper reports
it and exits 0.

## The daily run

`scripts/experiments-daily.sh` is the cron entrypoint: sync, then tick, then one line.

```bash
bash scripts/experiments-daily.sh             # sync + tick, weekday or weekend per `date`
bash scripts/experiments-daily.sh --dry-run   # both steps in plan mode
```

What it guarantees:

- **Its own lock.** `flock -n` on `docs/material/.state/experiments.lock` — not material's.
  A held agent must not block the GPU job of the day, and a six-hour weekend drain must not
  block tomorrow's notes. A second run prints `experiments: another run holds the lock` and
  exits 0, because a cron job that fails on contention retries and storms.
- **One line on stdout**, and it comes from `tick`, not from `report`: only `tick` knows
  whether the box was reachable, and `box unreachable: 3 queued` is exactly the fact a silent
  13:00 cron would otherwise hide. `report` prints the same counters offline for a human.
- **Exit 0 when nothing was due or the box was unreachable**, exit 1 only when a job actually
  failed, exit 2 for config/environment. That keeps the DM quiet on a green day and honest on
  a red one.

Wiring, next to the material job (18:00 local, after the research loop has had the day):

```bash
# ~/.hermes/scripts/experiments-daily.sh
exec bash /home/boris/WORKSPACE/machinum-sft/scripts/experiments-daily.sh "$@"
```

```bash
hermes cron create '0 18 * * *' \
  --name machinum-experiments-daily \
  --workdir /home/boris/WORKSPACE/machinum-sft \
  --script experiments-daily.sh \
  --no-agent \
  --deliver 'telegram:Boris Velichko (dm)'
```

Plain crontab equivalent:

```cron
0 18 * * * cd /home/boris/WORKSPACE/machinum-sft && scripts/experiments-daily.sh 2>>$HOME/.local/state/experiments-daily.err.log
```

**Raise the script timeout before trusting a weekend.** Hermes cron's
`cron.script_timeout_seconds` defaults to 3600 s, and a weekend drain may watch a job for
21 600 s; the outer kill would land mid-job and leave a `running` row the next tick has to
reap. Set the timeout above `21600 + 600`, or run the weekend drain by hand.

Environment overrides (the CLI reads them itself; the script only forwards the two paths):

| Variable | Default |
| --- | --- |
| `MSFT_MATERIAL_DIR` | `<repo>/docs/material` |
| `MSFT_EXPERIMENTS_CATALOG` | `<repo>/configs/experiments/catalog.yaml` |
| `MSFT_EXPERIMENTS_BUDGET` | `auto` (weekday/weekend rule) |
| `MSFT_EXPERIMENTS_TIME_BUDGET` | 5400 weekday / 21600 weekend |
| `MSFT_REMOTE_HOST`, `MSFT_REMOTE_DIR` | from `configs/worker.local.yaml` |

## When a run is red

| What you see | What it means | First command |
| --- | --- | --- |
| `box unreachable: 3 queued`, exit 0 | ssh failed or timed out: nothing was signalled and the rows went back to `queued` with the reason | `msft remote status --execute`, then `msft experiments list` and read `error` |
| `N refused (config not committed)`, row `skipped` | the catalog names a config that is not in `HEAD` (or has uncommitted changes): the box runs a `git clone`, so this job would fail ten minutes later | `git status -- <config>`, commit and push, then `msft experiments tick --execute` |
| exit 2, `unknown recipe '…'` | the note proposed a name that is not in the catalog, or the catalog is unreadable | `msft experiments catalog`; the note's `experiment_job.recipe` is the thing to fix |
| `unknown_recipe` > 0 after `sync` | a note named an invented recipe; it was dropped, not queued | nothing — the prose line is still in `experiments.md`; broaden the catalog only if the idea is worth a measured recipe |
| `a job failed`, exit 1 | the remote stage exited non-zero | `msft experiments list --status failed`, then `msft remote logs --job <id>`; the row's `log_path` is the local tail |
| row `failed`, `watch timed out` | the tick's `--time-budget` ran out while the job was still running on the box | `msft remote logs --job <id> --wait`; the box keeps running, the local record stopped |
| row `failed`, `interrupted: a previous tick left this row running` | the watcher process died (cron killed, laptop asleep) and the box has no verdict | check the box with `msft remote status --execute`; re-queue with `msft experiments add <recipe>` if it still matters |
| `box busy, nothing started` | the box is already running a job (or a row is `running` here) | `msft remote status --execute`; the next tick picks the queue up |
| `nothing due`, exit 0 | the queue is empty: today's notes proposed nothing, or everything is done | `msft experiments sync --dry-run` to see what the notes actually contained |

Never "fix" a red queue by deleting `experiments.db`: it is the record of what was already
run, and the recipes are cheap enough that a re-run is a real cost on a real card.

## What this does not do

- **It does not decide what is interesting.** The catalog says what is *runnable and cheap*,
  not what is *worth* running; the hypothesis in the row is the reasoning, and widening the
  menu is still a human diff.
- **It does not touch the material pipeline.** `sync` reads the notes and `experiments.md`;
  it writes neither. A day whose notes have no block simply queues nothing.
- **It does not retry a failed job.** A failed row stays failed; the fix is a new recipe or a
  new `add`, so a broken config cannot burn the same GPU hour every night.
- **It does not run two jobs at once**, even on a weekend drain: the card is the limit, and
  the worker queue is the enforcement, not this module.
- **It does not sync data to the box.** A job that needs a new dataset is a `remote sync`
  plus a recipe, in that order, by hand.
