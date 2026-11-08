# The guard: never merge a red branch

`scripts/guard.sh` runs the same checks as `.github/workflows/ci.yml`, in the same order, on
this machine, and answers one question with one exit code: **is this tree safe to merge?**

The rule it enforces is simple and non-negotiable: **CI must be green before merging.** A red
`main` is not a cosmetic problem here — `main` is what the GPU box pulls, and a broken
`msft` package or a broken test run means an adapter run that fails hours in, after the GPU
time is already spent. Branch protection on GitHub is the real gate; the guard is its local
mirror, cheap enough to run before every push instead of after.

## What it runs

| # | Step name (`--json`) | Command | Fails when |
| --- | --- | --- | --- |
| 0 | `dirty-tree` | `git status --porcelain` | the tree is dirty and `--allow-dirty` was not passed |
| 1 | `ruff` | `ruff check .` | any lint finding |
| 2 | `pytest` | `pytest -q` | any failing test |
| 3 | `ci-validate` | `wrkflw validate .github/workflows/ci.yml` | the workflow YAML is invalid |
| 4 | `ci-run` | `wrkflw run --runtime emulation .github/workflows/ci.yml` | the workflow fails locally |

`ruff` and `pytest` are taken from `./.venv/bin` first, then from `PATH`. Execution stops at
the first failing step: a red lint run never gets to spend 8 seconds on the CI emulation, and
you never read test output that was produced from code that would not pass CI anyway.

The order matches CI (`Lint`, then `Test`), so a green guard means the same two commands CI
runs have passed here. The wrkflw steps then replay the workflow file itself, which catches
what a hand-run `pytest` cannot: a workflow that no longer parses, a step whose `run:` block
was edited into something broken, a runner image that no longer resolves.

## Usage

```bash
./scripts/guard.sh                     # before you merge: dirty tree is refused
./scripts/guard.sh --allow-dirty       # work in progress, same checks
./scripts/guard.sh --json              # one JSON object on stdout, banners on stderr
./scripts/guard.sh --require-ci        # a missing wrkflw is a failure, not a skip
./scripts/guard.sh --allow-dirty --json > guard.json   # what a script or CI job consumes
```

| Flag | Effect |
| --- | --- |
| `--allow-dirty` | run on a dirty tree; the offending paths are listed either way |
| `--json` | emit exactly one JSON object on stdout for the run itself; every banner, hint and log tail goes to stderr. `--help` still prints usage to stdout, and the usage/config errors that precede the run exit before any object is written |
| `--require-ci` | exit 4 when `wrkflw` is missing instead of skipping the CI steps |
| `-h`, `--help` | usage |

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | every required step passed — `GUARD OK` |
| `1` | a step failed; the red banner names it, and in `--json` mode the last 40 lines of that step are reprinted to stderr (banner mode already streamed them) |
| `2` | usage error, not a git repository, or `ruff`/`pytest` missing |
| `3` | the working tree is dirty and `--allow-dirty` was not passed |
| `4` | `--require-ci` was passed but `wrkflw` is not installed |

Skipped CI steps do **not** fail the run: a laptop without `wrkflw` still gets a real
`ruff` + `pytest` verdict, and a yellow `SKIP` line says plainly that the workflow was not
replayed. Use `--require-ci` wherever the answer has to be complete (nightly, pre-push hook,
release branch).

## The `--json` contract

One object, one line, on stdout (wrapped below for readability). Nothing else is written to
stdout by a run in this mode — usage errors and `--help` exit before the object is written,
and every step's log goes to stderr or to the run directory:

```json
{"steps": [{"name": "ruff", "cmd": ".venv/bin/ruff check .", "status": "ok", "seconds": 0.011},
           {"name": "pytest", "cmd": ".venv/bin/pytest -q", "status": "ok", "seconds": 0.725}],
 "ok": true}
```

- `status` is `ok`, `failed` or `skipped`.
- `seconds` is wall-clock time as a float.
- `ok` is `true` only when no step failed and no required step was skipped. A dirty tree that
  was refused produces `{"steps": [{"name": "dirty-tree", ... "status": "failed"}], "ok": false}`
  and exit 3; the explanation is on stderr.
- `cmd` shows the resolved binary, so it may be `.venv/bin/ruff` or an absolute `PATH` hit.

Consumers should read `ok` and the step statuses, not scrape the human banners.

## Why a dirty tree is refused

Uncommitted work means the tree you tested is not the tree you are about to merge. That is
exactly how a green local run becomes a red `main`: the fix was edited in after the last
`pytest`, or a `git checkout -- .` reverted it. The guard lists every dirty path (including
untracked files) and stops, so `--allow-dirty` is a deliberate act for work in progress, not
the default. `--allow-dirty` prints a `note:` with the file count — a run that skipped the
gate should never look like a run that passed it.

## Installing wrkflw (one line, no sudo)

```bash
mkdir -p ~/.local/bin && curl -fsSL https://github.com/bahdotsh/wrkflw/releases/download/v0.8.0/wrkflw-x86_64-unknown-linux-musl.tar.gz | tar -xz -C ~/.local/bin
```

The tarball holds a single `wrkflw` binary. The musl build is static, so no libc or runtime
dependencies come with it:

```console
$ wrkflw --version
wrkflw 0.8.0
$ ldd ~/.local/bin/wrkflw
        statically linked
```

`file` reports the musl build as "dynamically linked" — it is a static-PIE binary; `ldd` is
the reliable check. If `~/.local/bin` is not on `PATH`, `guard.sh` cannot see it and will
print the `SKIP` line; add it to `PATH` in your shell profile.

## How this enforces "CI green before merging"

The rule needs a command that can be wrong, not a convention that can be forgotten:

```bash
# before every push/merge
./scripts/guard.sh --require-ci || { echo "not merging: guard is red"; exit 1; }

git push            # only reached when ruff, pytest and the workflow all passed
```

As a pre-push hook (`.git/hooks/pre-push`, `chmod +x`):

```bash
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/scripts/guard.sh" --require-ci
```

A pre-push hook is local and bypassable (`--no-verify`), so it is a convenience, not the gate.
The gate is branch protection requiring the `ci` workflow on `main`; the guard exists so you
find out about a red branch before CI does, not instead of CI.

`scripts/ci-local.sh` is the same program for unattended runs — identical steps, but it never
refuses a dirty tree, because a scheduled job has to report on whatever the checkout holds:

```bash
./scripts/ci-local.sh --json --require-ci       # nightly
# crontab: 30 3 * * *  cd /path/to/machinum-sft && ./scripts/ci-local.sh --json --require-ci >> /var/log/msft-guard.log 2>&1
```

It is a thin wrapper (`exec guard.sh --allow-dirty "$@"`), so the checks cannot drift apart.

## Reading a failure

The banner names the step and its exit code, then prints the last 40 lines of that step's
output — enough for a traceback or a lint diff, short enough to stay readable. The full log
stays behind at the printed path (`…/msft-guard.XXXXXX/<step>.log`) whenever a step fails, and
the directory is removed on success.

Output is scrubbed on the paths that quote it back — the dirty-file list and the `--json`
failure excerpt; in banner mode a failing step's output is streamed raw as it runs, which is
what you want while watching a test suite. Credentials embedded in URLs
(`https://user:token@…`) and `token=`/`key=`/`password=` query parameters are masked there.
The guard never prints the git remote URL, never dumps the environment, and never echoes
`.env`.

## Caveats

**Emulation is not a container.** `--runtime emulation` executes the workflow's `run:` steps
directly on this host, with this host's Python, network and filesystem. It is fast and needs
no Docker, but it is not isolation: the `Install` step really does `pip install -e .[dev]`.
The CI steps therefore run with two pins: `$REPO_ROOT/.venv/bin` first on `PATH`, so the
workflow's `python`/`pip` are the project venv rather than whatever the caller's shell had,
and `PIP_TARGET` set to a temporary directory, so the install cannot overwrite that venv's
editable install (without it, your `python -m msft` ends up pointing at a deleted sandbox).
The install also leaves `src/machinum_sft.egg-info/` in the checkout — it is ignored, so the
tree stays clean, but it is a write. Do not run the emulation against a workflow you do not
trust.

**wrkflw 0.8.0 is young.** The emulated checkout honours `.gitignore` but not its `!`
re-includes, so a rule like `*.jsonl` plus `!tests/fixtures/*.jsonl` silently removes tracked
fixtures and `pytest` fails inside the emulation while real CI passes. This repository hit
exactly that and removed the pair ( `data/` and `out/` are ignored wholesale). If a future
`.gitignore` reintroduces a negation, expect a red `ci-run` that is a wrkflw defect, not your
code — compare `git ls-files tests/fixtures` with what the run sees before debugging further.
