# Weekend experiment keeper — one tick

Fired by `at loop` every 20 minutes from the repo root. Today is **2026-11-28 (Saturday)**.
You are the keeper of the day's experiment campaign: start the next piece of work, check what is
running, heal what broke, record what happened, commit what is green, and exit.

**Budget: 15 minutes.** Long work is started detached and checked on a later tick. Never sit
waiting on a training run, a download or an agent. If a command would block for more than ~3 min,
start it with `setsid nohup … &` (or `at -p`) and leave.

## Files

- Campaign (what runs, in what order): `docs/plans/weekend-2026-11-28.md`
- Day journal (append-only, real numbers only): `docs/run-2026-11-28.md`
- Keeper state (git-ignored): `docs/material/.state/weekend.json`
- Experiment results: `out/weekend/<ID>/RESULT.md` (git-ignored), report tables copied to the
  journal.

State shape (create once if missing):

```json
{"date": "2026-11-28",
 "items": {"W2": {"status": "todo"}, "W1": {"status": "todo"},
           "W3": {"status": "todo"}, "W4": {"status": "todo"},
           "W5": {"status": "todo"}},
 "failures": 0, "last_tick": null}
```

Item status: `todo | running | done | failed | blocked`. Store the `at` run id in the item.

## Hard rules (AGENTS.md applies in full)

- Never read or print `.env*`, `configs/worker.local.yaml`, `docs/secrets.json` or any key file.
- Never edit code on the box. The box is a `git pull` clone; the container bind-mounts it.
- Never `git stash`, `git reset`, `git rebase`, `--amend`, or force-push.
- Never commit `out/`, `data/`, `docs/material/.state/`, `*.gguf`, `*.local.yaml`, `.env*`.
- Conventional Commits, no `Co-Authored-By`, 100-column lines, work in English.
- One GPU job at a time. Do not parallelise training.
- A CPU run is a bug: if a job reports `cpu`, stop and report; never set `MSFT_ALLOW_CPU=1`.
- Never hand-write a chat template; the GGUF carries its own.
- No claim without the command, the rc and the number.

Reaching the box without exposing its host — put this function in the shell you use:

```bash
box() { .venv/bin/python -c 'import sys;from msft.remote import remote_host,_run_ssh as r;rc,o,e=r(remote_host(),sys.argv[1]);print(o,end="");print(e,end="",file=sys.stderr);sys.exit(rc)' "$1"; }
H=$(.venv/bin/python -c "from msft.remote import remote_host;print(remote_host().split('@')[-1])")  # never echo $H
```

## Tick

### 1. Read state

```bash
date '+%F %T'
git status --porcelain
tail -25 docs/run-2026-11-28.md
cat docs/plans/weekend-2026-11-28.md
```

Load `docs/material/.state/weekend.json` (create it if missing). If its `date` is not
`2026-11-28`, start a fresh state for today and leave yesterday alone.

### 2. Health check (three commands, no waiting)

```bash
.venv/bin/msft --json remote status --execute        # gpu line + counts, rc
pgrep -af 'experiments-daily.sh' || true            # a drain in progress?
at status | head -15                                 # agent runs (global, other projects included)
```

- No `GPU:` string, or a `WARNING:` → **heal the box** (§5) before anything else.
- `at serve --daemon` not running (`at loop list` says so, or `pgrep -f 'at serve --daemon'` is
  empty) → `at serve --daemon >/dev/null 2>&1 &` and confirm the daemon is alive next tick.

### 3. The experiment queue wins

If `msft experiments list` shows a `queued` row and nothing is running on the box and no
`experiments-daily.sh` process is alive:

```bash
setsid nohup bash scripts/experiments-daily.sh \
  > "docs/material/.state/weekend-daily-$(date +%H%M).out" 2>&1 < /dev/null &
echo "drain pid $!"
```

That script is the weekend rule (`--budget auto` = drain), one job at a time, and it exits 0 when
the box is unreachable and 1 when a job failed. Do not replace it with ad-hoc `remote signal`
unless §5 says to.

### 4. Otherwise run the next campaign item

When the queue is empty and no item is `running`:

1. Take the first item in order `W2, W1, W3, W4, W5` whose status is `todo` or `failed` (one retry
   maximum per item; `failures >= 2` → `blocked`).
2. Mark it `running` in the state file **before** dispatching.
3. Dispatch it detached, with the spec as the prompt:

```bash
at -p "Execute experiment <ID> for the machinum-sft weekend campaign.
Read /media/boris/data_nvme1n1/WORKSPACES/WORKSPACE/machinum-sft/docs/plans/weekend-2026-11-28.md section <ID> and follow it exactly.
The repo is the cwd. Do not commit or push: the keeper commits.
Write out/weekend/<ID>/RESULT.md with the exact commands, rc and numbers, ending with a final line 'RESULT: done' or 'RESULT: failed: <reason>'.
Report the same numbers in your last message." --timeout 5400000
```

`at -p` returns immediately; copy the run id from its output into the state item.

### 5. Heal what is red

| Symptom | Action |
| --- | --- |
| box unreachable / no `GPU:` line | `timeout 900 .venv/bin/msft remote deploy --execute`, then `msft remote status --execute` again. Still red → alert, do not start work. |
| worker container unhealthy | same `remote deploy`; confirm `heartbeat.json` fresh |
| job `failed` rc≠0 | `msft experiments list --status failed`; read the row's `log_path`; then `msft remote logs --job <id>` |
| `no predictions` / rc=2 on an eval row | known producer gap: do not retry it; the answer is W1, queue nothing else for it |
| `CUDA out of memory` / `OutOfMemoryError` | `box 'docker restart ollama'`, wait for the GPU line, requeue the item once |
| `config not committed` (catalog) | the box runs a clone: use `msft remote signal <kind> --config <cfg> --execute` (it pushes the config, measured in S34), or commit the config if the experiment owns it |
| agent run `failed` / `stuck` | `at status <id> -n 80`; read the tail. If it is a one-line code error you can fix safely, fix it, run the guard, commit. Otherwise re-dispatch once with the error text in the prompt, then mark `failed`. |
| disk < 10 GB free on `/workspace` | report it, stop starting new work, pause the loop |
| `git status` shows a file you did not expect | do not commit it; name it in the journal and leave it to its owner |

### 6. Check runs that finished this tick

For every item marked `running`:

```bash
at status <runId> -n 70
```

- `done` → read `out/weekend/<ID>/RESULT.md`, open the artefact it cites, re-run one cheap command
  from it yourself, then set `done` (or `failed` if `RESULT: failed`).
- `running` → leave it.
- `failed` → §5.

### 7. Commit, gate, push

```bash
git status --porcelain            # ignored paths never appear here
```

- **Docs only** (`docs/...`) → commit directly,
  `docs(experiments): record <what> — <the one number>`.
- **Code/configs/tests changed** → run the gate first, never piped:
  `bash scripts/guard.sh --allow-dirty > /tmp/wk-guard-$(date +%H%M).out 2>&1; rc=$?`
  - `rc=0` → commit with a Conventional Commit, re-run `bash scripts/guard.sh` clean, and only
    then `git push origin main` (only on `main`).
  - `rc≠0` → do **not** commit code; leave the tree for the next tick, write `GUARD RED: <first
    failing line>` in the journal, and fix it next tick.
- Nothing changed → nothing to commit.
- If `index.lock` exists, wait 10 s and retry once; never delete a fresh lock. Delete only a lock
  older than 5 minutes **and** with no `git` process alive.

### 8. Journal

Append under `## Ticks` in `docs/run-2026-11-28.md`; one line, real numbers, no prose:

```
- <HH:MM> — <queue|W1|W2|W3|W4> — <what happened: rc, minutes, gate counts> — next: <action>
```

Commit the journal in the same commit as the other docs changes.

### 9. Stop conditions

Pause the loop (`at loop pause "$AT_LOOP_ID"`) and write the reason in the journal when:

- three consecutive ticks end with the same hard failure and no work running;
- the box stays unreachable after two `remote deploy` attempts;
- a destructive change would be needed (that is the owner's call).

## Output

End with exactly one line:

`KEEPER <HH:MM> — <idle|running:<ID>|queue|healed:<what>|blocked> — <one sentence> — next: <action>`
