# Worker mode: the GPU box as a queue

The laptop never runs training; the box never holds a second copy of the code. The loop is:

```
laptop (source of truth)                     GPU box: ~/WORKSPACE/machinum-sft
------------------------                     ---------------------------------
edit → guard → commit → push   ──pull──►     git pull --ff-only
msft remote signal train --execute           docker exec msft-worker python -m msft worker signal …
                                             worker (container) picks the job, runs it
                                             .worker/{inbox,running,done,logs} + heartbeat.json
```

## Why a queue and not "ssh and run"

- **One at a time.** Two trainings on one 16 GB card is an OOM, not a speedup. The daemon
  serialises.
- **Survives the ssh session.** The container owns the job; closing the laptop does not kill it.
- **Auditable.** Every job leaves `done/<id>.json` (rc, duration, log path) and
  `logs/<id>.log` with the exact command line, so a failed run can be explained later.
- **Health-checkable.** `heartbeat.json` is rewritten every loop; compose's healthcheck reads
  its age, so a wedged worker shows as unhealthy instead of silently idle.

## Job file

```json
{"id": "train-20261108T091500Z", "kind": "train", "config": "configs/qwen3.5-4b-lora.yaml",
 "extra_args": [], "created": "...", "status": "pending"}
```

`kind` is one of `train`, `merge`, `quantize`, `eval`, `data` (mapped to `data build`), `doctor`.
The daemon appends `--execute`; a job never runs in dry-run mode by accident.

## Commands

```bash
# on the laptop
cp configs/worker.example.yaml configs/worker.local.yaml   # host lives here, git-ignored
msft remote deploy --execute            # pull + docker compose up --build
msft remote signal train --execute      # queue
msft remote status --execute            # container + queue + heartbeat
msft remote down --execute

# on the box, or through docker exec
msft worker serve --dir .worker [--once]
msft worker signal merge --config configs/qwen3.5-4b-lora.yaml
msft worker status --dir .worker
msft worker log --id train-... --lines 80
msft worker reset --dir .worker          # clears the queue, keeps logs
```

## Failure modes seen in practice

| Symptom | Cause | Fix |
| --- | --- | --- |
| job exits 2 in 0.0 s, log shows argparse usage | job kind does not map to a CLI leaf | `data` maps to `data build`; keep `JOB_KINDS` in sync with the CLI |
| job exits 1 with a transformers version error | model newer than the pinned stack | upgrade inside the container, then bake it into the image |
| status shows `since None` | the running file is written before `started` | cosmetic; the `done` record carries the real timestamps |
| worker keeps running old code after a pull | the daemon imported the module at start | restart the container after a pull that changes `worker.py` |
