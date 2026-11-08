# AGENTS.md — rules for anyone (human or agent) changing this repo

This repo drives real GPU time on a real machine. A wrong assumption costs an hour, not a test
run. Read this, then read `docs/memory.md` (traps and their fixes) and `docs/handoff.md` (the
working loop) before touching `src/msft/{train,merge,quantize,worker,deploy}.py`,
`deploy/remote/`, `scripts/`, or any `configs/*.yaml`.

## Non-negotiables

1. **One source of truth.** Code lives here, is committed here, and reaches the box by
   `git pull` into `~/WORKSPACE/machinum-sft`. Never edit code directly on the box, never copy
   files over, never keep a second checkout. The container bind-mounts the repo; that is the
   only sync mechanism.
2. **`scripts/guard.sh` before every merge.** `ruff` → `pytest` → `wrkflw validate` → `wrkflw
   run --runtime emulation`. Never merge a branch whose CI is red, local or on GitHub. `GUARD
   OK` is the only acceptable green.
3. **Never install into a running container as the fix.** The writable layer is reverted by the
   next `docker compose up`. Anything a run needs goes into `deploy/remote/Dockerfile`, and you
   then verify with the exact command in `docs/memory.md` → "The container's writable layer is
   disposable".
4. **Never hand-write a chat template.** The GGUF carries the tokenizer's own `chat_template`.
   A hand-copied one (this repo shipped Gemma's `<|turn>` to a Qwen model) makes the model
   narrate its reasoning instead of answering.
5. **Send `think: false`.** Qwen3.5 defaults to thinking. Any client, script or test that talks
   to the exported model must set it, or every reply is a reasoning transcript.
6. **`rc=0` is not proof.** Verify the artifact: `ls -la out/*.gguf`, compare against the
   expected size (Q4_K_S ≈ 30 % of f16; a 4B Q4 file is ~2.5 GB), and load the tokenizer before
   declaring success. A 516 MB `model-4b.gguf` was once a 0.8B model with the right filename.
7. **A CPU run is a bug.** The generated train script refuses without CUDA or when the model
   lands on `cpu`. Do not "fix" that by setting `MSFT_ALLOW_CPU=1` in a job; find why the GPU
   is missing.
8. **No secrets in the repo, ever.** Hosts, keys and tokens live in `configs/worker.local.yaml`
   (git-ignored) or the environment. Docs must not contain hostnames, IPs or usernames with
   credentials; `[REDACTED]` is the convention for a key that must be mentioned. Do not
   `printenv` an API key into any output.

## How to change things

- **New behaviour → test first-ish, not last.** `pytest -q` must stay green; `ruff check .` is
  part of the gate. 100-column lines.
- **Configs are data, code is code.** A new model = a new `configs/<model>.yaml`; do not fork
  the training script per model.
- **Every number in docs must come from a real run** with the command that produced it. Do not
  write "should be fast" — write `166 s`, and say which stage ate the time.
- **When a run fails, fix the cause and add the check that would have caught it** (a test, an
  assertion, a guard line, a sentence in `docs/memory.md`), in the same commit.
- **Commits:** Conventional Commits (`fix(merge): …`), no merge commits, no `git stash`, no
  `git reset`. Push to `origin main` when the guard is green.
- **Before queueing work on the box:** `msft remote status --execute` must print a `GPU:` line,
  not a `WARNING:`. If it warns, stop.

## Definition of done for a model run

```
msft remote status --execute          # GPU line, queue empty
job logs                              # train/merge/quantize rc=0 with timings
ls -la out/*.gguf                     # size matches the model, not the last one
bash scripts/ollama-export.sh …       # model appears in `ollama list`
msft eval --predictions …             # gates green on hand-checked cases
```

Miss any line and the run is not done — it is a job that exited 0.
