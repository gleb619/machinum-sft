# CLI coverage audit — 2026-11-08

The rule for this repo: a step that had to be done once must become a command, so the next run
is commands and not shell archaeology. This is the honest split from the first real run.

## What already went through `msft`

| Step | Command |
| --- | --- |
| dataset | `msft data build --execute` (queued as a `data` job) |
| training | `msft remote signal train --config …` → `msft train` inside the worker |
| merge | `msft remote signal merge --config …` |
| quantize | `msft remote signal quantize --config …` |
| gates | `msft eval --predictions out/eval-*.jsonl` |
| deploy / stop | `msft remote deploy`, `msft remote down` |
| queue state | `msft remote status`, `msft worker status|log|reset` |

## What was done by hand, and the command that replaces it

| Done by hand | Why it was wrong | Replaced by |
| --- | --- | --- |
| `python -c "snapshot_download(...)"` pasted into the container to resume the 4B download | a one-off that nobody could reproduce, and it silently gave up without a failure signal | `msft fetch --config …` (job kind `fetch`) |
| `uv pip install "transformers>=5.2.0"` inside the running container | reverted by the next recreate; the 4B job then died on 4.57.6 | baked into `deploy/remote/Dockerfile` with a version assertion |
| `git clone` + `cmake` of llama.cpp inside the container | lost with the container; quantize then had no `llama-quantize` | baked into `deploy/remote/Dockerfile`, `LLAMA_CPP_DIR` in compose |
| editing `merged/tokenizer_config.json` to fix `tokenizer_class` | a manual fix to a file the tool wrote | `merge.py` normalises `tokenizer_class` and `extra_special_tokens`, and fails if the tokenizer will not load |
| copying `tokenizer.json` / `vocab.json` / `merges.txt` from the base model over the merged dir | same, one layer deeper | `merge.py` copies them automatically and retries the load once |
| `rm out/model-4b.gguf` after discovering it was the 0.8B model | a symptom; the cause was a shared `out/merged` | `out/merged-<config name>` per run, plus `msft verify --config` |
| writing a `/tmp/Modelfile` and `docker cp` + `ollama create` by hand | the template was hand-written and wrong (Gemma format for a Qwen model) | `msft export --gguf … --name …` |
| `ollama cp` + `ollama push` to the account namespace | shell trivia that fails with a confusing error | `msft export --namespace <user> --push` |
| a bespoke `urllib` script to call `/api/chat` with `think: false` and dump predictions | the exact thing that made the gates go green, kept only in a scratch file | `msft probe --endpoint … --model … --cases …` |
| `docker compose restart worker` after pulling worker code | easy to forget, leaves old code running | `msft remote deploy --restart` |
| `ssh … du -sh /hf/…` to watch a download | no signal in the queue | `msft fetch` job logs progress; `msft remote log` shows it |

## Still manual, deliberately

- Choosing the config and reading the numbers in the job log. That is judgement, not plumbing.
- Deciding to push a model to a public namespace. It is irreversible from the repo's side, so it
  stays an explicit `--push` flag.
- Anything on the GPU box that needs a human at the console (BIOS, driver install, disk).

## Rule for the next session

If you find yourself typing `python -c` against a container, `docker cp`, or a `curl` to a model
endpoint, stop: either the command exists (`msft fetch|verify|export|probe`) or the command is
missing and belongs in this CLI, with a test, in the same commit.
