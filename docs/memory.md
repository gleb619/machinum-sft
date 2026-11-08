# memory.md — caveats and their fixes, as of 2026-11-08

This is the short, load-bearing list. `docs/handoff.md` explains the workflow; this file is
what to check before you believe anything. Each entry is a fact that cost real time once.

## Container and image

**The container's writable layer is disposable.** `docker compose up -d` after any compose edit
recreates the container and silently reverts everything installed inside it. On 2026-11-08 this
cost two runs: `transformers` went back to 4.57.6 and `/opt/llama.cpp` vanished, so the 4B job
died on import and quantize died on a missing binary.
→ Anything a run needs belongs in `deploy/remote/Dockerfile`. After a recreate, assert:
```
docker exec msft-worker python -c "import transformers;print(transformers.__version__)"   # >= 5.2
docker exec msft-worker ls /opt/llama.cpp/build/bin/llama-quantize
```

**Qwen3.5 needs `transformers >= 5.2.0`.** Unsloth raises
`ImportError: Your transformers version of X does not support Qwen3.5`. The image pins it and
asserts the version at build time.

**`pip` does not exist in a uv venv.** The entrypoint installs the mounted package with
`uv pip install --python /opt/venv/bin/python --no-deps -e /workspace`. Using `pip` there fails
with `command not found` and the container looks healthy while `msft` is missing.

**The repo is bind-mounted, not copied.** Editing on the laptop + `git pull` on the box changes
the code the container runs, but the worker process imported `msft` at start — restart the
container (`docker compose restart worker`) after changing worker code.

**One job at a time is a feature.** Two 4B trainings on a 16 GB card is an OOM. The queue
serialises; do not "parallelise" by queueing two trains.

## Models and downloads

**HF drops shards on this link and `from_pretrained` does not always resume.** The 4B fell from
5.8 GB back to 4.2 GB mid-download. Use the detached `snapshot_download(..., resume_download=True)`
loop in `docs/handoff.md` §5 and verify with `ls …/incomplete` (must be empty).

**Expected sizes:** 0.8B 1.7 GB, 4B 9.4 GB, Gemma 4 12B ~23 GB. At 2–5 MB/s the 12B is an hour
of babysitting, not a background step.

**Both target families are ungated** (`gated=False`), so no HF token is required.

**Do not run QLoRA on Qwen3.5.** Unsloth documents 10 GB for bf16 LoRA at 4B and discourages
QLoRA there. Plain LoRA only, never DoRA — DoRA adapters are not GGUF-convertible.

## Training

**A CPU run is never legitimate here.** The generated script now refuses when CUDA is missing
or when the model lands on `cpu`, prints the device and its VRAM, and the worker prints a GPU
line at start and in `status`. Real VRAM: 0.8B ≈ 2 GB, 4B ≈ 13.4 GB.

**Timings are dominated by loading and tokenising, not by the steps.** A 4B run over 49 pairs
takes 166 s, of which ~80 s is training. Tokenising ran at ~3.3 examples/s (CPU, `num_proc=8`) —
that is the CPU load to expect, not a fallback.

**49 paragraph pairs cannot teach a protocol.** The 0.8B and 4B adapters both reproduced the
base model's habits rather than the `mt` format. Mine `MT_TRANSLATE_PROMPT_LOG` and `.mt`
checkpoints for protocol-faithful triples before drawing conclusions about a model size.

## Merge and quantize

**`save_pretrained_merged` writes a tokenizer config that breaks the GGUF converter.** It emits
`tokenizer_class: Qwen3_5Tokenizer` (a class transformers cannot import) and can emit
`extra_special_tokens` as a list where newer transformers wants a mapping. `merge.py` now
normalises the class to the one that actually loaded, converts a list to an indexed mapping, and
fails the merge if `AutoTokenizer.from_pretrained(merged_dir)` does not load.
→ If it still fails, copy the base model's `tokenizer.json` / `tokenizer_config.json` /
`vocab.json` / `merges.txt` over the merged directory and re-run quantize.

**`out/merged` used to be shared by every config.** Quantizing after a 0.8B run under the 4B
config produced a 516 MB file named `model-4b.gguf` — a wrong artifact with a right name and
`rc=0`. The merged directory is now `out/merged-<config name>`.
→ Sanity check every GGUF by size: Q4_K_S is ~30 % of f16, and a 4B Q4 file is ~2.5 GB, never
~500 MB.

**llama.cpp must exist at `LLAMA_CPP_DIR`.** Compose sets `LLAMA_CPP_DIR=/opt/llama.cpp`; the
image builds `llama-quantize` and installs `requirements-convert_hf_to_gguf.txt`. Building it
inside a running container is a temporary fix that the next recreate undoes.

## Serving with ollama

**Never hand-write a chat template.** The GGUF carries the tokenizer's own `chat_template`, and
`ollama` renders it. This repo shipped a hand-copied **Gemma `<|turn>`** template to a Qwen3.5
model; prompted in a format it never saw, it answered with `Thinking Process: 1. **Analyze the
Request:** …` instead of a translation. `scripts/ollama-export.sh` now omits TEMPLATE unless
`TEMPLATE_FILE=` is given.

**Qwen3.5 thinks by default (`enable_thinking: true`).** Even with the right template it
narrates reasoning unless the request sets `"think": false`. With that flag the same 2.6 GB file
returns `Ночь была холодной, и дом стоял в тишине.` and `msft eval` goes green (6 gates).
→ Any pipeline integration must send `think: false`; it is a request field, not a Modelfile
default.

**Push needs a namespace you own.** A bare name targets `library/` and is refused with
`you are not authorized to push to this namespace`.
```bash
docker exec ollama ollama cp machinum-book-ru:4b <user>/machinum-book-ru:4b
docker exec ollama ollama push <user>/machinum-book-ru:4b
```
The account in use is `boris86496`. The ollama container publishes `11434` on host port
**7869**, and its CLI has no `--system`/`--raw` flags — use `/api/chat` over HTTP for anything
non-trivial.

**`OLLAMA_API_KEY` lives in the container environment.** Never `printenv` it into a log; use
`docker exec ollama printenv OLLAMA_API_KEY` only inside a command substitution that does not
echo it. It was exposed once today — rotate it if any output was kept.

## Worker

**`msft data` is not a command.** Job kinds map to actions: `data → data build`, `train`,
`merge`, `quantize`, `eval`. A missing mapping fails in 0.0 s with `rc=2`.

**`worker status` reads the queue; the heartbeat is written between jobs.** A long job shows
`state idle` while it is running, so read `running/` or the job log, not just the heartbeat.

**Jobs are plain JSON in the queue.** Editing `<dir>/inbox/<id>.json` before it is picked up is
a supported way to tweak a run; `worker reset` clears the queue and keeps the logs.

## Guard and CI

**`scripts/guard.sh` is the merge gate** (`ruff` → `pytest` → `wrkflw validate` → `wrkflw run
--runtime emulation`). Never merge red. Exit codes: 0 ok, 1 a step failed, 2 usage/tooling,
3 dirty tree, 4 `--require-ci` without `wrkflw`.

**The emulated CI inherits the caller's shell** unless pinned: the guard puts `.venv/bin` first
on `PATH` so the workflow's `python`/`pip` are the project venv, and sets `PIP_TARGET` to a temp
dir so `pip install -e .` cannot overwrite the editable install (without it `import msft` breaks
and points at a deleted sandbox).

**`wrkflw` 0.8.0 ignores `.gitignore` `!` re-includes.** A `*.jsonl` rule plus
`!tests/fixtures/*.jsonl` silently removes tracked fixtures from the emulated checkout. The pair
was removed; keep it that way, or expect a red `ci-run` that is not your bug.

**`.gitignore` must not hide fixtures.** `*.pt` is ignored, `tests/fixtures/*` is tracked.
