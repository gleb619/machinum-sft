# Training on the remote box — the working loop, and what bites

Everything here is measured on the 2026-11-08 run (two RTX 4060 Ti 16 GB, one container).
Read `docs/memory.md` before changing anything: it is the same knowledge, organised as traps.

## 1. Shape of the system

```
laptop <repo>/          <- the only source of truth, git + guard + tests
        |  git push
        v
box    ~/WORKSPACE/machinum-sft               <- git clone, bind-mounted into the container
       msft-worker container (image machinum-sft:cu128)
         /workspace  = that clone (read-write: the worker writes out/ and .worker/)
         /hf         = HF cache (a download survives a container recreate)
         /opt/llama.cpp, /opt/venv          <- inside the IMAGE, not the container
       ollama container  :7869 on the host, models live in its own volume
```

The worker is a **directory queue**, not a daemon that runs git: it picks `inbox/<id>.json`,
moves it to `running/`, runs one job at a time, writes `done/<id>.json` + `logs/<id>.log`,
and keeps `heartbeat.json` fresh for the compose healthcheck. One job at a time is deliberate:
two trainings on one 16 GB card is an OOM, not a speed-up.

## 2. First-time setup on a fresh box

```bash
# on the box
mkdir -p ~/WORKSPACE && cd ~/WORKSPACE
git clone https://github.com/gleb619/machinum-sft.git && cd machinum-sft
mkdir -p .worker

# on the laptop: configs/worker.local.yaml is git-ignored, the host lives only here
cp configs/worker.example.yaml configs/worker.local.yaml   # then edit `host:`

# on the laptop
msft remote deploy --execute       # pull on the box, build the image, compose up
msft remote status --execute       # expect: state serving/idle and a GPU line
```

The first build is ~10–12 GB of CUDA + torch + unsloth, so give it 10–20 minutes and do not
interrupt it. `msft remote status` must print a line like
`GPU: 0, NVIDIA GeForce RTX 4060 Ti, 16380 MiB | 1, …`; if it prints a `WARNING`, stop and fix
that before queueing anything.

## 3. The daily loop

```bash
# laptop
$EDITOR src/msft/whatever.py
bash scripts/guard.sh                 # ruff -> pytest -> wrkflw validate -> wrkflw emulated CI
git commit -am "..." && git push

# queue work on the box (it pulls the commit itself)
msft remote signal train    --config configs/qwen3.5-4b-lora.yaml --execute
msft remote signal merge    --config configs/qwen3.5-4b-lora.yaml --execute
msft remote signal quantize --config configs/qwen3.5-4b-lora.yaml --execute
msft remote status --execute          # state, queue counts, GPU line
msft remote log <job-id> --execute    # the job's log
```

Jobs run in the order they were queued and each one is a fresh process, so a failure costs
only that stage. Re-queue with a new `--id` after fixing the cause; `worker reset` clears the
queue but keeps the logs.

## 4. What each stage actually costs (measured)

| Stage | 0.8B | 4B |
| --- | --- | --- |
| `data build` | < 5 s | < 5 s |
| `train` (49 pairs, 2–3 epochs) | 167 s total | 166 s total (80 s of steps, the rest is loading 9.4 GB + tokenising) |
| VRAM | ~2 GB | 13.4 GB |
| `merge` | 32 s → 1.7 GB | 83 s → 11 GB |
| `quantize` | 9 s → 516 MB | ~5 min → 2.63 GB (Q4_K_S ≈ 30 % of f16) |
| `ollama-export` | ~10 s | ~1 min for 2.6 GB |

The 166 s is not the training. If a 4B job takes 10 minutes, look at the log timestamps before
blaming the GPU: `Loading weights` and `Tokenizing` dominate at this dataset size.

## 5. Downloads are the real bottleneck

On this link HF moves at 2–5 MB/s and **drops shards**. `from_pretrained` alone left the 4B at
5.8 GB then silently fell back to 4.2 GB. Use a resumable, retrying pull detached:

```bash
docker exec -d msft-worker python -c "
from huggingface_hub import snapshot_download
import time
for attempt in range(1, 21):
    try:
        print(snapshot_download('Qwen/Qwen3.5-4B', max_workers=2, resume_download=True)); break
    except Exception as e:
        print('attempt', attempt, e); time.sleep(15)
"
# check it without guessing
du -sh /hf/hub/models--Qwen--Qwen3.5-4B          # expect 9.4G for the 4B, 23G for 12B
ls /hf/hub/models--Qwen--Qwen3.5-4B/incomplete   # must be empty when finished
```

Both Qwen3.5 and Gemma 4 are **not gated** (`gated=False`), so no HF token is needed.

## 6. Picking a config

| Config | Model | VRAM | When |
| --- | --- | --- | --- |
| `smoke-qwen3.5-0.8b.yaml` | 0.8B | ~2 GB | after touching the training path — 3 minutes, proves the stack |
| `qwen3.5-0.8b-lora.yaml` | 0.8B | ~2 GB | pipeline tests, GGUF/ollama plumbing |
| `qwen3.5-4b-lora.yaml` | 4B | 13.4 GB | the smallest model worth judging on translations |
| `gemma-4-12b-qlora.yaml` | 12B | ~20 GB | only with a stable link; 23 GB to download |

Rules that held: plain LoRA, never DoRA (not GGUF-convertible); `completion_only_loss: true`;
`lm_head` in `modules_to_save`; bf16 LoRA on Qwen3.5 rather than QLoRA (Unsloth discourages
QLoRA there and documents 10 GB at 4B).

## 7. From weights to a servable model

```bash
msft remote signal merge    --config … # out/merged-<config name>, 16-bit
msft remote signal quantize --config … # out/model-*.gguf via LLAMA_CPP_DIR
bash scripts/ollama-export.sh out/model-4b.gguf machinum-book-ru:4b
```

Two things decide whether the result is usable, and neither is the weights:

1. **The chat template must come from the GGUF.** It carries the tokenizer's own
   `chat_template`; a hand-copied one (this repo shipped Gemma's `<|turn>` format to a Qwen
   model) makes the model narrate its reasoning instead of answering.
2. **`think: false` on every request.** Qwen3.5 defaults to thinking. Through the API:

```json
{"model": "machinum-book-ru:4b", "think": false, "stream": false,
 "messages": [{"role": "system", "content": "Translate English prose into Russian. Reply with the translation only."},
              {"role": "user", "content": "The night was cold and the house stood silent."}]}
```

With both, `msft eval` goes green: `Ночь была холодной, и дом стоял в тишине.`
Without them, the same file answers with `Thinking Process: 1. **Analyze the Request:** …`.

Push to a namespace you own — a bare name means `library/` and is refused:

```bash
docker exec ollama ollama cp machinum-book-ru:4b <user>/machinum-book-ru:4b
docker exec ollama ollama push <user>/machinum-book-ru:4b
```

## 8. Failure playbook (all of these happened)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ImportError: transformers 4.57.6 does not support Qwen3.5` | the upgrade lived in the container image, not the container's layer | `uv pip install "transformers>=5.2.0"`, then bake it into `deploy/remote/Dockerfile` |
| `convert_hf_to_gguf.py` or `llama-quantize` missing | `/opt/llama.cpp` was built in the old container and lost on recreate | rebuild in the image (now in the Dockerfile), `LLAMA_CPP_DIR=/opt/llama.cpp` |
| `Tokenizer class Qwen3_5Tokenizer does not exist` | `save_pretrained_merged` wrote a class transformers cannot import | `merge.py` normalises `tokenizer_class` to the loaded class and fails if it does not load |
| `'list' object has no attribute 'keys'` in the converter | `extra_special_tokens` saved as a list | copy the base model's `tokenizer_config.json` over the merged one |
| job `rc=2` in 0.0 s | the worker called `msft data` without the `build` action | fixed in `build_command`; keep `tests/test_worker.py` green |
| a training job silently uses the CPU | no CUDA in the container, or the model placed on `cpu` | the generated script now refuses and prints the device; `MSFT_ALLOW_CPU=1` overrides |
| GGUF is the wrong size / wrong model | two configs merged into one `out/merged` | `out/merged-<config name>` per run (fixed in `config.py`) |
| `ollama push` → "not authorized to push to this namespace" | bare model name targets `library/` | `ollama cp` to `<user>/name` first |
| guard red on a clean tree | `ci-run` inherited the caller's `python`/`pip` | guard prepends `.venv/bin` to `PATH` for the CI steps |

## 9. Habits that would have saved most of the day

- After **any** compose edit, assert the tools are still there:
  `docker exec msft-worker python -c "import transformers;print(transformers.__version__)"` and
  `ls /opt/llama.cpp/build/bin/llama-quantize`. A recreate is a silent revert.
- Read the **timestamps** in the job log before blaming hardware; loading and tokenising are
  CPU-bound and dominate at 49 examples.
- Never trust an artifact because a job said `rc=0`: `ls -la out/*.gguf` and compare sizes —
  516 MB for a 4B model is a 0.8B model with the wrong name.
- Run the smoke config for 3 minutes after touching train/merge/quantize. It has caught every
  regression on this path so far.

## Which way to run a stage

`remote run` has three modes and they are not interchangeable. Queue is the default for anything
GPU-heavy; tmux is the escape hatch; foreground is for a short job you want to watch.

| | foreground | `--tmux` | `--queue` |
|---|---|---|---|
| survives a dropped link | no (detected in ~2 min) | yes | yes |
| survives a box reboot | no | no (`lost`) | marked failed, never requeued |
| serialised against other GPU jobs | no | guard only | yes |
| records rc / duration / status | no | rc file | `done/<id>.json` |
| attach and debug live | no | yes (`tmux attach`) | no |
| needs the worker container up | yes | yes | yes |

```bash
msft remote run --config configs/qwen3.5-4b-lora.yaml --execute              # foreground, streams
msft remote run --tmux  --config configs/qwen3.5-4b-lora.yaml --execute      # detached on the box
msft remote logs --tmux <session> --wait --execute                           # block on its exit code
msft remote stop --tmux <session> --execute                                  # SIGTERM, then kill

msft remote run --queue --config configs/qwen3.5-4b-lora.yaml --timeout 2h --execute
msft remote logs --job <id> --wait --execute
```

There is no per-job cancel for the queue: `worker reset` or stopping the container are the tools.
