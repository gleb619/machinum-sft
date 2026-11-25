# machinum-sft

The project runs two models with different jobs. This repo builds and trains the **`misc`**
model (`machinum:4b`): bullet summaries, NER / `<terms>` extraction, proofreading and typo
fixing, normalisation and short protocol answers against a strict line/XML protocol. EN→RU
literary translation of book prose — and only that — runs on an external 12B GGUF served on the
same box; this repo does not train it. Roles, tags and the wiring block are in
`docs/model-roles.md`.

The point of this repo is a **repeatable, boring** run: dataset in, adapter out, GGUF out,
gates green. Everything that touches hardware, paths or secrets is configuration; nothing is
hardcoded.

**Model card (public):** <https://gleb619.github.io/machinum-sft/> — how `machinum:4b`, the
misc model, was built, what it costs and where it fails. `docs/pipeline-check.md` records it
running through the real pipeline, and `docs/model-candidates.md` compares it with the
`translation` model (`igorls/gemma-4-12B-it-heretic-GGUF:latest`) and says why everything else
was discarded.

```bash
uv venv --python 3.11 && . .venv/bin/activate
uv pip install -e ".[dev]"
python -m msft --help
python -m msft doctor
```

## What it does

| Command | What it is for |
| --- | --- |
| `msft doctor` | env/GPU/VRAM/disk/tokenizer check before you burn hours |
| `msft data build` | mine the pipeline's JSONL artifacts into chat triples, filter, dedupe, split |
| `msft train` | QLoRA/SFT via Unsloth + TRL; `--dry-run` prints the exact plan and script |
| `msft merge` | merge the adapter into 16-bit weights |
| `msft quantize` | GGUF conversion, optional imatrix calibration, quantise to Q4_K_S/Q5_0 |
| `msft eval` | protocol gates: parse rate, line parity, schema validity, chatter rate |
| `msft remote` | run the same config on a remote GPU box over ssh (host comes from `.env`) |

## Design rules

- **Core modules are stdlib-only** (+ PyYAML). Heavy deps (`torch`, `unsloth`, `trl`) live in
  the `train` extra and are only imported inside the code paths that need them.
- **`--dry-run` is the default for anything that trains or downloads.** You opt into work.
- **No secrets, no hosts in the tree.** `.env` is git-ignored; `.env.example` shows the keys.
- **Tests never touch the network or a GPU.**

See `CONTRACT.md` for the module interfaces, and `docs/model-choice.md` for the base-model
decision this repo follows. `docs/data-flow.md` explains how the pipeline's dataset tools
hand JSONL to `msft data build`.

## Start here if you are picking this up cold

| Document | Why |
| --- | --- |
| `AGENTS.md` | the rules — read before changing anything that costs GPU time |
| `docs/memory.md` | every trap that cost time, with its fix and the command that proves it |
| `docs/handoff.md` | the working loop: setup, daily commands, stage timings, failure playbook |
| `docs/run-2026-11-08.md` | one real run end to end, with numbers |
| `docs/memory-stack.md` | the one-container project journal — start it, seed it, read it |
| `docs/model-roles.md` | the two roles — `misc` on the 4B, `translation` on the 12B — and the env block |
| `docs/index.md` | model card: how `machinum:4b` was built, what it costs, where it fails |
| `docs/pipeline-check.md` | the model through `mt` on held-out prose, with the defects quoted |
| `docs/model-candidates.md` | the models the internet recommends, checked against the registry |
| `docs/material-pipeline.md` | the daily research loop: what it collects, where the notes go, why the raw material is never committed |
| `docs/arena.md` | `msft arena`: the same prompts through several models, scored by independent frameworks, with the honesty rules |
| `docs/model-tags.md` | every `machinum:*` tag: corpus, hyperparameters, digests, and which one to run |

## Remote worker

The GPU box is a queue, not a second codebase. Code is committed and pushed here; the box
pulls; a container serves jobs.

```bash
cp configs/worker.example.yaml configs/worker.local.yaml   # host goes here; git-ignored
msft remote deploy --execute        # pull on the box + docker compose up --build
msft remote signal train --execute  # queue a job
msft remote status --execute        # container health + queue + heartbeat
msft remote down --execute          # stop it
```

Inside the box the same CLI is used locally:

```bash
msft worker serve --dir .worker             # the daemon (this is what the container runs)
msft worker signal train --config configs/gemma-4-12b-qlora.yaml
msft worker status --dir .worker
msft worker log --dir .worker --lines 60
```

The queue is a directory of JSON files (`inbox` → `running` → `done`), each job carrying its
kind, config and result. The daemon writes `heartbeat.json` every loop, which is what the
compose healthcheck reads.

## Calling the exported model

Two models, one endpoint, two requests. Both GGUFs carry their tokenizer's own chat template,
and both families think by default — a request without the flag comes back as a reasoning
transcript instead of the answer.

Translation, on the `translation` model (external 12B):

```json
{"model": "igorls/gemma-4-12B-it-heretic-GGUF:latest", "think": false, "stream": false,
 "messages": [{"role": "system", "content":
               "Translate the English line into Russian. Reply with the translation only."},
              {"role": "user", "content": "The night was cold and the house stood silent."}]}
```

Misc — here a bullet summary — on `machinum:4b`:

```json
{"model": "machinum:4b", "think": false, "stream": false,
 "messages": [{"role": "system", "content":
               "Summarise the report as three markdown bullets. Reply with the bullets only."},
              {"role": "user", "content": "INCIDENT: the night shift handed over the manifest."}]}
```

On the OpenAI-compatible endpoint ollama ignores `think` and honours
`"reasoning_effort": "none"` instead (`docs/memory.md`); keep one of the two on every request.
The variables that pick the tag, and what the pipeline must do to run the two roles, are in
`docs/model-roles.md`.

## Verified run (2026-11-08)

End-to-end on a 16 GB RTX 4060 Ti box, from a clean Python 3.12 venv:

```
msft data build  →  63 records in, 52 out, 2 898 approx tokens, 51 train / 1 val
msft train       →  42 rows, effective batch 4, 11 steps, 1 epoch
                    train_loss 1.919 · eval_loss 1.144 · 64.6 s on GPU0 (1.9 GB VRAM)
                    adapter_model.safetensors 25.6 MB
```

The smoke config (`configs/smoke-qwen3.5-0.8b.yaml`) uses the same code path as the real
configs — Unsloth `FastLanguageModel` → PEFT → TRL `SFTTrainer` — so a green smoke means the
stack, the generated script and the artifact layout all work. Size and quality claims need
`configs/gemma-4-12b-qlora.yaml` or `configs/qwen3.5-4b-lora.yaml`, not the smoke.

MIT licensed.
