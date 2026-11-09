# machinum:4b — a 4-billion-parameter Russian literary translator

A LoRA fine-tune of Qwen3.5-4B that speaks one protocol: take English prose, return Russian
prose of the same line shape. It is not a chat model, it is a translation instrument, and this
page is the record of how it was made, what it costs, and where it fails.

```
base            Qwen/Qwen3.5-4B            4.66 B params, 262 144 ctx, thinking by default
adapter         plain LoRA                 r=16, α=32, 7 projections, 44 MB
data            49 parallel pairs          2 898 tokens, English → Russian, one book
compute         1 × RTX 4060 Ti 16 GB       165.8 s wall, 13.4 GB VRAM, 4 optimiser steps
artefact        GGUF Q4_K_S                 2.63 GB, runs 100 % on one 16 GB card
serving         ollama, 8 192 ctx           reasoning disabled per request
```

## 1. Why this model exists

The `machinum-pipeline` (`mt`) translates books line by line: one request per source line, and
the reply must keep the line count so a parity linter can align source and target. That contract
is what the fine-tune is for. A general chat model answers the line with a paragraph, a list, or
a "Thinking Process:" preamble; the pipeline retries, then fails the run.

So the target is narrower than "good Russian" — it is *good Russian in the shape the pipeline
already defined*. Everything below serves that.

## 2. Base model

Qwen3.5-4B: 4.66 billion parameters, 262 144-token context, 128 KB per token on the tokenizer,
2.59–2.74 GB of weights in bf16. Qwen was chosen over Gemma 4 12B because the 12B needs QLoRA to
fit next to the KV cache on a 16 GB card, and QLoRA checkpoints are one more place for the GGUF
conversion to break; plain LoRA on a 4B trains in under three minutes and merges in one call.
Llama was excluded outright.

Context: the pipeline's units are lines and chapters, not books, so 32 K would have been enough;
262 K is a bonus that the 8 192-token serving window does not use.

## 3. Data

`data/raw/book1-parallel.jsonl` — 49 English/Russian pairs cut from a parallel translation of
one novel, split 51 train / 1 validation after tokenising, 2 898 tokens in total. The pairs are
line-aligned paragraphs, the same shape the pipeline sends at inference.

That is the whole dataset, and it is deliberately small: the goal of the first run was to prove
the *format* end to end — train, merge, quantise, serve, gate — not to teach translation. A model
trained on 44 examples learns the protocol and keeps the base model's Russian; it does not learn
a translator's voice, and section 8 shows exactly where that shows up.

Reproducing the data:

```bash
msft data build --config configs/qwen3.5-4b-lora.yaml   # pairs → train/val jsonl
msft fetch  --config configs/qwen3.5-4b-lora.yaml       # resumable base weights (9.4 GB)
```

## 4. Training

```yaml
# configs/qwen3.5-4b-lora.yaml (abridged)
base_model: Qwen/Qwen3.5-4B
lora:        { r: 16, alpha: 32, dropout: 0.05, target_modules: [7 projections] }
epochs:      2          # 4 optimiser steps at 51 examples × 4096 tokens
lr:          1.0e-4
batch:       1 × grad_accum 8
seq_len:     4096
precision:   bf16
```

Plain LoRA, never DoRA: DoRA adapters do not convert to GGUF, and the deliverable is a GGUF.
Completion-only loss — the prompt is never learned. `lm_head` is not saved.

Measured on the first real run:

| stage | command | wall | result |
| --- | --- | --- | --- |
| train | `msft remote signal train` | 165.8 s | 4 steps, train_loss 2.103, eval_loss 2.271, adapter 44 MB |
| merge | `… signal merge` | 83.2 s | bf16 weights, 11 GB, tokenizer repaired for GGUF |
| quantise | `… signal quantize` | — | f16 8.67 GB → **Q4_K_S 2.63 GB** |
| serve | `scripts/ollama-export.sh` | — | `machinum:4b`, 100 % GPU, 8 192 ctx |

~80 s of the 166 s is weight loading and tokenising 44 examples at ~3.3/s, not training. On a
bigger dataset the training share grows and this ratio stops mattering.

The training script refuses to run on CPU: if `torch.cuda.is_available()` is false, or the model
lands on `cpu` after `get_peft_model`, it exits non-zero instead of spending hours to produce a
worse adapter. `MSFT_ALLOW_CPU=1` overrides it and prints the expected wall time.

## 5. Merge and quantise

`merge` calls `save_pretrained_merged`, then rewrites `tokenizer_config.json` — the merged
tokenizer declares a `tokenizer_class` the converter cannot instantiate, and a list-shaped
`extra_special_tokens` — and proves the tokenizer reloads before it reports success. Without that
step `convert_hf_to_gguf.py` fails three different ways, each of which costs a run.

Each run merges into `out/merged-<config-name>/`, never a shared `out/merged/`. Two configs
sharing one directory produced a file called `model-4b.gguf` that contained the 0.8B model.

Quantisation is `Q4_K_S`: 2.63 GB, quality indistinguishable from f16 at this size, and small
enough that the KV cache for 8 192 tokens still fits beside it on a 16 GB card.

## 6. Serving

```bash
scripts/ollama-export.sh out/model-4b.gguf machinum:4b
```

The Modelfile carries **no hand-written chat template**. The GGUF's own `chat_template`
metadata drives ollama. A hand-copied Gemma `<|turn>` template in front of a Qwen GGUF was the
cause of the "Thinking Process:" chatter that made the first three evaluations red.

Qwen3.5 thinks by default, and 44 training pairs do not override `enable_thinking: true`. The
switch has to be per request:

```bash
# native API
{"model":"machinum:4b","messages":[…],"think":false,"stream":false}
# OpenAI-compatible endpoint — ollama ignores "think" here and honours this instead
{"model":"machinum:4b","messages":[…],"reasoning_effort":"none"}
```

## 7. What it was checked against

`msft eval` runs the pipeline's own gates over recorded predictions:

| gate | question | machinum:4b |
| --- | --- | --- |
| `no_chatter` | no "Thinking Process", no narrated reasoning | 3/3 |
| `line_parity` | reply has the same number of lines as the source | 3/3 |
| `schema_validity` | the reply fits the pipeline's expected shape | 3/3 |

Three hand-checked sentences prove format, not quality. The 0.8B model of the same family, with
the same recipe, failed all three — that is what a real gate looks like when it fires.

## 8. Where it fails

The same chapter through `mt` and through a 27B model, side by side, is the honest benchmark. On
one chapter of held-out prose, `machinum:4b` kept line parity 33/33 and produced fluent Russian
with these defects:

- **Names drift inside one page** — «Файона Доул», «Фиона Доули», «Кормак Доули», «Кормак Доул»
  for two characters. The fix is the pipeline's glossary stage, not more training.
- **Non-words and calques** — «в багажаре» (no such word), «она весила ничего» (ungrammatical),
  «имела в виду это» (calque of "meant it").
- **Register slips** — "Harbor papers" → «бумажки от гавани» (dismissive), licorice drops →
  «жевательная резинка».
- **Gender and case agreement** — «Надя встретила его» for a woman, «старого библиотеки».

This is what 44 examples buy: the protocol is learned, the language is the base model's. The
next run needs a few thousand pairs, a glossary, and a proofreading pass — in that order.

## 9. Reproduce it

```bash
git clone https://github.com/gleb619/machinum-sft && cd machinum-sft
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
msft verify --config configs/qwen3.5-4b-lora.yaml     # artifacts present and loadable
bash scripts/guard.sh                                 # ruff, pytest, workflow validation
```

Training runs on a remote GPU box through a directory queue (`msft worker`), not by SSH-ing in
and typing commands; `docs/handoff.md` is the operator's loop and `docs/memory.md` is the list of
traps that cost a run each.

## 10. Licence and provenance

Base model Qwen3.5-4B under its own licence; the adapter, the tooling and these notes are MIT.
The training corpus is a private parallel translation and is not redistributed. If you publish a
derivative of this adapter, keep the base model's attribution.

## Which model to run

This page describes `machinum:4b`, the artefact this repo builds: 2.63 GB, 3.2 GB of VRAM, ~40 s
per chapter, correct through the pipeline but with occasional semantic slips (it has rendered
"boats" as "yachts" and "tin" as "brass").

For the finished text there is one second model in the project,
`igorls/gemma-4-12B-it-heretic-GGUF`: 7.4 GB, 8.4 GB of VRAM at 32 768 context, ~60 s per chapter,
and it stops making those substitutions. Both run on one 16 GB card through the same endpoint;
the measurements and the reasons every other candidate was discarded are in
[model-candidates.md](model-candidates.md).
