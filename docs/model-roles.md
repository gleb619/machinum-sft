# Model roles: `misc` and `translation`

Two models, two jobs, one box. A chapter of book prose is translated by the **translation**
model; everything else the pipeline does — summarisation into markdown bullets, NER / `<terms>`
extraction, glossary filling, proofreading and typo fixing, normalisation, short protocol
answers — runs on the **misc** model, the `machinum:4b` family this repo builds. A job that
needs both chains them in that order: translate on the big model, post-process on the 4B.
Neither model is asked to do the other's job.

The tag is a **deployment choice, not a property of the code**. It is written as a copyable
block once, under "How to wire it" below; every other page refers to a model by role and tag.

| role | tag | quant + size | resident VRAM | measured cost | what it is for | owned by |
| --- | --- | --- | --- | --- | --- | --- |
| `misc` | `machinum:4b` family — `machinum:4b`, `-v2-data`, `-v2-tuned`, `-heretic` | Q4_K_S, 2.63 GB | 3.2 GB at 8 192 ctx | ~40 s per chapter; 28–33 s on one 33-line chapter | bullet summaries, NER / `<terms>`, proofreading and typo fixing, normalisation, short protocol answers | this repo: built, trained, merged and quantised here |
| `translation` | `igorls/gemma-4-12B-it-heretic-GGUF:latest` | Q4_K_M, 7.4 GB | 8.4 GB at 32 768 ctx | ~60 s per chapter | EN→RU literary translation of book prose — and only that | an external GGUF served on the same box; this repo does not train it and must not imply it does |

Sizes, VRAM and per-chapter costs are `docs/model-candidates.md` ("How the two compare"); the
28–33 s figure is the 33-line chapter in `docs/pipeline-check.md`; the tag list and digests of
the 4B family are `docs/model-tags.md`.

## How to wire it

The model is chosen per **process invocation**, not per stage. A pipeline passes no backend —
"each LLM tool resolves the process backend at run time"
(`samples/sample2-practical/pipelines/preparation.ts:34`) — so the pipeline that wires the tool
decides which stage gets which instance. One block per role:

```bash
# misc — the 4B built here: summaries, NER / <terms>, glossary, preprocess, proofread
export MT_LLM_TOOL=ollama
export MT_OLLAMA_MODEL=machinum:4b
export MT_OLLAMA_URL=http://<gpu-host>:7869/v1
export MT_LLM_REASONING_EFFORT=none

# translation — the external 12B: EN->RU book prose, and only that
export MT_LLM_TOOL=ollama
export MT_OLLAMA_MODEL=igorls/gemma-4-12B-it-heretic-GGUF:latest
export MT_OLLAMA_URL=http://<gpu-host>:7869/v1
export MT_LLM_REASONING_EFFORT=none
```

Four things to check before copying, because the knobs are not the obvious ones:

- **`MT_OLLAMA_MODEL`, not `MT_LLM_MODEL`, selects the model on the ollama backend.**
  `resolveLlmConfig` resolves the model as input > tool > `meta.llm` > mt.json
  `settings.llm.model` > provider env (`MT_OLLAMA_MODEL`) > preset defaults > generic env
  (`MT_LLM_MODEL`) — `packages/tools/src/core/llm-config.ts:342-350`, and the ollama preset
  ships `gemma4:latest` as its own default (`packages/tools/src/core/ollama-tool.ts:13`), which
  therefore outranks a bare `MT_LLM_MODEL`.
- **`MT_LLM_MODEL_<STAGE>` is not on the product path.** `docs/roadmap.md:115` lists
  `MT_LLM_MODEL_SUMMARY|NER|PREPROCESS|TRANSLATE`, but the tools read only the flat `llm.model`,
  never `llm.model.<stage>` (`packages/tools/src/core/llm-config.ts:135-142`). Per-stage routing
  is the pipeline's job — a follow-up there, not in this repo.
- **`MT_LLM_REASONING_EFFORT=none` is mandatory.** On the OpenAI-compatible endpoint ollama
  ignores `think: false`; a thinking model then returns `content: ""` with the budget spent in
  `reasoning` (`docs/memory.md`, "`think: false` does nothing on the OpenAI-compatible
  endpoint").
- **`MT_OLLAMA_URL` must end in `/v1`** and, on this box, carries host port `7869`
  (`docs/memory.md`). Without `/v1` every call is a 404 from a server that answers `/api/tags`.

## Changing the translation model

One edit: the `MT_OLLAMA_MODEL` line in the block above — or the same value as
`settings.llm.model` in the pipeline's `mt.json`. Nothing in this repo names it, so no code, no
config and no test here changes.

Then re-measure three things before believing the swap, in this order:

1. **Protocol gates** — `msft eval --predictions <recorded replies>`: `no_chatter`,
   `line_parity`, `schema_validity`. A model that fails one of these is a retry loop, not an
   upgrade.
2. **Line parity on a real chapter** — the pipeline's `chapterPairsLinter` plus
   `scripts/audit_translation.py`, which counts parity failures, metadata leaks, empty lines
   and foreign-script tokens over a finished book.
3. **tok/s and wall** — `msft arena` prints `tok/s`, `speed_index`, `p50 s`, `worst s` and
   `load s` side by side. A per-chapter time without the endpoint and the context size is not
   a number (`docs/arena.md`).

## What has not been measured

The 12B is a serving decision, not a measured artefact of this repo:

- **No `msft eval`.** There is no schema-validity, line-parity or chatter rate over a recorded
  reply set, so no gate result may be claimed for it.
- **No `msft arena`.** `docs/arena-cases.example.jsonl` holds 11 cases and none of them has
  been sent to this tag here: there is no BLEU, chrF, glossary, judge or `tok/s` number for it
  in this repo.
- **No wall-time table of its own.** The ~60 s per chapter is the single held-out chapter in
  `docs/model-candidates.md`, not a book-length run.

What does exist: that held-out chapter (book2 ch.1) at 17/17 lines, 0 metadata leaks, 0
foreign characters, Cyrillic share 0.79, and the three semantic examples — licorice drops,
the boats, the reel in the tin — all in `docs/model-candidates.md`. Quote those; do not derive
a quality score from them.

## See also

- [model-tags.md](model-tags.md) — every `machinum:*` tag: corpus, hyperparameters, digest.
- [model-candidates.md](model-candidates.md) — the two models side by side, and every
  discarded candidate with the measurement that eliminated it.
- [pipeline-check.md](pipeline-check.md), [score-improvement.md](score-improvement.md),
  [arena.md](arena.md) — 4B-as-translator records from before the split: the numbers stand,
  the role has changed.
