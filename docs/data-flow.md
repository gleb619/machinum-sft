# Data flow: pipeline → dataset → model

Two repos, one seam: **JSONL of chat triples**.

```
machinum-pipeline                              machinum-sft
─────────────────                              ────────────
books/*.md  ──► parallelPairs (TL255)  ─┐
.mt/* + logs ─► sftTriples    (TL254)  ─┼─► data/raw/*.jsonl ──► msft data build ──► out/dataset/{train,val}.jsonl
                                        ├─► datasetSplit  (TL256)                          │
                                        └─► datasetCard   (TL257)                          ▼
                                                                                    msft train (QLoRA)
                                                                                           │
                                                                            msft merge → msft quantize
                                                                                           │
                                                                                    out/model.gguf → mt.json
```

## The record contract

Every line of `data/raw/*.jsonl` is one object:

```json
{"system": "...", "user": "...", "reply": "...", "source": "book1/chapter-1.en.md#p3", "prompt_hash": "…"}
```

or the equivalent canonical form

```json
{"messages": [{"role": "system", "content": "…"}, {"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}], "source": "…"}
```

`msft data build` accepts either, plus `.json` files containing a list of such objects. It
normalises to `messages`, drops records without a user or assistant turn, drops replies that
are only a fence or whitespace, filters by `min_chars`/`max_chars`, dedupes by SHA-256 of the
normalised messages, and splits deterministically by hash (never at random) into
`train.jsonl` / `val.jsonl` plus `stats.json`.

## Why triples and not just parallel text

The model has to emit a *protocol*: XML blocks for proofreading, `N=text` properties, one
`<terms>` JSON block, one output line per input line. A parallel book pair teaches the
translation; the `system` field is what binds the behaviour to the protocol. Both come from
the same pipeline run, so keep the `system` text identical to what `mt` sends at inference —
the trainer applies the model's chat template to exactly these three roles.

## Producing the first dataset by hand

Until the pipeline step is wired into a `definePipeline` chain, the same shape can be built
from the repo's human-parallel chapters:

```bash
python scripts/pairs_from_book.py \
  --en ../machinum-pipeline/books/book1 \
  --out data/raw/book1-parallel.jsonl     # paragraph-level EN→RU pairs, one triple each
msft data build --config configs/gemma-4-12b-qlora.yaml --execute
msft data build --config configs/gemma-4-12b-qlora.yaml --json   # same numbers, no writes
```

`scripts/pairs_from_book.py` is deliberately dumb: paragraph split, drop headings, keep
pairs where both sides are non-empty. The pipeline's TL255 does the same with alignment
scoring and a 60 % alignment floor — use it once the run is wired.
