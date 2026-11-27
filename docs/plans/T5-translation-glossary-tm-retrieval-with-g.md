# T5 — translation: glossary/TM retrieval with Giga-Embeddings-480M

> Task as specified: "Spike: embed glossary.json + chapter text with Giga-Embeddings-instruct-480M
> (MIT, 0.97 GB) and pick the per-chapter term subset; measure recall of the curated terms and the
> prompt saving before any pipeline change." The board row carries a 124-char compression of this
> because `at kanban add` caps `Description` at 128 chars; the full text lives here.

## Keywords

Keywords: embeddings, glossary, translation, gigachat, retrieval

## Goal

Decide, with numbers, whether an embedding-based term selector is worth a pipeline change.
Not to ship it.

## Why now

- `Giga-Embeddings-instruct-480M-0826` is MIT, 0.97 GB bf16, 1024-dim, mean pooling + L2
  normalization, instruction on the query side only.
- MTEB-rus 70.98 vs 63.64 for `Qwen3-Embedding-0.6B` — the Russian retrieval gap is the reason
  to look at this model rather than the Qwen embedder already on the box.
- Today `glossary.json` is the only curated term source that reaches the model.
- A full glossary for a long book does not fit the prompt budget.

## Step 0 — find the production caller first

Recorded 2026-11-27 from `/home/boris/WORKSPACE/machinum-pipeline` (read-only):

- **Producer** — `packages/tools/src/modification/book-outputs.ts:50` emits `glossary.json`
  from the `glossaryJson` variable; serializer is
  `packages/tools/src/normalization/glossary-extractor/glossary.ts:309`.
- **Loader** — `packages/tools/src/normalization/translation-stage/slots.ts:12`
  (`loadGlossaryOrEmpty`); reads `chaptersDir/glossary.json` at `:13`, throws on a malformed
  file rather than translating without pinned terms.
- **Prompt consumer** — `packages/tools/src/normalization/translation-prompt.ts:277`
  (`buildTranslationPrompt`). `glossaryBlock` is assembled at `:313` and rendered into the
  `{{glossaryBlock}}` slot of the bundled `translate.system` template.
- **Live callers** — `packages/tools/src/normalization/translation-stage/lines.ts:90`
  (`linePrompt`) and `:209` (block prompt), plus
  `packages/tools/src/normalization/translation-stage/title.ts:56`. This is a live production
  path, not a dead branch.

Observed selection today: `termIdsIn` (`translation-prompt.ts:142`) collects the `<term id>`
tags present in the unit's text, and `:284` keeps only those glossary entries; `locked` terms
are appended unconditionally at `:290`. So the current filter is **tag-occurrence, per unit**,
not semantic — and a book-wide, LLM-free occurrence index already exists at
`packages/tools/src/normalization/glossary-index.ts:230`. Any embedding selector therefore has
to beat a tag filter that is already near-perfect on tagged text, and its real opening is
untagged units and prompt-budget reduction. That is the question the spike must answer with
numbers.

## Steps

- [ ] Fetch the model to the GPU box (host from `configs/worker.local.yaml`; torch lives in the
      `msft-worker` container) or a scratch venv. Never into the repo's `.venv`, never committed.
- [ ] Embed ≥100 glossary entries and the source text of ≥8 chapters; write vectors under
      `out/embeddings-spike/`.
- [ ] Per chapter, rank the glossary by cosine and take top-k for k = 5/10/20/40.
- [ ] Measure **term recall**: of the curated terms that actually occur (morphologically) in that
      chapter's reference translation, how many land in top-k. The whole-glossary baseline is 1.0
      by construction, so the question is how small k can get while recall stays 1.0.
- [ ] Measure the prompt-char delta at each k against the full-glossary prompt.
- [ ] Write `out/embeddings-spike/report.md` (git-ignored) with the table, the delta and a yes/no.

## Constraints

- Spike only: no production code, no schema change, no new dependency in the repo's `.venv`,
  nothing committed outside `docs/kanban.md` and `docs/plans/`; `out/` is git-ignored.
- The operator decides whether a positive result becomes a pipeline change.
- Instruction goes on the query side only; passages are embedded without it, or the comparison
  is not the model's documented use.

## Done when

- `out/embeddings-spike/report.md` has a recall@k table over ≥8 chapters and ≥100 glossary
  entries, the prompt-size delta, and the named caller from Step 0.
- The report ends with one recommendation line.

## Time

<!-- Planned: the planning agent's commitment (>= 2m per step). Estimated: written by `at kanban estimate` — do not edit. -->
Planned: 2h
Estimated: -
<!-- provenance: row=7710a15b74a9 -->
