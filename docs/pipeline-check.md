# Pipeline check: machinum:4b through `mt`, and what it actually produced

Date: 2026-11-09. Two models, one held-out chapter, the real pipeline, no mock.

```bash
cd machinum-pipeline/samples/sample2-practical
pnpm run runner:en                                   # 21 chapters extracted, no LLM
MT_ONLY_CHAPTERS=1 MT_LLM_TOOL=ollama MT_LLM_MODEL=machinum:4b \
  MT_OLLAMA_URL=http://<gpu-host>:7869/v1 MT_LLM_REASONING_EFFORT=none \
  pnpm run runner:chapters
```

The book is `books/book2`, the chapter is not in the training set.

## What the pipeline did

| model | wall per chapter | line parity | output |
| --- | --- | --- | --- |
| `machinum:4b` (Qwen3.5-4B + our LoRA, Q4_K_S, 2.6 GB) | 28–33 s | 33/33 lines | fluent, with the defects below |
| `qwen3.8:27b` (base, Q4_K_M, 20 GB resident) | ~2 min | 33/33 lines | markedly better, two garbled tokens |

Both run 100 % on GPU. `machinum:4b` holds 8 192 tokens of context and 3.2 GB resident; the 27B
takes 20 GB at 32 768 and needs both cards.

## The defect list, from the actual text

`machinum:4b`, first paragraph:

> Дождь лил с залива весь день, и **Файона Доул** приехала к двери погрузки на улице Фенвик с
> воротником, поднятым наверх, и четырьмя картонными коробками **в багажаре заимствованного
> седана**.

| line in source | what came out | why it is wrong |
| --- | --- | --- |
| *Fiona Doyle* | «Файона Доул» … «Фиона Доули» … «Кормак Доули» … «Кормак Доул» | three spellings for two characters **inside one page** |
| *Nadia met her* | «Надя встретила **его**» | wrong gender — the referent is Fiona |
| *the old Carnegie library* | «старого библиотеки Карнеги» | case/gender agreement |
| *in the back of a borrowed station wagon* | «в **багажаре** заимствованного **седана**» | «багажар» is not a word; a station wagon is not a sedan |
| *as though it weighed nothing* | «словно она **весила ничего**» | ungrammatical |
| *licorice drops* | «жевательная резинка» | wrong referent (chewing gum) |
| *Harbor papers* | «**бумажки** от гавани» | wrong register; «портовые документы» |
| *she said, and meant it* | «и **имела в виду это**» | calque of an English idiom |
| *No food or drink in the collection rooms* | «В залах коллекции **не еда и не напитки**» | broken syntax; «нельзя есть и пить» |

The 27B, same paragraph:

> Дождь лил с бухты всё утро, и **Фиона Дойл** прибыла к погрузочным воротам на Фенвик-стрит…
> **Надия встретила её**… в подвале **старой библиотеки** Карнеги…

Names stay stable, agreement is right, «портовые документы» replaces «бумажки». It still says
«заимствованного седана» for the station wagon, «всю утро» instead of «всё утро», and produced two
decoding artefacts: «Корمאك» (an Arabic letter inside a Cyrillic name — the exact kind of token
the Q4 quantiser's tail can emit) and «льявной насос» (not a word).

## The glossary, and what filling it did

`chapters/glossary.json` came out of `runner:glossary` with **103 terms and `ru: null` on every
one** — the pipeline's own comment says the Russian forms are a human step. They are also the
whole point: `tagWithGlossary` can only pin a name if it knows the target form.

`fill-glossary-ru.ts` fills them (character / place / item, temperature 0, no reasoning), 97 terms
in about 90 s. Re-running chapter 1 afterwards:

| | before | after |
| --- | --- | --- |
| spellings of Fiona Doyle | «Файона Доул», «Фиона Доули», «Дойл» | **«Файона Доул» ×8** |
| spellings of Nadia | «Надя», «Надия» | **«Надя» ×7** |
| spellings of Cormac | «Кормак Доули», «Кормак Доул» | **«Кормак» ×5, «Доул» ×3** |
| `<terms>` metadata leaking into prose | 5 of 33 lines | **0** |
| wall | 28 s | 30 s |

Two model-level defects survive, and they are what a bigger model buys: «химический дыхание»
(agreement) and «в багажаре» (a non-word for "trunk") — the 27B says «в багаже», correctly.

The leaked `<terms>` block was a second, independent bug: `translateLines` never called the
existing `cleanTranslation`, and when the model omits the closing tag — which it did on five
lines — the pair-based regex matched nothing. Both fixed in `machinum-pipeline` `687411b`.

## The book-length run

The same job over the whole book (21 chapters, 331 lines of prose), `machinum:4b`, glossary first:

| step | wall | result |
| --- | --- | --- |
| `runner:glossary` (whole book, one batch) | 26 s | 103 terms |
| `fill-glossary-ru.ts` | 20 s | **97 of 103** filled; 6 left null (Coast Guard, Mayday, a date, three company names) |
| `runner:chapters` (21 chapters, 5-way) | **437 s** | 21/21, no retries, 27.6 s (ch9) – 91.2 s (ch16) per chapter |
| audit (`scripts/audit_translation.py`) | <1 s | 331/331 lines, 0 parity failures, 0 metadata leaks, 0 empty lines |

Two defects survived to the audit and both are now fixed in the pipeline:

- **one foreign-script token in 331 lines** — «палубный甲板 магазина» (Chinese for "deck" emitted
  inside a Russian word, chapter 18). No ratio check can see it. `fix(parity)` now reports
  `foreign-script` as an error, so the existing retry machinery re-asks that line; re-running
  chapter 18 produced «продели его через старую палубу магазина» and the book audits clean.
- **the ship's name** — the glossary filled `Pelargic` as «Пеларгический» (an adjective), which
  then appeared 10 times. The fill prompt now says a ship keeps its transliterated noun
  («Пеларгик»); the entry was corrected by hand — which is exactly the workflow the file's
  comment describes — and the seven affected chapters re-ran in 4 minutes.

### The glossary itself was the worst defect

The automated fill made the drift *worse* in one respect: it translated each term in isolation,
so the file ended up containing `Doyle → «Дойл»` alongside `Fiona Doyle → «Файона Доул»`,
`Cormac Doyle → «Кормак До»` (truncated mid-name), `Fiona's → «Файон»` — and, because the
extractor's heuristic also proposes ordinary words, `March → «Мэрч»`, `Lock → «Лок»`,
`Rain → «Дождь»`, `Bring → «Приносить»` and `Rusk Marine → «Российский морской пехотинец»`.
`tagWithGlossary` pins whatever the file says, so those entries were being injected into every
prompt: a poisoned glossary corrupts more reliably than a good one helps.

Fixed by separating the two jobs:

- `chapters/names.json` — an editorial table of the book's proper nouns, check in the repo, with
  the Russian form and its genitive (Доул, Раска, Файоны, Оконкво…). Names are editorial
  decisions; they should not be re-invented per run.
- `scripts/apply_names.py` — rewrites every glossary entry from that table, marks the extractor's
  non-names `ignored` (the pipeline honours the flag, so they stay on record but never enter a
  prompt), and creates entries the extractor missed (`Kestrel Point`, `Marrow Bay Municipal
  Archive`, the ship `The Anselm`). Result: 63 active terms, 45 ignored.
- `scripts/compare_translations.py` — the measurement that caught it: parity, words, sentence
  length, Cyrillic share, repetition, foreign script, and the name forms each model produced.

After the re-run, across all 21 chapters: «Доул» ×18 and zero «Дойл»/«Доули», «Раск» ×23 and zero
«Ракс», «Надя» ×99 with one stray «Надия». Name drift is now a rounding error, not a defect class.

Residual, measured honestly: **10 of 331 lines** still use a second spelling of a name
(«Надия» ×2, «Доул» ×2, «Доули» ×1, «Дойл» ×5) against 97 × «Надя» and 24 × «Файона». The
glossary pins most lines, not all; a proofreading pass over the finished book is the next step,
not more training.

## What this means

1. **The protocol is learned, the language is not.** Line parity 33/33 and no reasoning chatter:
   the fine-tune did its job. Russian morphology, register and terminology are still the base
   model's, and a 4B base is weak at all three. 44 training pairs cannot fix that.
2. **Name drift is a pipeline problem, not a model problem.** Even the 27B drifted on a longer
   page in earlier runs; `mt` ships a glossary stage for exactly this, and the check above ran
   without it (`chapters/glossary.json` was absent). Running `pnpm run runner:glossary` before
   `runner:chapters` feeds the terms into the prompt builder and is the first thing to add to the
   loop — it costs one LLM pass over the book.
3. **Garbled tokens need a guard, not a bigger model.** «Корمאك» and «льявной» are single-token
   corruptions. A linter rule — "reject a line whose target contains characters outside
   Cyrillic + Latin + punctuation, or a word absent from both a dictionary and the glossary" —
   would retry exactly those lines. `line-parity` already has the retry machinery; this is one
   more check beside it.
4. **A 26–27 B model is the cheapest quality jump available.** No retraining, no new tooling;
   `docs/model-candidates.md` records which sizes fit which card budget.

## The bug this run exposed

`mt` was sending translations to ollama with no `reasoning_effort`. On the OpenAI-compatible
endpoint ollama ignores `think: false`, so every request came back with `content: ""` and the
whole budget spent in `reasoning`; the linter saw Cyrillic ratio 0.00 and retried until the run
died. Fixed in `machinum-pipeline` branch `fix/llm-reasoning-effort` (`65b8fe5`) — `MT_LLM_REASONING_EFFORT`
now travels from the config layers into the request body. Verified: the same chapter that failed
after three retries now finishes in 28 s.

Two more traps were found the same way and are recorded in `docs/memory.md`:

- `MT_OLLAMA_URL` must end in `/v1`; without it every call is a 404 from a server that answers
  `/api/tags` perfectly.
- `mt` imports the built `@mt/tools` `dist`, so editing `src` changes nothing until
  `pnpm --filter @mt/tools build`.
