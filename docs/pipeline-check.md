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
