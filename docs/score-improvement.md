# Raising the arena score: what to change, and in what order

`docs/arena.md` records the first comparison: on six EN→RU cases `machinum:4b` scored chrF 63.1,
BLEU 43.3, judge 4.17/4.00 — and failed `line_parity` on both multi-line cases (2/6), which dropped
`schema_validity` to 0.667 while every other model kept the shape. That failure, not the chrF gap,
is the first thing worth fixing: in the pipeline a collapsed reply is a retry, so it costs a
chapter, not a point.

This page is the research pass behind the second training run. It says what was read, what it
changed, and what was deliberately left alone.

## What was read

- **The corpus itself.** `data/raw/book1-parallel.jsonl` holds 49 pairs, all single-line: zero
  records contain a newline in either side. The source book has 12 chapters of human-parallel
  EN/RU prose; `scripts/pairs_from_book.py` was reading them one paragraph at a time and
  `min_chars = 40` dropped the short lines. The model had never seen a multi-line input, so a
  multi-line prompt could only be answered by collapsing it.
- **Habr, «Локальные LLM в переводах текстов — бенчмарк на RTX 5060Ti 16GB VRAM»**
  (https://habr.com/ru/articles/1102338/). Same hardware class as this project's 16 GB cards. The
  useful part is methodological: a reference translation produced by a stronger model, blind
  judging, and speed measured beside quality — the shape `arena` now implements.
- **Habr, «Бенчмарк качества англо-русского перевода для ведущих западных и китайских LLM»**
  (https://habr.com/ru/companies/ruvds/articles/1085636/). An EN→RU literary benchmark that
  replaced "count the editor's edits" with a reproducible judge protocol, and is explicit that a
  judge is taste-bound: two judges rank the same translations differently. That is why `arena`
  reports a judge column *next to* chrF/BLEU and never a composite.
- **The project journal** (`projects/machinum-sft/material`), in particular the LoRA recipe notes:
  rank/alpha folklore at these sizes (`r=16, alpha=32`, attention projections, low LR, cosine
  decay), and the QLoRA-vs-LoRA warning that already governs this repo.
- **A model's second opinion** (Google AI Mode, asked for line-alignment advice): batch several
  lines per sample, keep the rank modest on a small corpus, do not touch the MLP projections when
  the failure is formatting. Treated as a lead, not as evidence.

## What the evidence changed

1. **Data before hyperparameters.** 49 pairs over 2 epochs at `grad_accum: 32` is four optimiser
   steps; no rank or learning-rate sweep can beat a corpus the model has not seen. The 12
   chapters yield roughly three times the pairs once the `min_chars` filter is not the only rule,
   and they can be grouped into 2–4 line samples — which is exactly the missing capability.
2. **The formatting failure is a data failure.** The system prompt already says *one line per
   input line, keep paragraph breaks*; the training data contains no line break to keep. Adding
   grouped samples teaches the shape without inventing anything.
3. **One hyperparameter arm, not a sweep.** A second arm changes rank/alpha (`r=16, alpha=32`),
   LR (`1e-4`) and epochs (3) on the same data. Two arms, one comparison, both scored by `arena`
   against the current tag — the arena exists so this is a measurement, not a belief.
4. **No synthetic-data round yet.** Teacher-generated pairs are the obvious next lever, but they
   change the corpus's provenance, need their own filtering, and would confound this comparison.
   They are listed as the next step, not run here.

## The two arms

| arm | data | lora | train |
| --- | --- | --- | --- |
| `machinum-4b-v2-data` | every pair from 12 chapters + 2–3 line groups | unchanged (`r=32, alpha=16`) | lr `5e-5`, 3 epochs |
| `machinum-4b-v2-tuned` | same | `r=16, alpha=32` | lr `1e-4`, 3 epochs |

Both keep `completion_only_loss: true`, bf16 LoRA, the seven projections and `lm_head` — the
choices `docs/memory.md` defends. The winner is tagged `machinum:4b-v2`; the loser is kept as a
tag too, because a comparison nobody can re-run is not a comparison.

## How it was judged

`msft arena` on `docs/arena-cases.example.jsonl`, same endpoint, same judge, `--seed 7`:

```bash
msft arena --models machinum:4b,machinum:4b-v2-data,machinum:4b-v2-tuned \
           --cases docs/arena-cases.example.jsonl --judge <third-model> --out out/arena/v2
```

Success was defined before the run as `schema_validity` 1.000 for the new arms — the two multi-line
cases answered with the same number of lines — and chrF no worse than 63.1.

## What actually happened

Arena, six cases, `gpt-oss:20b` judge, default prompt: the new arms scored `schema_validity`
0.667 (`v2-data`) and 0.833 (`v2-tuned`) — the multi-line cases still collapsed, so by the rule
above this was **not** a win. Two of the three cases that did change showed the capability was
there: `v2-tuned` kept the three-line dialogue, `v2-data` did not.

The follow-up experiment found the wire: the corpus builder's system prompt is not the prompt the
arena sends. Re-running the same models with the *training* prompt:

| model | schema | chrF | BLEU |
| --- | --- | --- | --- |
| `machinum:4b-v2-data` | **1.000** | 62.9 | **45.6** |
| `machinum:4b` | **1.000** | 61.9 | 42.5 |
| `machinum:4b-v2-tuned` | **1.000** | 60.5 | 43.2 |

So the failure was never only in the weights: a model keeps the lines it was taught to keep when
the instruction matches, and collapses them when the instruction changes. The corpus also earned
its keep — `v2-data` is +3.1 BLEU over `machinum:4b` on the same prompt, while the tuned arm
(r=16/alpha=32, lr 1e-4) bought nothing. Conclusion: **corpus 1, hyperparameters 0**, and the next
run must render the product's own prompt (`packages/tools/prompts/translate.system.md`) into the
training data instead of this repository's short one.

`docs/model-tags.md` records the four tags this produced, their digests and the abliterated twin.

