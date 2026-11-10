# `msft arena`: same prompts, several models, several scoring frameworks

`msft eval` scores a file of recorded replies. `msft probe` sends one model through the serve
path and runs the gates. Neither answers the question that decides which GGUF goes into the
pipeline: **on the same cases, which model keeps the protocol, keeps the terms, and reads like
Russian — and where do the answers disagree?**

`arena` sends one deterministic request per case to every model, scores every reply with
several independent frameworks, writes one reply file per model plus a leaderboard, and
refuses to fold the frameworks into a single number.

```bash
export MSFT_ARENA_ENDPOINT=<your ollama base URL>   # never committed, never in this repo
msft arena --models machinum:4b,gemma-4-12b --cases data/eval/cases.jsonl \
           --judge qwen3.5:27b --out out/arena/run1
```

## What one row of the table means

Per model, the report prints, side by side: how many cases it answered, whether the replies
fit the pipeline's protocol, how close they are to the references, how many glossary terms
survived, what a blinded judge thought of fidelity and fluency, how often scripts mixed, how
often the source was copied, the Cyrillic share, the token rate and the model-load time. Each
column answers one question; none of them is "quality".

## The frameworks

| column | level | one-line definition | blind to |
| --- | --- | --- | --- |
| `schema_validity` | per case | the `msft eval` gates: no chatter, shape, line parity, terms block | translation quality |
| `chrF` | per case, mean over referenced cases | character n-gram F<sub>β</sub>: `P` and `R` from matched n-grams for n = 1..6, `F = (1+β²)PR/(β²P+R)` | word order past six characters, meaning |
| `BLEU` | corpus, over referenced cases | `100 · BP · exp(meanₙ ln pₙ)`, `p₁ = m₁/t₁`, `pₙ = (mₙ+1)/(tₙ+1)`, `BP = exp(1 − r/c)` when `c ≤ r` | synonyms, recall, meaning |
| `glossary` | per case with terms | fraction of the expected Russian terms that appear as substrings | whether the term is used correctly |
| `judge fidelity` | per judged reply | 1–5: is the source meaning preserved | everything the judge model cannot see |
| `judge fluency` | per judged reply | 1–5: is the Russian natural | fidelity by construction |
| `mixed_script` | per reply, summed | tokens containing letters from two scripts (the `палубный甲板` failure) | correctness |
| `source_leak` | per reply, summed | reply lines identical to a source line | paraphrase quality |
| `cyrillic_ratio` | per reply, mean | Cyrillic letters / all letters | everything else |
| `tok/s` | per reply, mean | `eval_count / (eval_duration / 1e9)` — generation only | prompt processing |
| `load s` | per run, mean | `load_duration` in seconds | generation |

### Why no composite score

A weighted sum makes a decision for you and hides the evidence. Two models can land on the
same number with opposite failure modes — one fluent and wrong, one literal and right — and a
composite column would call them equal. The table keeps the frameworks apart on purpose; the
sort order is printed above it so nobody has to guess which column won.

### Why `chrf` and `bleu` here are internal variants

`chrf` normalises to NFC and **drops every whitespace character** before counting n-grams, so
spacing and line wrapping inside a reply cannot move the score. `bleu` tokenises with
`\w+|[^\w\s]`, smooths orders 2..4 with add-one, and applies a corpus-level brevity penalty.
Both are written for comparing variants of the same pipeline on the same cases. They are
**not comparable** to sacreBLEU, to published chrF++ numbers, or to another repo's BLEU: the
tokenisation, the smoothing and the whitespace rule all differ. Quote them only next to the
command that produced them.

### The script flags stay separate

`mixed_script_tokens`, `source_leak_lines`, `cyrillic_ratio` and `length_ratio` are printed as
four numbers. A model that mixes scripts on one token and one that copies three source lines
are different failures with different fixes; averaging them into a "script score" would
report neither.

## The judge

The judge is one more model behind the same endpoint. For every successful candidate reply it
receives `SOURCE`, an optional `REFERENCE (may be imperfect)`, and `CANDIDATE`, each cut to
6 000 characters, and is asked for JSON only. It **never sees a model name** — the request
contains the text, not the tag — so it cannot reward the bigger or the familiar name. It runs
once, after every candidate has finished, so the GPU swaps models once rather than once per
case.

The judge may not be a candidate. A model asked to rate itself scores itself high, and in a
two-model arena that is enough to decide the table. The run refuses (exit 2) unless
`--allow-self-judge` is passed; when it is, that model's judge column is marked `*`, and its
judge mean is excluded from the ranking. The judge is not retried: a failed or unparsable
judge reply is a row with `judge_error`, counted as a failed judge call, not as a zero. A
zero would quietly reward a model whose replies the judge could not read.

## What the report will not claim

- **Coverage.** A failed request is a row with `_error`. It counts as `answered` = no and as 0
  for protocol, chrF and BLEU on that case. The `answered n/m` column makes that visible.
- **Reference coverage.** chrF and BLEU average only over the cases that have a reference;
  the header prints that count. With no reference anywhere the columns show `-` and the
  ranking is format + judge only.
- **Noise.** Fewer than 30 cases and the report says the differences are within noise. The
  order is still printed because you asked for one; it is not a verdict.
- **`≈ chrF`.** For every other model the paired per-case chrF difference against the leader is
  taken on the cases both answered. If the 95 % interval `mean ± 1.96·stdev/√n` contains zero
  **and** the model matches the leader's `schema_validity`, its chrF is marked `≈`: on this case
  set the translations are statistically indistinguishable. A model that fails a protocol gate is
  never marked `≈` — a line-parity failure is not a tie, it is unusable in the pipeline. The
  report never names a winner.
- **Token rate.** `tok/s` is eval tokens per eval second. Model load is excluded and reported
  separately in `load s`, because a cold model and a cached one differ by seconds, not by
  tokens.
- **Thinking models.** Every request carries `think: false`, `stream: false`,
  `options.temperature = 0` and the `--seed` / `--num-predict` you passed. If a model thinks
  anyway, `thinking_chars` records the length so the transcript is not invisible.
- **Moving targets.** ollama tags move. `meta.json` records the digest of every model and of
  the judge at the moment of the run; the same command tomorrow may load different bytes.
- **Determinism.** `temperature 0` and a fixed seed make a run repeatable in practice, not
  bit-exact: GPU kernels, batching and library versions still change the arithmetic.
- **The judge is a model.** Its 1–5 scores carry its own biases, its language preferences and
  its context window. They are one column, not ground truth.

## Cases

`--cases FILE` reads JSONL, one object per line:

```json
{"case": "ch1-042", "prompt": "Translate this line.", "source": "The deck stands.",
 "reference": "Палуба стоит.", "expected_lines": 1,
 "terms": [{"src": "deck", "tgt": "палуба"}], "tags": ["chapter-1"]}
```

- `prompt` is read from `prompt`, then `user`, then the last user message of `messages`.
- `source` defaults to `prompt`; `reference` stays absent when the record has none.
- A malformed JSON line, or a record with no prompt, becomes a case that fails every gate
  (`_error`) rather than aborting the run.
- `--prompt TEXT` (repeatable) builds `ad-hoc-1..N` cases with no reference.
- Case names must be unique; a duplicate is a usage error.

## Artifacts

```
out/arena/<UTC stamp>/
  cases.jsonl                 the resolved case set, one object per line
  replies/<slug>.jsonl        one row per case; a valid `msft eval --predictions` input
  judge.jsonl                 one row per judge call (scores, or judge_error + raw[:200])
  leaderboard.json            the table, the notes and the sort key
  leaderboard.md              the same table as markdown
  meta.json                   redacted endpoint, model digests, judge, rubric hash, system,
                              request options, cases hash, git commit + dirty flag,
                              wall seconds per model, case count, n_ref
```

`<slug>` is the model tag with `:` and `/` replaced by `_`. Two models whose tags collide as
file names are a usage error, not a silent overwrite.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | every model answered every case, and every judge call parsed |
| 1 | a request or a judge row failed; the report is still written |
| 2 | usage or configuration: no endpoint, unknown model, no cases, both `--cases` and `--prompt`, self-judge without `--allow-self-judge`, duplicate case name, colliding model tags |

## Worked example

Plan first — `--dry-run` makes no request at all and prints the cases, models, request counts
and the artifact directory:

```bash
export MSFT_ARENA_ENDPOINT=<your ollama base URL>
msft arena --models machinum:4b,gemma-4-12b \
           --cases data/eval/cases.jsonl --judge qwen3.5:27b \
           --out out/arena/run1 --dry-run
```

Then the real run, and the per-model gate check on the recorded replies:

```bash
msft arena --models machinum:4b,gemma4:latest --cases docs/arena-cases.example.jsonl \
           --judge gpt-oss:20b --out out/arena/run1
msft eval --predictions out/arena/run1/replies/machinum_4b.jsonl
jq '.rows[] | {model, answered, chrf, judge_mean}' out/arena/run1/leaderboard.json
```

## What the first real run in this repo found

`docs/arena-cases.example.jsonl` — six EN→RU cases, four of them with a reference — through four
candidates on one endpoint, `gpt-oss:20b` as the judge, `--seed 7 --num-predict 4096`:

| model | answered | schema_validity | chrF (4) | BLEU (4) | glossary | judge f/f | mixed_script | tok/s | load s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4:latest | 6/6 | 1.000 | 69.7 | 61.5 | 1.00 | 4.83/4.33 | 0 | 122.8 | 1.74 |
| qwen3.8:latest | 6/6 | 1.000 | 63.0 | 41.7 | 0.50 | 4.67/4.33 | 1 | 34.3 | 6.88 |
| machinum:4b | 6/6 | 0.667 | 63.1 | 43.3 | 1.00 | 4.17/4.00 | 0 | 80.2 | 1.26 |
| machinum:4b-cov2 | 6/6 | 0.667 | 63.1 | 43.3 | 1.00 | 4.17/4.00 | 0 | 80.3 | 1.86 |

Six cases is under the noise floor, so this is not a ranking. A second run of the same command
reproduced chrF, BLEU and every judge score exactly; `tok/s` and `load s` moved with the cache
(122.8 → 117.9 tok/s for `gemma4`, 1.74 → 3.73 s of load), which is why they are separate columns
and never folded into quality. What the columns did establish:

- **The fine-tune drops line breaks.** Both `machinum` adapters returned the two multi-line cases
  as one line: `line_parity` 2/6 failed, `schema_validity` 0.667. The other two models kept the
  shape. In the pipeline this is a retry, not a stylistic difference — the gates caught it and
  chrF alone (63.1 vs 63.0 for qwen) would not have.
- **`mixed_script` fired once.** `qwen3.8` rendered `Кормак` with an Arabic letter: one token,
  two scripts. That is the `палубный甲板` failure this flag exists for, and no ratio-based metric
  sees it.
- **Two digests, one answer.** `machinum:4b` and `machinum:4b-cov2` produced byte-identical
  replies on all six cases despite different digests; the tag names are not evidence of a
  different model.
- **Cost is a column.** `gemma4` decoded at 122.8 tok/s against `qwen3.8` at 34.3 tok/s with a
  6.88 s load — the same table, one row apart.

Quote a score with the command that produced it or do not quote it: the numbers above come from a
6-case file and say nothing about a chapter.
