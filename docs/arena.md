# `msft arena`: same prompts, several models, several scoring frameworks

`msft eval` scores a file of recorded replies. `msft probe` sends one model through the serve
path and runs the gates. Neither answers the question that decides which GGUF goes into the
pipeline: **on the same cases, which model keeps the instruction's format, keeps the terms,
reads like Russian, and how fast does it decode — and where do the answers disagree?**

`arena` sends one deterministic request per case to every model, scores every reply with
several independent frameworks, writes one reply file per model plus a leaderboard, and
refuses to fold the frameworks into a single number. Two of those frameworks answer the questions
the pipeline actually gates on: `contract` (did the reply have the shape the instruction asked
for — bullets, JSON, term table, edit pairs, plain text) and `speed_index` (how fast this model
decoded relative to the fastest one in the same run).

```bash
export MSFT_ARENA_ENDPOINT=<your ollama base URL>   # never committed, never in this repo
msft arena --models machinum:4b,gemma-4-12b --cases data/eval/cases.jsonl \
           --judge qwen3.5:27b --out out/arena/run1
```

## What one row of the table means

Per model, the report prints, side by side: how many cases it answered, whether the replies
fit the pipeline's protocol, how often they fit the format the instruction asked for, how close
they are to the references, how many glossary terms survived, what a blinded judge thought of
fidelity and fluency, how often scripts mixed, how often the source was copied, the Cyrillic
share, the token rate, the speed share, the median and worst reply time and the model-load time.
Each column answers one question; none of them is "quality".

## The frameworks

| column | level | one-line definition | blind to |
| --- | --- | --- | --- |
| `schema_validity` | per case | the `msft eval` gates: no chatter, shape, line parity, terms block | translation quality |
| `contract` | per case with `expect`, mean over those | fraction of the instruction's format rules the reply satisfies (see below) | whether the content is right |
| `chrF` | per case, mean over referenced cases | character n-gram F<sub>β</sub>: `P` and `R` from matched n-grams for n = 1..6, `F = (1+β²)PR/(β²P+R)` | word order past six characters, meaning |
| `BLEU` | corpus, over referenced cases | `100 · BP · exp(meanₙ ln pₙ)`, `p₁ = m₁/t₁`, `pₙ = (mₙ+1)/(tₙ+1)`, `BP = exp(1 − r/c)` when `c ≤ r` | synonyms, recall, meaning |
| `glossary` | per case with terms | fraction of the expected Russian terms that appear as substrings | whether the term is used correctly |
| `judge fidelity` | per judged reply | 1–5: is the source meaning preserved | everything the judge model cannot see |
| `judge fluency` | per judged reply | 1–5: is the Russian natural | fidelity by construction |
| `mixed_script` | per reply, summed | tokens containing letters from two scripts (the `палубный甲板` failure) | correctness |
| `source_leak` | per reply, summed | reply lines identical to a source line | paraphrase quality |
| `cyrillic_ratio` | per reply, mean | Cyrillic letters / all letters | everything else |
| `tok/s` | per reply, mean | `eval_count / (eval_duration / 1e9)` — generation only | prompt processing |
| `speed_index` | per run, per model | `tok/s ÷ max(tok/s across the models in this run)`; 1.0 is this run's fastest | absolute speed, everything about quality |
| `p50 s` | per run, per model | median per-reply wall seconds (`total_duration`), load included | tokens produced |
| `worst s` | per run, per model | the slowest single reply in seconds | how typical that reply was |
| `load s` | per run, mean | `load_duration` in seconds | generation |

### The contract score

`schema_validity` asks "would the pipeline accept this line at all". `contract` asks a different
question: **did the model do what this particular instruction asked it to do** — a bullet-list
summary, a term table, an edit list, one JSON object, or nothing but the text. A summariser that
returns a good paragraph where the bullets were requested has failed the request, and no
translation metric can see that. It is measured per reply, against the case's `expect` object, and
averaged over the cases that carry one.

Every format is a small rule list, and the score is the **fraction of rules the reply satisfies**,
not a pass/fail — a bullet list carrying two of three required bullets scores 0.667, which keeps
the distance from the contract visible. `contract_violations` on the reply row names each broken
rule with the measured value (`bullets_min: 2 bullet(s) < bullets_min 3`).

| format | rules |
| --- | --- |
| `bullet_list` | every non-bullet line is at most a one-line lead-in before the list; at least `bullets_min` (default 3) bullets, each line starting `- `, `* ` or `N. `; no prose line inside or after the list |
| `json_object` | the reply, after stripping an optional JSON code fence, contains exactly one parseable JSON object; every name in `keys` is present; nothing before it; nothing after it. Chatter around a good object is a violation, not a parse failure — the object still parses, the breach is still recorded |
| `term_table` | at least `pairs_min` (default 1) lines of `term -> translation`, `term: translation` or `term — translation`; every non-empty line has that shape (a leading `- `/`* `/`N. ` is tolerated); no repeated source term |
| `edit_pairs` | the same shape rules with `old -> new`; an `old -> old` line is allowed as an unchanged marker; two different new sides for one old side is a conflict; with `unchanged: true` the case's `unchanged_line` must also appear byte-identically |
| `plain_text` | no code fence, no `**bold**`, no markdown heading, no bullet line — "just the text" |

Three things the number deliberately does not do:

- **A case with no `expect` scores `null` and is left out of the mean.** It never asked for a
  shape, so it cannot fail one, and averaging a zero in would punish a model for the case file.
- **An unanswered contract case scores 0**, exactly as it does for `schema_validity` and chrF: a
  case that asked for a shape and got no reply is the worst possible compliance, not an untested
  one. The `contract (n)` header prints how many cases carried an `expect`.
- **It does not read the content.** A bullet list of three lies scores 1.0. That is what `chrF`,
  the glossary column and the judge are for; the axes are meant to be read together, never summed.

With at least one `expect` case in the file the sort order becomes
`schema_validity, then contract, then chrF`: an unusable shape is a retry in the pipeline, while
a two-point chrF gap may be noise. It is still a sort order, not a verdict.

### The speed index

`tok/s` is a rate, and rates are not comparable across runs: one run's 120 tok/s is another
run's 90 tok/s once the GPU is shared. `speed_index` normalises inside the run —
`tok/s ÷ the fastest model's tok/s` — so 1.0 always means "this run's fastest" and 0.5 means
"half as fast as the fastest model on this machine today". It scales linearly, it is printed as
its own column, and it is never averaged with anything: it says nothing about quality and no
quality column says anything about it. Two latencies sit beside it because a mean rate hides
the tail that hurts a batch: `p50 s` is the median reply and `worst s` is the slowest one, both
end-to-end and both including model load (which is why `load s` stays a separate column).

### Why no composite score

A weighted sum makes a decision for you and hides the evidence. Two models can land on the
same number with opposite failure modes — one fluent and wrong, one literal and right — and a
composite column would call them equal. The table keeps the frameworks apart on purpose; the
sort order is printed above it so nobody has to guess which column won. The same rule governs
the new columns: `contract` is not folded into `speed_index`, `speed_index` is not folded into
`contract`, and nothing is folded into chrF. A model that is fastest and worst-formatted shows
up as exactly that.

## Who leads on each axis

Above the model table, the report prints one row per axis: `axis | model | value`. It answers the
question the model table cannot — not "which model is best" but "which model leads this one
axis", so a 4B model that keeps the bullets while a 12B model wins chrF is stated in one view.

```
who leads on each axis — the axes are not comparable to each other:

| axis | model | value |
| --- | --- | --- |
| schema_validity | machinum:4b | 1.000 |
| contract | machinum:4b | 0.917 |
| chrf | gemma4:latest | 69.7 |
| speed_index | gemma4:latest | 1.000 |
```

The same object is in `leaderboard.json` as `leaders`, one model name per axis, for the axes
`schema_validity`, `contract`, `chrF`, `glossary`, `judge_fidelity`, `judge_fluency` and
`speed_index`. **The axes are not comparable to each other**: 1.000 on `speed_index` is a share of
this run's fastest rate, 1.000 on `contract` is a share of one format's rules, and 69.7 on chrF is
characters — putting them in one column would be a unit error, not an average. An axis that no
model has a value on is absent rather than zero. Ties break on the model name so two runs of the
same command print the same table. A self-judged model is skipped on the judge axes for the same
reason its judge mean is excluded from the ranking.

The markdown also carries a **per-task table**, one block row per model and `task` label
(`summarize`, `ner`, `proofread`, `translate`), because "best model" is not a portable property:
the same model can top the bullet-list rows and trail the translation rows, and that is the
comparison the pipeline actually needs.

`--rank-by <axis>` re-sorts the table on ONE axis instead of the default order, for the days when
the question is "which of these is fast enough" rather than "which reads best": any axis from the
leaders table, plus `tok_per_s`, `answered` and `p50_reply_s` (latency ranks lowest-first). The
`sort_key` in `leaderboard.json` then reads `requested: <axis>`, so a sorted table can never be
mistaken for the default one; an unknown axis is a usage error (exit 2), not a silent fallback.

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
- **`contract` is shape, not truth.** A three-bullet list of three falsehoods scores 1.0 on
  `contract`. The column exists to catch the model that ignored the instruction, and it is read
  beside chrF, the glossary and the judge — never instead of them.
- **`speed_index` is relative to the run.** 1.0 means "fastest of the models in this run on this
  machine", not a hardware benchmark: change the model list and every other model's index moves.
  It is never averaged with a quality column, and it never enters the sort.
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
{"case": "ch1-042", "task": "translate", "prompt": "Translate this line.",
 "source": "The deck stands.", "reference": "Палуба стоит.", "expected_lines": 1,
 "terms": [{"src": "deck", "tgt": "палуба"}], "tags": ["chapter-1"]}
```

- `prompt` is read from `prompt`, then `user`, then the last user message of `messages`.
- `source` defaults to `prompt`; `reference` stays absent when the record has none.
- `task` labels the instruction: `summarize`, `ner`, `proofread`, `translate`. It is carried into
  every reply row and into the per-task table. Any other non-empty string is kept as its own
  label, so a fifth task shows up instead of disappearing.
- `expect` is optional and describes the reply shape the instruction demanded. It is the only
  thing the `contract` column reads; a record without it has `contract: null` and is left out of
  the contract mean.

```json
{"case": "summary-bullets", "task": "summarize",
 "prompt": "Summarise the incident report below as a markdown bullet list.\n\nINCIDENT: ...",
 "expect": {"format": "bullet_list", "bullets_min": 3}}

{"case": "ner-json", "task": "ner", "prompt": "Extract the entities as JSON only.",
 "expect": {"format": "json_object", "keys": ["persons", "places", "dates"]}}

{"case": "ner-terms", "task": "ner", "prompt": "Return one `term -> translation` per line.",
 "expect": {"format": "term_table", "pairs_min": 4}}

{"case": "proofread-edits", "task": "proofread", "prompt": "One edit per line as `old -> new`.",
 "expect": {"format": "edit_pairs", "pairs_min": 2, "unchanged": true,
            "unchanged_line": "The night shift had already handed over the manifest."}}
```

`expect` fields, all optional except `format`:

| field | applies to | meaning |
| --- | --- | --- |
| `format` | all | `bullet_list`, `json_object`, `term_table`, `edit_pairs` or `plain_text` |
| `keys` | `json_object` | every name must be a key of the one object |
| `bullets_min` | `bullet_list` | minimum bullets (default 3) |
| `pairs_min` | `term_table`, `edit_pairs` | minimum pair lines (default 1) |
| `unchanged` | `edit_pairs` | `true` adds the frozen-line rule |
| `unchanged_line` | `edit_pairs` | the exact line that must survive byte-identically; required when `unchanged` is true |

- A malformed JSON line, or a record with no prompt, becomes a case that fails every gate
  (`_error`) rather than aborting the run.
- An unknown `expect.format` is scored 0.0 with an `unknown format` violation rather than being
  dropped: a typo in the format name is visible, not silently an untested case.
- `--prompt TEXT` (repeatable) builds `ad-hoc-1..N` cases with no reference and no `expect`.
- Case names must be unique; a duplicate is a usage error.

## Artifacts

```
out/arena/<UTC stamp>/
  cases.jsonl                 the resolved case set, one object per line (keeps `task`, `expect`)
  replies/<slug>.jsonl        one row per case; a valid `msft eval --predictions` input, with
                              `task`, `contract` and `contract_violations` on each scored row
  judge.jsonl                 one row per judge call (scores, or judge_error + raw[:200])
  leaderboard.json            the table, the leaders, the per-task means, the notes, the sort key
  leaderboard.md              the leaders table, the model table and the per-task table
  meta.json                   redacted endpoint, model digests, judge, rubric hash, system,
                              request options, cases hash, git commit + dirty flag,
                              wall seconds per model, case count, n_ref, n_contract
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
jq '.rows[] | {model, answered, contract, chrf, speed_index, judge_mean}' \
   out/arena/run1/leaderboard.json
jq '.leaders' out/arena/run1/leaderboard.json
```

`docs/arena-cases.example.jsonl` now carries one case per task: six EN→RU translation lines
(four with a reference), one `plain_text` translation, a bullet-list summary, two NER cases
(a term table and a JSON object with required keys) and a proofread case with an edit list and a
line that must not change. Every `expect.format` the scorer knows is exercised by the file, so a
smoke run shows both new columns on a real endpoint.

```bash
msft arena --models machinum:4b,gemma4:latest --cases docs/arena-cases.example.jsonl \
           --judge gpt-oss:20b --out out/arena/run1 --dry-run
```

prints the case count, how many of them carry a `contract`, and the task labels, without making a
request.

## What the first real run in this repo found

`docs/arena-cases.example.jsonl` — at the time six EN→RU cases, four of them with a reference —
through four candidates on one endpoint, `gpt-oss:20b` as the judge, `--seed 7 --num-predict 4096`.
**That run predates the `contract`, `speed_index`, `p50 s`, `worst s` and per-task columns**, so the
table below has no row for them; it is kept as the record of what the older columns established,
not as a template:

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
  6.88 s load — the same table, one row apart. Re-read as the index the table now prints, that is
  `speed_index` 1.000 for `gemma4` and 0.279 for `qwen3.8`: the slower model is not "a bit slower",
  it is a quarter of the rate.

Quote a score with the command that produced it or do not quote it: the numbers above come from a
6-case file and say nothing about a chapter.
