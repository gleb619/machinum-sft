# The material pipeline: a daily loop that reads the LoRA/SFT literature for us

`msft material` is a small state machine that visits the sites the stack already knows how to
drive — Habr, GitHub, Hugging Face, Google AI Mode, DeepSeek, Qwen, X/Twitter, Grok — and free-form
`ask` notes, scores what comes back, packs the survivors into one task file, hands that file to a
coding agent, and posts the resulting notes into the ai-memory wiki.

It exists because the product moves. `machinum:4b` was trained with plain LoRA, a 49-pair
dataset and a hand-written eval gate; every one of those choices is a decision the field
revisits every few months — QLoRA vs LoRA on 16 GB cards, completion-only loss, which
trainer survives a version bump, how to build a Russian literary corpus without poisoning it
with machine translation. Human reading of that literature does not scale, and it is exactly
the kind of reading an agent loop does well: fetch, score, summarise, skip the press releases.

The rule that shapes everything below: **the raw material is bulky, private and disposable;
the distilled note is small, cited and permanent.** Raw bodies stay on this disk and out of
git; what survives a machine reinstall is one page per finding in the wiki, written by
`memorize`.

## The loop

```
seed ─▶ run ─▶ distill ─▶ memorize ─▶ report
        │        │            │
        │        │            └─ POST /admin/write-page → ai-memory wiki
        │        └─ `at` agent reads pack/<date>.md, writes notes/<date>/*.md
        └─ webbridge search/extract/ask → raw/<date>/*, items rows, scores
```

Each arrow is one CLI command, and each is safe to re-run: the same URL never becomes two
items, the same page path never becomes two notes. There is no daemon and no scheduler
inside the CLI — `scripts/material-daily.sh` plus one cron line is the scheduler.

### State machine

Candidates are the frontier — what to look for, not what was found. Items are pages.

```
  candidates: the frontier (what to look for)
      seed / expansion -> INSERT, UNIQUE(kind, value, site)
                              |
                              v
                        +-----------+
                        |    new    |
                        +-----------+
                          |    |    |
      the call ran ok ----+    |    +---- never called again
                              |                    |
                              v                    v
                        +-----------+      +-------------+
                        |   done    |      |   dropped   |
                        +-----------+      +-------------+

      the call failed --> +-----------+
                           |   error   |   attempts+1, error = the stderr tail
                           +-----------+
```

```
  items: pages (one row per canonicalised URL)
      search hit / ask answer
              |
              v
        +---------+
        |  seen   |   score computed from configs/material.yaml
        +---------+
          |     |
          |     +-- negative keyword, radio LoRa without a domain term, or
          |         score < keep_min -------------------------> [ scored_out ]
          |
          |     +-- content_sha256 already in items
          |         (cross-post, second URL) -----------------> [ duplicate ]
          |
          |  score >= keep_min: fetch, normalise, write raw/<date>/
          v
        +---------+
        | fetched |   raw/<date>/<site>-<slug>.json and .txt
        +---------+
              |
              |  the distill run packs it into pack/<date>.md
              v
        +---------+   at exits non-zero or prints no summary
        | packed  |---> stays packed, the next run retries
        +---------+   the agent wrote notes/<date>/<slug>.md
              |
              v
        +------------+  distilled_at set, non-"none" experiment -> experiments.md
        | distilled  |
        +------------+
              |
              |  POST /admin/write-page -> 200 or 201
              v
        +------------+  memorized_at set, INDEX.md rewritten
        | memorized  |
        +------------+

  anything that cannot proceed: status = error, the error column keeps the reason
```

Nothing is inferred from files on disk. A note file without a `distilled` row is an orphan;
a `packed` row without a pack entry is a bug. That is deliberate — `status` is the only
truth, so a re-run after a crash resumes instead of guessing.

## Subcommands

| Command | What it does | Writes |
| --- | --- | --- |
| `seed` | inserts the configured queries into the frontier | `candidates` |
| `add` | inserts explicit URLs into the frontier, ahead of every query | `candidates` |
| `run` | calls webbridge for the due candidates (fresh, retryable, and `done` ones past `limits.refresh_days`), tops a short day up with synthesized pairs, scores, fetches (with Habr's comment tree), expands, records outbound links | `candidates`, `items`, `raw/` |
| `distill` | packs `packed` items into one file, runs the `at` agent over it | `pack/`, `notes/`, `experiments.md` |
| `memorize` | posts each `distilled` note to ai-memory | wiki, `INDEX.md` |
| `report` | one line (or one JSON object) for a date | nothing |
| `status` | counters for the whole state dir | nothing |
| `tree` | the net one day drew: candidate → item(s) → their outbound links, as an ASCII tree or `--json` | nothing |
| `rescore` | re-applies the *current* brief to items an older brief threw away (`scored_out` → `new`); status never moves the other way | `items` |

Exit codes are the contract: `0` ok, `1` the run finished but some items failed, `2` a
config or environment error (bad YAML, missing token, unwritable directory). Only `2` means
"fix the operator, then re-run"; `1` means the loop worked and some sites did not.

```bash
msft material seed                                  # once, and again after editing configs/material.yaml
msft material seed --dry-run                        # what would be inserted
msft material run --limit 6                         # six queries, the daily default
msft material run --limit 3 --site habr --site hf    # one narrow sweep
msft material run --max-items 5 --no-expand         # cheap: no tag/author children
msft material run --date 2026-11-09                 # backfill into last Monday's directory
msft material distill                               # pack + agent
msft material distill --agent ds-code --timeout-ms 900000
msft material distill --dry-run                     # show the pack and the at command, never call at
msft material memorize                              # everything distilled, any date
msft material memorize --workspace default --project data
msft material report                                # today
msft material report --date 2026-11-09 --json       # one JSON object on stdout
msft material status --json
msft material tree                                  # today's net, one screen
msft material tree --date 2026-11-10 --depth 3      # two days back, one level deeper
msft material tree --json                           # same structure, for a dashboard
msft material rescore --dry-run                     # what today's brief would bring back
msft material rescore                               # flip them to `new`; the next run fetches them
```

**A brief is not frozen.** `run` scores an item once, and `scored_out` is final — correct for a
stable brief, wrong every time the priorities move. Re-targeting the research to
summarization/NER/proofreading changed the keyword list, and the 22 posts the old list had
discarded had to be re-read (`rescore`: 22 rows, 5 back, one at 1.0). `run` therefore also
drains rows left at `status='new'` — a `rescore` flip, or a run that died between insert and
fetch — because the query loop only reads what it just found and `state.item(url)` stops the
same post from being discovered twice.

**The keywords carry both alphabets.** The corpus is mostly Russian while the brief is English,
and an English-only list scores a perfectly on-topic habr post at `0.00` (found on 2026-11-12:
six posts on structured output dropped, titles like «Почему ваш LLM-сервис ведёт себя как
хочет»). Each priority therefore has a Russian stem beside it (`структурированн`, `сущност`,
`опечатк`, `перевод`, …).

`report` prints one line, and that line is what the cron delivers:

```
material 2026-11-10: 6 queries → 12 items, 8 kept (habr 5, github 2, hf 1), 3 duplicates,
2 distilled, 2 memorized, 1 experiment · frontier 41 · raw 14 files · 92 s
```

The numbers above are the *format*, not a measurement from this machine. With `--json` the
same facts arrive as `{date, queries, items, kept, duplicates, fetched, packed, distilled,
memorized, experiments, frontier, errors, seconds}` — and then everything human-readable goes
to stderr, so `report --json | jq` is safe in a pipeline. `run --json` additionally carries
`comments_fetched`, `comments_failed`, `links_queued` and `links_known`; `report` does not sum
those four, because they describe the fetch, not the funnel.

Environment overrides the CLI reads itself (the script only forwards the path ones):

| Variable | Default |
| --- | --- |
| `MSFT_TUNNELSTACK` | `/home/boris/tunnelstack` — where `src/webbridge.py` lives |
| `MSFT_MATERIAL_DIR` | `<repo>/docs/material` |
| `MSFT_MATERIAL_CONFIG` | `<repo>/configs/material.yaml` |
| `MSFT_MEMORY_URL` | `http://127.0.0.1:49374` |
| `MSFT_MEMORY_TOKEN` | first `AI_MEMORY_AUTH_TOKEN=` line of `deploy/memory/.env` |
| `MSFT_AT_BIN` | `~/.local/bin/at` |

## The config file

`configs/material.yaml` is committed, because the *policy* of what to read is a decision worth
reviewing in a diff. Four blocks matter:

| Block | Knob | What changing it buys you |
| --- | --- | --- |
| `product` | `keywords`, `negative`, `domain_terms` | the definition of relevance; every score is computed against these lists |
| `limits` | `queries_per_run`, `items_per_run`, `items_per_site` | how much a single run may spend; the first thing to cut when a run is too slow |
| `limits` | `max_frontier`, `children_per_item` | how fast the frontier may grow from expansion; the second thing to cut |
| `limits` | `links_per_item` | outbound links per item, per kind: how many hosts-with-a-recipe become fetch jobs (default 4) and how many hints are recorded (same cap). `0` turns link extraction into hints-only |
| `limits` | `refresh_days` | how long an answered query stays answered. `3` means a keyword is re-asked every fourth day for whatever appeared under it; `0` restores the old one-shot crawl |
| `limits` | `synthesize_queries` | `true` tops a short day up with keyword pairs from `product.keywords`, so the budget is spent even when nothing is due |
| `scoring` | `keep_min`, `expand_min` | the two thresholds: below `keep_min` an item dies, below `expand_min` it is not used as a parent |
| `scoring` | `recency_days`, `site_prior` | taste: how much a 2026 post beats a 2024 one, how much GitHub/HF beat Habr |
| `expansion` | `per_site` | which child kinds a site may produce — `habr: [tag, author]`, `github: [repo]`, `hf: [model]` |
| `distill` | `agent`, `timeout_ms`, `notes_dir` | which agent writes the notes and how long you are willing to wait |
| `memory` | `workspace`, `project`, `path_prefix` | where the notes land in the wiki |

`product.domain_terms` is the one list that is not about ranking. It is the
disambiguation escape hatch described below; deleting an entry there makes every Habr post
that happens to say "LoRa" (the 868 MHz radio) a candidate again.

## Files, and why they are not in git

```
docs/material/
  .state/material.db            SQLite: candidates, items, runs, kv — the only source of truth
  .state/lock                   flock target; never committed, never read
  raw/<date>/<site>-<slug>.json exact webbridge output, for re-parsing without the network
  raw/<date>/<site>-<slug>.txt  normalize_text(body) + the comment tree, what the agent reads
  raw/<date>/<site>-<slug>.comments.json  the exact Habr comment payload, provenance
  pack/<date>.md                the distill task file: brief, rules, schema, one section per item
  notes/<date>/<slug>.md        written by the agent: frontmatter + five sections
  INDEX.md                      table of memorized notes (date, title, site, score, experiment)
  experiments.md                append-only list of concrete experiment proposals
```

`slug` is 60 characters of the URL path, `[a-z0-9-]`, plus 8 hex characters of a sha1 of the
URL: stable across runs, unique across sites. `normalize_text` strips
`<script|style|nav|aside|footer>` blocks and HTML tags, collapses three or more newlines to
two, trims trailing spaces and caps the body at 40 000 characters, so a file on disk is at
most ~40 KB and the same input always produces the same bytes.

`.gitignore` carries one line for the whole tree:

```
docs/material/
```

Why the raw material stays out:

- **It is other people's text.** A 40 000-character article body copied into this repo is a
  licence question with a date on it, and git history is forever. The note that quotes three
  numbers and links the source is not.
- **It is bulky and regenerable.** A dozen items a day at tens of kilobytes each, plus the
  API JSON, is a repository that grows every night for a diff nobody reads. The `.json` copy
  exists so a parsing bug can be debugged offline, not so it can be versioned.
- **It is not the deliverable.** The deliverable is knowledge in a store the next session
  reads at start — the ai-memory wiki, under
  `projects/machinum-sft/material/<date>-<slug>.md`. That wiki is markdown on disk, backed up
  with `tar czf memory-$(date +%F).tgz -C deploy/memory data`, and queryable over MCP without
  this repo being checked out at all. See `memory-stack.md`.

Consequence worth knowing: `INDEX.md` and `experiments.md` live inside the ignored directory,
so they are local conveniences, not history. The durable index is the wiki page list
(`memory_recent`), and the durable experiment list is the `## Experiment` section of each
page. If a future decision wants proposals in git, add
`!docs/material/experiments.md` below the ignore line — deliberately not done here, because
the tree is private by contract.

Losing `docs/material/` costs the corpus, not the knowledge: the notes are in the wiki, and
`seed` plus a run rebuilds a frontier from scratch.

## The smart part

Three mechanisms keep a keyword sweep from turning into a firehose of press releases, radio
modules and duplicate translations.

**1. Expansion — the frontier feeds itself.** A good item is a *lead*: it names its own tags,
its author, its topics, its model card. When `score >= expand_min` and expansion is on, the
run enqueues children with `parent=<url>`, at most `children_per_item` of them:

| Site | Child | Guard |
| --- | --- | --- |
| Habr | `kind=tag` for each tag | tags shorter than 3 characters are skipped |
| Habr | `kind=author` | only when `score >= expand_min + 0.15` — a good article by a named author is worth their back catalogue, a mediocre one is not |
| GitHub | `kind=repo` from `topics`, searched as `topic:<t>` | needs the `extract` payload, not just the search hit |
| HF | `kind=model` from the model card's tags | filtered to `lora, qlora, peft, sft, dpo, grpo, unsloth, trl, axolotl, dataset` |

The frontier is capped at `limits.max_frontier` rows; when it is full the run counts and skips
(`frontier_full` in the counters) rather than dropping what it already has. `--no-expand`
turns the whole mechanism off for a cheap sweep — tags, authors *and* links.

**Links — a page is also a list of leads.** Expansion asks what a page *is*; links ask what it
*points at*, and for this product that is usually the more useful question (an article about
QLoRA ends with the repo, the paper and the dataset). After a successful fetch every http(s) URL
in the article body **and** in the comment tree is classified by host:

| Link host | Row | What happens |
| --- | --- | --- |
| `github.com`, `huggingface.co`, `habr.com`, `x.com`/`twitter.com` | `kind='url'`, `reason='link:<host>'`, `parent=<item url>`, `priority=<item score>` | a real frontier node: the next run extracts it like any other URL |
| anything else (arxiv, a vendor blog) | `kind='hint'`, `status='hint'` | a leaf the note and `tree` can show; no recipe, so it is *never* a fetch job |

Both kinds hang off the item URL, so the chain is readable in both directions. `UNIQUE(kind,
value, site)` makes a link found twice — in the body and in a comment — one node, and
`limits.links_per_item` caps each kind per item, because a note with forty links is not a reading
list. A hint is stored in `candidates` with `status='hint'`, which no frontier query selects:
`candidates()`, `frontier()` and `status` never see it as work, and the tree can still draw it.
Counters: `links_queued` (new fetch jobs), `links_known` (the frontier already had it).

Habr's own JSON endpoints (`habr.com/kek/…`) are excluded on purpose: they sit on a host with a
recipe, but fetching a comment payload as if it were an article wastes a browser call.

**Refresh — a question is not one-shot.** A `done` candidate whose `updated_at` is at least
`limits.refresh_days` old is eligible again (3 days in the shipped config). This is the
difference between a daily loop and a crawl that finishes: a post published today was not in
the answer to `QLoRA обучение больших языковых моделей` yesterday, so the same question is
worth asking again on a schedule. Two rules keep it bounded. `kind='url'` never refreshes —
the operator pointed at one page, and one read of that page answers it for ever. And due
`done` nodes are taken **oldest `updated_at` first**, behind the fresh and retryable ones, so
six queries a day walk the whole pool instead of re-asking the same six. Re-executing a node
writes `updated_at` again, which re-arms it for another `refresh_days` — no extra column.

**Synthesis — widening the pool.** Refresh only hands back the questions the seeds happen to
contain: with six seeds and three days, a day with nothing fresh has two due nodes. When
`limits.synthesize_queries` is on and the run still has budget left, it pairs the keywords
from `product.keywords` in order (`lora qlora`, `lora peft`, `qlora peft`, …) — skipping the
hyphenated scoring stems such as `fine-tun`, which exist so that `fine-tuned` matches during
scoring and are not words anyone types into a search box — skips any
string the frontier already holds for *any* site and status, and spreads the new questions
round-robin over the sites named in `seeds`. It is deterministic — the same state produces
the same plan, so `--dry-run` is a real preview — and it respects `limits.max_frontier`. A
`--site` sweep is an explicit narrow question and is not topped up. Two limits worth knowing:
the pairs are finite (19 keywords give 171 questions, so a few weeks of idle days exhaust
them and the answer is a longer `product.keywords`, not code), and a pair that was asked and
answered is never asked again.

**2. Three layers of dedupe.** Each layer catches a different duplicate:

1. `candidates UNIQUE(kind, value, site)` — the same *question* is asked once. Re-running
   `seed` on an unchanged config inserts nothing.
2. `items PRIMARY KEY(url)` after canonicalisation — lowercase host, `utm_*` parameters,
   `#fragment` and the trailing `/` removed. The same *page* found by two queries is one row,
   and `first_seen_at` is kept, not overwritten.
3. `content_sha256(normalize_text(title + "\n" + summary_or_body))` — the same *content* under
   two URLs. This is what catches a Habr translation of a vendor blog post, the same repo
   surfaced by two searches, an arXiv abstract and its HF model card. The second URL becomes
   `status='duplicate'` and is never fetched, which is also where the fetch budget is saved.

**3. Radio LoRa is not a fine-tuning technique.** `(?i)\blora(wan)?\b` matches both the
adapter family and the 868 MHz modulation, and a keyword sweep of Habr will find the latter:
antennas, gateways, TTN. The rule is a veto, not a bonus — an item that mentions LoRa and
nothing from `product.domain_terms` (`llm, language model, fine-tun, adapter, peft, qlora,
neural, transformer, нейросет, модел, обучен, токениз, весов`) scores 0 and becomes
`scored_out`. Both lists are data in `configs/material.yaml`, so a new false positive family
(e.g. "LoRa mesh" flooding the tags) is a one-line config change plus a re-run — no code.

The same veto shape handles the other recurring noise: `negative: [hiring, webinar, podcast,
discount, black friday]` scores 0 outright, before any recency or star bonus can rescue it.

## The net: comments, links, and `tree`

The loop is not a keyword sweep that ends in a list of hits. A page that survives the score
arrives with its discussion and leaves a trail of links, and the point of both is that the reader
can follow them further.

**Habr's comment tree.** The browser recipe returns `#post-content-body`, which is the author's
half of the page: the numbers people actually ask about (VRAM, batch size, how long the run took,
what the dataloader cost) usually show up in the comments, and the author often answers there.
So after a Habr article is extracted, `run` also reads

```
https://habr.com/kek/v2/articles/<id>/comments/?sort=date&fl=ru&hl=ru
```

public JSON, one GET, no browser and no auth. The rendered tree is appended to the raw text under
a `# Комментарии` delimiter, the exact payload is kept as
`raw/<date>/habr-<slug>.comments.json`, and `items.comments` counts the published comments.
Roots come from `parentId == null` only — Habr's `threads` array also lists replies, and walking
both printed the author's answers twice.

The rules that keep it safe:

- **A comment failure never fails the item.** `_fetch_comments` is a module-level seam returning
  `(payload, error)`; an error is `comments_failed` in the counters and a stderr line, and the
  article stays `fetched`. The article is the expensive half (a real browser render) and must not
  be thrown away because a JSON endpoint was throttled.
- **Comments get their own slice of the pack.** The pack's per-item body is the article head plus
  `PACK_COMMENT_LIMIT` characters of comment tree, so a 14 kB article no longer hides the
  discussion behind the 8 000-char budget. The `.txt` itself is still capped at `TEXT_LIMIT`
  (40 000) by `normalize_text`; a very long thread is truncated *there*, and the
  `.comments.json` sidecar holds all of it.
- **The item row says so.** `items.comments` is an added column; an existing `material.db` is
  migrated in place by a `PRAGMA table_info` check plus `ALTER TABLE` on open (`State._migrate`),
  because `CREATE TABLE IF NOT EXISTS` never touches a table that already exists and rebuilding
  the database would throw away weeks of reading history.

**`tree` — the shape of a day.** Three edges make the net: `candidates.parent` (the item a child
question came from), `items.parent` (inherited from the producing candidate) and the link rows
hanging off the item URL. `msft material tree` draws exactly those rows:

```
material tree 2026-11-12 · 9 item(s) · 21 candidate(s) · 1 link node(s) · 1 hint(s)
root · 8 item(s) whose parent is not in this day
├─ habr 1 · memorized  Parameter-Efficient Fine-Tuning (PEFT): методы LoRA, Prefix…
│  ├─ [tag] habr · done  peft
│  ├─ [author] habr · done  dbaturova
│  ├─ link github  https://github.com/huggingface/peft
│  └─ habr 0.36 · memorized  Эффективный инференс множества LoRA адаптеров  ← 791966
└─ github 0.88 · memorized  modelscope/ms-swift
```

`--depth` bounds the walk (default 2: the day's items plus their branches), `--json` returns the
same structure for a dashboard, and a day with no items prints one line instead of an empty box.
Nothing is inferred from files: the tree is a rendering of the rows, so a link that was never
fetched still appears (`link`), a lead with no recipe appears as a hint, and an item whose parent
lives on an earlier day is a root with the parent's id marked (`← 791966`).

**X/Twitter and Grok are search sources too.** `SITES` carries `twitter` and `grok`:

- `kind=query, site=twitter` → `search twitter <query>`, parsed one row per tweet
  (`author`, `handle`, `text`, ISO `time`, permalink). The permalink is the item URL — a tweet is
  addressable — and a relative one (`/user/status/1234`) gets the `https://x.com` prefix; the
  handle lands in `items.author`. A high-scoring tweet expands to `kind=author`, which
  `candidate_argv` turns into `search twitter from:<handle>`, because X has no author page to
  extract and `from:` is how its search grammar asks for one account.
- `kind=note, site=grok` → `ask grok <question>`: grok is a chat site, so the frontier node holds
  a *question*, not a keyword, and the answer is one item. Its URL is the site home plus a hash
  of the question (`https://grok.com/?ask=<sha1>`), because a chat answer has no permalink that
  survives a re-ask: the marker in the *query* (not a fragment — `canonical_url` drops fragments,
  which would collapse every answer into one item) makes the same question dedupe onto one row
  and a different question a different row. Grok expands nothing: an answer has no tags, no
  author and no topics, and turning a model's citations into fetch jobs is not discovery.

Both degrade the way every other source does: a payload with no `items`/`data`/`text` yields no
items and the candidate is `done` — never an exception, never a failed run.

**One tweet on purpose: `add` a status URL.** `x.com` and `twitter.com` are in `URL_SITES`, so
`msft material add https://x.com/<user>/status/<id>` is an ordinary explicit read
(`kind='url'`, priority 1.0) that goes through `extract twitter`. Three things are specific to a
tweet:

- **The item is the tweet, not the page.** The extract payload is a one-row result list
  (`items: [{author, handle, text, time, link}]`) with a page shell at the top level whose `title`
  is just `"X"`, so a `url` candidate reuses the same `_normalise_entry` the `search twitter` rows
  use: permalink → `items.url`, handle → `items.author`, ISO `time` → `items.published`, text →
  `items.summary`. A payload without `items` falls back to the generic page text.
- **Its links are recovered from the anchors.** X wraps long URLs across lines and truncates
  them with `…` in the rendered text ("https://" / "huggingface.co/unsloth/Qwen3." / "…"), so
  `extract_links` finds nothing usable. The tweet's `[data-testid="tweetText"] a` hrefs are read
  with one `webbridge send evaluate` against the tab the extract left open, accepted only when
  `location.href` is the same status (`_same_page` — the browser is shared, so a moved tab must
  not donate another page's links), then each `t.co` short link is resolved over plain HTTP and
  appended to the body under `# Links`. `_record_links` then mints candidates from them like any
  other body's links. All of it is best-effort: a dead evaluate costs a log line, never the item.
- **`t.co` is read from the redirect page, not from a header.** Asked without a browser session
  it answers `200` with `<noscript><META …URL=…>` plus `location.replace("https:\/\/…")` and no
  `Location` header; a `HEAD` returns the short link itself. So `_resolve_short_link` does a
  capped `GET` and parses the body (JS form first, then the meta/title forms).

`_same_host` folds `twitter.com` onto `x.com` for the same reason: every status canonicalises to
`x.com`, and an extract of a `twitter.com` URL legitimately comes back on the other host.

**A fetch validates the payload, not the exit code.** webbridge drives one shared, stateful
browser: an `extract` issued while the tab still sits on another page returns *that* page — its
own `url`, `found: false`, and `text: '""'` (how the client prints an empty extraction). The first
live run after the re-target stored one such payload as a success: a 3-byte `.txt` for a 12 KB
article, marked `fetched`. `_extract_error` now refuses a foreign host, `found: false`, and an
empty-or-quoted-empty text as retryable errors, and a whitespace-only body is an error too — so
after a bad day read `items.status='error'`, not the run's `errors: 0`.

## Reading one page on purpose

A keyword sweep is a guess about what matters; sometimes you already know. One command puts a
page in the frontier, and it then flows through the same pipeline as everything else — raw
file, pack, note, wiki page:

```bash
msft material add https://habr.com/ru/articles/1044422/
msft material add URL URL URL          # repeatable: one candidate per URL, deduped
msft material add URL --dry-run        # what the frontier would gain, state untouched
msft material add URL --json           # one object on stdout
```

The committed equivalent, for a page that stays interesting for weeks, is a seed at the top of
`configs/material.yaml`:

```yaml
seeds:
  - {kind: url, site: habr, value: "https://habr.com/ru/articles/1044422/"}
```

The site is inferred from the host: `habr.com` → `habr`, `github.com` → `github`,
`huggingface.co` → `hf`, `x.com`/`twitter.com` → `twitter`, `unsloth.ai` → `unsloth`, with
`www.` and subdomains accepted. **An unknown host is a config error** (exit 2, `no site recipe
for host '…'`), never a search — there is no recipe to extract a page from an unknown site, and
searching for a URL string finds nothing.

A vendor docs site with a markdown endpoint is a `url` source like any other: the `unsloth`
recipe reads the page's `.md` twin (`https://unsloth.ai/docs/models/qwen3.8` →
`…/qwen3.8.md`), so the body is markdown — headings, code fences, tables — not text scraped
out of a rendered DOM.

Three properties make an explicit URL different from a sweep:

- **It runs first.** The candidate carries `priority = 1.0` while every seed and child carries
  `0.0` or a score, and the frontier is ordered by priority, so explicit URLs are executed
  before the keyword sweeps — even on a run budgeted to a single query.
- **`scoring.keep_min` does not apply.** The operator already decided the page is relevant, so
  `run` computes the score, stores it and shows it in the pack, but does not veto on it: an item
  scoring `0.0` is still fetched. The rule is deliberately url-only — the same page found by a
  *search* is scored out exactly as before.
- **Nothing else is special.** All three dedupe layers still apply (candidate
  `UNIQUE(kind, value, site)` after canonicalisation, `items` by canonical URL, `content_sha`
  across URLs), and fetch → `raw/<date>/` → pack → notes → wiki is the ordinary path. A URL
  already in `items` is reported as seen and is not fetched again.

## The daily run

`scripts/material-daily.sh` is the cron entrypoint. It is a wrapper over four CLI calls, and
it exists to add the four things cron needs: a lock, budgets, an ordering rule, and one line
of output.

```bash
bash scripts/material-daily.sh               # collect → distill → memorize → report
bash scripts/material-daily.sh --dry-run     # every step in plan mode, nothing written or posted
bash scripts/material-daily.sh --limit 10    # more queries this run
MATERIAL_SKIP_DISTILL=1 bash scripts/material-daily.sh   # collect only
```

What the script guarantees:

- **One run at a time.** `flock -n` on `docs/material/.state/lock`. A second run — cron firing
  while a hand-started run is still in the agent — prints `material: another run holds the
  lock` and exits **0**, because a cron job that fails on contention retries and storms.
  A dry run writes no lock file of its own, so it joins a lock only if one already exists.
- **Ordering with a purpose.** `run` → `distill` → `memorize` → `report`. A failed `run` skips
  `distill` (a pack built from a broken collect wastes an agent run) but still runs `memorize`
  and `report`, because those walk every `distilled` item from earlier days and are
  independent of today's network luck.
- **Budgets.** Each step runs under `timeout`: `MATERIAL_TIMEOUT_COLLECT=1800`,
  `MATERIAL_TIMEOUT_DISTILL=1200`, `MATERIAL_TIMEOUT_MEMORIZE=300`,
  `MATERIAL_TIMEOUT_REPORT=60`. They sum to 3360 s, inside Hermes cron's 3600 s script
  timeout, so a hung step is killed by the inner budget and the outer kill stays a backstop.
  These are **budgets, not measurements** — the only honest number for a real run is the
  `92 s`-style figure in the report line.
- **One line on stdout.** The report. Everything else — progress, the failing step, the reason
  — goes to stderr. That is a deliberate fit with Hermes cron: `--no-agent` jobs deliver
  stdout verbatim on success and include *both* streams in the alert on a non-zero exit, so a
  green day costs one short message and no model tokens, and a red day explains itself.

### Wiring it

Hermes cron jails `--script` to `~/.hermes/scripts/`, so the repo script is reached through a
one-line wrapper that already exists:

```bash
# ~/.hermes/scripts/material-daily.sh
exec bash /home/boris/WORKSPACE/machinum-sft/scripts/material-daily.sh "$@"
```

The job — 13:00 local, no LLM in the loop (`no_agent: true`), already created as
`machinum-material-daily`:

```bash
hermes cron create '0 13 * * *' \
  --name machinum-material-daily \
  --workdir /home/boris/WORKSPACE/machinum-sft \
  --script material-daily.sh \
  --no-agent \
  --deliver 'telegram:Boris Velichko (dm)'
```

`hermes cron list` should show it with `next_run_at` at the next 13:00. `--no-agent` matters
twice: the run costs no model tokens, and stdout is delivered verbatim, so the report line is
literally the message. On a non-zero exit Hermes sends an alert instead (with both streams),
which is the red-path notification.

Plain crontab equivalent, if the Hermes scheduler is not running (`hermes cron status`):

```cron
0 13 * * * cd /home/boris/WORKSPACE/machinum-sft && scripts/material-daily.sh 2>>$HOME/.local/state/material-daily.err.log
```

stdout still reaches cron mail; stderr appends to the log. Either way the flock makes a
catch-up run after a laptop that was asleep at 13:00 safe to fire next to a manual run.

First-time setup, once: `msft material seed` (the script does not seed — a daily run that
silently creates a frontier hides a typo in `configs/material.yaml`), then
`msft material run --limit 3 --dry-run` to see the plan without the network.

### When a run is red

| What you see | What it means | First command |
| --- | --- | --- |
| `another run holds the lock`, exit 0 | a run is in flight, or a killed run left a live process | `pgrep -af 'material'`; if nothing, `fuser docs/material/.state/lock` |
| exit 2, `error: …` on stderr | config or environment: unreadable YAML, missing token, unwritable dir | `msft material status` then re-read the message; the CLI prints the offending path |
| exit 1, `collect failed` | webbridge: site down, browser not attached, JSON garbage — **not** an empty search (zero results are a `done` candidate) | `msft material run --limit 1 --site habr` and read the candidate's `error` column |
| `distill: skipped (collect failed)` | correct behaviour, not a bug | nothing; tomorrow's run picks the `packed` items up |
| `at` non-zero / no summary | agent crashed, timed out, or printed no JSON. The note files are the evidence, not the exit code: `distill` books every note that landed, so a kill *after* the notes were written is a success with `agent_failed: true` and the items go to `distilled`. Only notes that are missing leave their item `packed` | read `docs/material/pack/<date>.md` and `docs/material/notes/<date>/`, then run `at` by hand on that file with a bigger `--timeout` |
| items stuck `packed` for days | the agent never accepted the pack | `SELECT url, error FROM items WHERE status='packed'` |
| `memorize` 4xx | ai-memory down, wrong workspace/project, stale token | `curl -s localhost:49374/healthz`, then `memory-stack.md` |
| `memorize` 5xx | the wiki store refused the write; the note file is kept | re-run `memorize` after fixing the service — it is idempotent by path |
| an item is `fetched` but its `.txt` is a few bytes | the browser tab was still on another page: `extract` returned that page's payload (`found: false`, a foreign `url`, `text: '""'`). `_extract_error` now refuses all three as retryable errors, so this row means it happened *before* that guard | `SELECT url, error FROM items WHERE status='error'`; the next run retries (`limits.retry_errors`) |
| everything green, report says `0 items` | the sites answered and nothing scored | `items` rows with `scored_out` say why; usually `keep_min` |
| everything green, report says `0 queries` | the frontier had nothing due: every candidate is `done` and inside `limits.refresh_days`, and synthesis is off or full | `msft material status` (frontier count), then `limits.refresh_days` / `limits.synthesize_queries` in `configs/material.yaml`; a real run also prints `frontier empty: nothing is due` on stderr |

Never "fix" a red run by deleting `docs/material/.state/material.db`. The DB is the record of
what was already read; deleting it re-fetches everything and re-posts nothing (paths dedupe),
so it is a slow way to look busy.

## Inspecting state

The state is one SQLite file, and the schema is the spec's — four tables, no views. This host
has no `sqlite3` binary, so the shell is `python3 -m sqlite3` (Python ≥3.12); the SQL is
identical if you do have the CLI:

```bash
DB=docs/material/.state/material.db
python3 -m sqlite3 "$DB" "SELECT status, COUNT(*) FROM candidates GROUP BY status;"
```

```sql
-- the frontier: what is queued, what is done, what failed
SELECT status, COUNT(*) AS n FROM candidates GROUP BY status ORDER BY n DESC;

SELECT priority, kind, site, value FROM candidates
WHERE status = 'new' ORDER BY priority DESC, id LIMIT 20;

SELECT site, kind, value, attempts, substr(error, 1, 120) AS why
FROM candidates WHERE status = 'error' ORDER BY updated_at DESC LIMIT 10;

-- the items: the funnel, per date and per site
SELECT status, COUNT(*) AS n FROM items GROUP BY status ORDER BY n DESC;

SELECT site, COUNT(*) AS n FROM items
WHERE first_seen_at >= '2026-11-10' GROUP BY site ORDER BY n DESC;

-- why the funnel narrowed
SELECT url, site, score, title FROM items
WHERE status = 'scored_out' ORDER BY score DESC LIMIT 20;

-- comments and links: what each page carried
SELECT url, comments FROM items WHERE comments > 0 ORDER BY comments DESC LIMIT 10;

SELECT c.parent AS item, c.kind, c.site, c.value
FROM candidates c WHERE c.reason LIKE 'link:%' OR c.reason LIKE 'hint:%'
ORDER BY c.parent, c.id;

-- the shape of one day, without the SQL
-- msft material tree --date 2026-11-12

SELECT url, site, content_sha FROM items WHERE status = 'duplicate';

-- what the agent owes us
SELECT url, site, score, fetched_at FROM items WHERE status = 'packed' ORDER BY score DESC;

SELECT url, site, substr(error, 1, 120) AS why FROM items WHERE status = 'error';

-- what reached the wiki, by day
SELECT substr(memorized_at, 1, 10) AS day, COUNT(*) AS n FROM items
WHERE memorized_at IS NOT NULL GROUP BY day ORDER BY day DESC LIMIT 7;

-- the last runs, and one counter out of the stored JSON
SELECT id, started_at, finished_at, counters FROM runs ORDER BY id DESC LIMIT 3;

SELECT json_extract(counters, '$.items') AS items,
       json_extract(counters, '$.kept')  AS kept
FROM runs ORDER BY id DESC LIMIT 1;

-- small key/value leftovers (last seed time, schema version)
SELECT key, value FROM kv ORDER BY key;
```

`msft material status --json` prints the same counters without SQL, which is what a dashboard
should use; the queries above are for the "why" that a counter cannot answer.

## What this does not do

- **It does not train anything.** A note's `## Experiment` section is a proposal with a cost
  estimate, appended to `experiments.md`; running it is a human decision, and `AGENTS.md`
  still governs the GPU.
- **It does not read the sites itself.** It shells out to
  `/home/boris/tunnelstack/src/webbridge.py` and takes the JSON. When a site changes its HTML,
  the fix belongs in tunnelstack, not here — this repo only sees a candidate in `error`.
- **It does not judge.** A score is a keyword count, a site prior, a recency bonus and a star
  bonus. A post that is wrong but well-keyworded gets through; the agent's note is where
  "this is marketing" gets written down, and a human reading `INDEX.md` is where it gets
  acted on.
- **It does not replace `memory.md`.** Gotchas that cost a run belong in that file, in git,
  reviewed in a diff. The wiki is the journal; the docs are the record.
