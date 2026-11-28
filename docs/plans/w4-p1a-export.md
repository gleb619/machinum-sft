# W4 / P1-a — export summarize + NER training triples

You are executing this export. Work in the repo root
`/media/boris/data_nvme1n1/WORKSPACES/WORKSPACE/machinum-sft`. Owner approved P1-a at 08:35 on
2026-11-28. Do not commit or push; the keeper commits after review.

## Why this is a teacher export, not a file copy

There are no real summarize or NER exchanges on disk. The only pipeline run that executed the
summary stage is `samples/sample2-practical/.mt/runs/bb6a34a7-20261127-rose-uqnn/steps.jsonl`, and
its summary variable records `{"fallbackUsed": true, "attempts": 63}` — the "summary" is the
chapter text, so it cannot become a triple. `chapters/schema/*.entities.json` holds NER replies
but not the prompts. Therefore P1-a is realised as a **teacher export with the product's own
prompts**: the same `summary.*` / `glossary.*` blocks the pipeline sends, rendered here, answered
by the shipped `machinum:4b` tag at temperature 0 with `think:false`.

Record this deviation, the teacher tag and every prompt sha256 in the result file.

## Inputs

- Chapters: `/home/boris/WORKSPACE/machinum-sft-loop/books/book2/chapter-<N>.en.md`, N = 1..21
  (21 files; book2 is not in any training corpus).
- Prompt blocks: `/home/boris/WORKSPACE/machinum-pipeline/packages/tools/prompts/`
  (`summary.system.md`, `summary.user.md`, `summary.output.bullets.md`, `glossary.system.md`,
  `glossary.user.md`, `llm.format.plain.md`).
- Teacher: ollama `machinum:4b` on the box, OpenAI-free `/api/chat`, `think:false`, temperature 0,
  `stream:false`. Resolve the host exactly as the repo does — never print it, never read
  `configs/worker.local.yaml`:
  `H=$(.venv/bin/python -c "from msft.remote import remote_host;print(remote_host().split('@')[-1])")`
  and use `http://$H:7869`. `--endpoint` may override.
- Normaliser: pipeline TL254 `buildSftTriples`, already built at
  `/home/boris/WORKSPACE/machinum-pipeline/packages/tools/dist/modification/sft-triples.js`
  (node v22). It accepts `{system,user,reply}` and emits `{messages:[...]}`.

## Prompt composition (must match the pipeline)

`summary` (the notes tool, `summary.output.bullets`, `DEFAULT_NOTE_COUNT = 10`):

- system = render(`summary.system`) + `"\n\n"` + render(`llm.format.plain`) + `"\n\n"` +
  render(`summary.output.bullets`)
- user = render(`summary.user`, `{count: 10, text: chapter}`)

`ner`:

- system = render(`glossary.system`)
- user = render(`glossary.user`, `{known: "none", candidates: "none", text: chapter}`)
  (`nameList([])` is the literal `none`).

`render` is plain `{{var}}` substitution; unmatched slots stay untouched.

## Deliverables

1. `scripts/export_protocol_triples.py` — stdlib only, no new dependencies. Plan by default;
   `--execute` performs the requests. Flags: `--protocol summary|ner` (repeatable or `all`),
   `--teacher`, `--endpoint`, `--chapters DIR`, `--prompts DIR`, `--tl254 PATH`,
   `--retries 3`, `--limit N`, `--out DIR`. It must:
   - write raw exchanges `out/export/<protocol>/raw.jsonl` (`{protocol, chapter, system, user,
     reply}`) and the normalised triples to `data/<protocol>-v1/<protocol>.jsonl`;
   - validate a reply before accepting it: summary = at least 3 lines starting `- `; ner = parses
     as a JSON array of objects with a `name`; retry up to `--retries`, then count it failed and
     exit 1 without writing a partial line for it;
   - send `think:false`, `stream:false`, temperature 0; never print the endpoint host;
   - normalise with TL254 by piping raw.jsonl through node `buildSftTriples`, print
     `recordsIn/recordsOut/dropped`, and refuse to write if `recordsOut < 20`;
   - write atomically (temp file + `os.replace`), print one line per item (`ok chapter=N
     chars=…`) and a final summary.
2. `tests/test_export_protocol_triples.py` — torch-free, no network: prompt rendering matches a
   fixture, a prompt-echo reply and an empty reply are rejected, a good summary/NER reply passes,
   and the output is one `{messages}` line per accepted item. Run only this test file.
3. Run it for both protocols over the 21 chapters (42 calls, ~10 min) and leave the results at
   `data/summaries-v1/summaries.jsonl` and `data/ner-v1/ner.jsonl` (both under git-ignored
   `data/`).
4. `out/weekend/W4a/RESULT.md` with: exact commands, rc, wall, per-protocol `recordsIn/recordsOut/
   dropped`, line counts, sha256 of both outputs, the teacher tag, the six prompt sha256s, and
   one line per chapter skipped if any. End with `RESULT: done` or `RESULT: failed: <reason>`.

## Acceptance

- `wc -l data/summaries-v1/summaries.jsonl` → 21; `data/ner-v1/ner.jsonl` → 21.
- `node -e` over each file shows every line `{messages:[...]}`, last role `assistant`, non-empty,
  and no assistant content equal to the user content.
- The TL254 stats lines show `recordsOut == recordsIn` for both raw files (no drops).
- `bash scripts/guard.sh --allow-dirty` is green after the script and test land.
