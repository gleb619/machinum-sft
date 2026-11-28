# Kanban Board

Lightweight planner for `at` implementation tasks. Machine-mutated by `at kanban` only.

* High-level planner only — details live in [`./docs/plans`](./plans).
* The agent **must** move `Status` when taking a task and again when the work is done.
* Rows stay sorted by `ID`; `at kanban add` enforces this mechanically.

## Status Legend

| Status        | Meaning                                                  | Who updates it  |
| ------------- | -------------------------------------------------------- | --------------- |
| `Backlog`     | Task is defined but not yet started.                     | Planner / agent |
| `In Progress` | An agent is actively working the task.                   | Coding agent    |
| `Blocked`     | Cannot continue until another task or issue is resolved. | Coding agent    |
| `Done`        | Task completed and accepted.                             | Coding agent    |

## Task Board

| ID | Status | Title | Description | Priority | Channel | Blocks | Blocked By | Plan File |
| :- | :----- | :---- | :---------- | :------- | :------ | :----- | :--------- | :-------- |
| T1 | Backlog | material: fetch requeued items without spending a query | retry flips error→new; refetch needs run --limit 1, which also searches. Add a pending-only fetch (or retry --fetch). | Low      | loop    | -      | -          | -         |
| T2 | Backlog | material status: probe webbridge extension, not just the daemon | status said 'webbridge ok' while two rows failed on 'extension not connected'; name the extension state.              | Low      | loop    | -      | -          | -         |
| T3 | Backlog | material: pack body eats \<user>/\<repo>:\<quant> placeholders  | Pack shows 'hf.co / :' where comments.json has hf.co\<user>/\<repo>:\<quant> — digest renderer strips it.             | Low      | loop    | -      | -          | -         |
| T4 | Backlog | material: reject non-URL candidates at ingest                   | candidate 198 value is the regex fragment 'llama3-3)\*.\*'; a 404 makes collect rc=1 and the hermes fire red          | Low      | loop    | -      | -          | -         |
| T5 | Done        | translation: glossary/TM retrieval with Giga-Embeddings-480M    | Spike: embed glossary.json + chapters with Giga-Embeddings-480M; pick per-chapter terms, measure recall\@k and prompt saving. | Low      | loop    | -      | -          | plans/T5-translation-glossary-tm-retrieval-with-g.md |
| T6 | Backlog | probe --dry-run still calls the endpoint and writes predictions | dry-run probe sent 11 requests and wrote docs/probe-predictions.jsonl; plan mode must touch nothing                           | Low      | loop    | -      | -          | -                                                    |
| T7 | Backlog | translation: plain-mode block contradicts one-line-per-input-line | W1: 12/50 book2 windows collapse 2 lines to 1; product plain block says keep ONE line. Fix the prompt.                        | High     | loop    | -      | -          | -                                                    |
| T8 | Backlog | translation: run the occurrence scan on untagged units            | W3/T5 measured: on 21 untagged chapters the tag filter yields 0 glossary terms, while glossary-index.ts:230 occurrence scan r... | High     | loop    | -      | -          | -                                                    |
| T9 | Done    | eval: terms_block fails on every gold reference (W4 protocols)    | W4b measured: predicting the gold reference for proofread/typofix gives terms_block 0/2 and schema_validity 0/8; NER replies ... | High     | loop    | -      | -          | -                                                    |
| T10 | Backlog | W4c: arm-A data (42 rec) is below the plan's --min-per-arm 60 gate | W4b dry-run: arm a = summarize 21 + ner 21 = 42 records; msft data slice --min-per-arm 60 exits 1 'arm a has 42 record(s), be... | High     | loop    | -      | -          | -                                                    |
| T11 | Backlog | eval: no per-task gate breakdown in the output                     | W4d: the gate table is one roll-up; to print pass/fail per gate per task I had to write out/weekend/W4d/per-task.py. msft eva... | Low      | loop    | -      | -          | -                                                    |
| T12 | Backlog | guard: pytest -q plus addopts '-q' drops the pass count            | guard JSON reports 19.6s but no 'N passed': pyproject addopts=-q makes 'pytest -q' = -qq, which suppresses the summary line. ... | Low      | loop    | -      | -          | -                                                    |
| T13 | Backlog | adapters merge self-check 3 crashed: apply_chat_template missing tokenize=False | W4c run: generated merge_script.py self-check 3 fed apply_chat_template's default tokenize=True id list back into the tokeniz... | Low      | loop    | -      | -          | -                                                    |
| T14 | Backlog | adapters: build_script(base_model=None) emits invalid python                    | src/msft/adapters.py build_script() json.dumps(base_model); with base_model None it writes 'BASE_FALLBACK = null', which is a... | Low      | loop    | -      | -          | docs/plans/weekend-2026-11-28.md                     |
| T15 | Backlog | worker: the same job id queued twice runs twice                                 | w6-train2 was enqueued at 08:45:27 and again at 08:46:43; both ran (17.9 min each) and the second overwrote logs/w6-train2.lo... | Medium   | loop    | -      | -          | -                                                    |
| T16 | Backlog | eval: summarize item count is not gated — w6 answered 19 items for 'exactly 10' and passed | W6: machinum:4b-w6-misc summarize reply had 19 dash items where the prompt says exactly 10 and the reference has 10; no_chatt... | Medium   | loop    | -      | -          | -                                                    |
