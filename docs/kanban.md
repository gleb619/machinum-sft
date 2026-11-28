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
