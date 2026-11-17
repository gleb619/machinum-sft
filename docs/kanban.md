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

| ID  | Status | Title                                                 | Description                                                                     | Priority | Channel     | Blocks | Blocked By | Plan File |
| :-- | :----- | :---------------------------------------------------- | :------------------------------------------------------------------------------ | :------- | :---------- | :----- | :--------- | :-------- |
| T1  | Done   | scaffold the machinum-sft CLI and config contract     | Packaging, pyproject, config contract and gitignore for the first commit        | High     | cli         | -      | -          | -         |
| T2  | Done   | build the CLI, dataset builder and training stages    | CLI, data builder, doctor, train/merge/quantize, evaluate gates and configs     | High     | cli         | -      | -          | -         |
| T3  | Done   | add the Qwen3.5-4B config and the book-pair script    | 4B config, base_dir resolution, data-flow docs and pairs_from_book              | High     | configs     | -      | -          | -         |
| T4  | Done   | verify the remote smoke run and record VRAM           | VRAM note for bf16 vs 4-bit, plus the verified remote smoke run                 | Medium   | train       | -      | -          | -         |
| T5  | Done   | mask hostnames and degrade dry-run plans              | Hide remote hostnames; a missing tool degrades the plan instead of erroring     | Medium   | cli         | -      | -          | -         |
| T6  | Done   | ship the worker queue and remote deploy/signal/status | Job-queue daemon, remote deploy/signal/status, docker compose for the GPU box   | High     | worker      | -      | -          | -         |
| T7  | Done   | add ollama export and worker status scripts           | ollama export and worker status report scripts                                  | Medium   | worker      | -      | -          | -         |
| T8  | Done   | export through a containerised ollama                 | ollama export works when ollama runs in a container                             | Medium   | ollama      | -      | -          | -         |
| T9  | Done   | install the mounted package with uv in the entrypoint | pip is absent in a uv venv, so the entrypoint must install with uv              | Medium   | deploy      | -      | -          | -         |
| T10 | Done   | map the data job to the data build action             | The worker maps a data job to the data build action                             | Medium   | worker      | -      | -          | -         |
| T11 | Done   | keep jsonl fixtures tracked for CI parity             | Local emulation and CI agree on the tracked jsonl fixtures                      | Medium   | ci          | -      | -          | -         |
| T12 | Done   | document the worker queue and its failure modes       | Queue design, commands and the failure modes hit during the first deploy        | Low      | docs        | -      | -          | -         |
| T13 | Done   | add the Qwen3.5-0.8B fallback config                  | A fast end-to-end artifact for pipeline checks                                  | Medium   | configs     | -      | -          | -         |
| T14 | Done   | point the worker at the image's llama.cpp build       | The worker uses the llama.cpp built inside the image                            | Medium   | deploy      | -      | -          | -         |
| T15 | Done   | refuse to train on CPU and surface the device         | A CPU run is a bug: refuse without CUDA and print the device                    | High     | train       | -      | -          | -         |
| T16 | Done   | normalise tokenizer_config after merge                | Make GGUF conversion work after a merge                                         | High     | merge       | -      | -          | -         |
| T17 | Done   | treat narrated reasoning as chatter in evaluate       | Narrated reasoning must not pass the evaluation gates                           | Medium   | evaluate    | -      | -          | -         |
| T18 | Done   | write the 2026-11-08 run report                       | What trained, what broke and what is next                                       | Low      | docs        | -      | -          | -         |
| T19 | Done   | let the GGUF's own chat template drive ollama         | Never hand-write a chat template; ship the tokenizer's own                      | High     | ollama      | -      | -          | -         |
| T20 | Done   | pass the 4B gates with think:false                    | The 4B artifact clears the gates with thinking off and its own template         | High     | evaluate    | -      | -          | -         |
| T21 | Done   | record handoff, memory and agent rules                | The working loop and its traps from the first real run                          | Medium   | docs        | -      | -          | -         |
| T22 | Done   | make fetch, verify, export and probe CLI commands     | The manual artifact steps become subcommands                                    | High     | cli         | -      | -          | -         |
| T23 | Done   | match ollama list rows by model and tag               | Match both model and tag, not just the model name                               | Medium   | ollama      | -      | -          | -         |
| T24 | Done   | ship the one-container project journal                | A project journal seeded from the repo's own notes                              | Medium   | memory      | -      | -          | -         |
| T25 | Done   | document the journal's Hermes MCP wiring              | Record that the journal is wired into Hermes as an MCP server                   | Low      | docs        | -      | -          | -         |
| T26 | Done   | add the editorial glossary name table                 | A name table plus the tooling that keeps the glossary honest                    | Medium   | glossary    | -      | -          | -         |
| T27 | Done   | audit a finished book and record the 21-chapter run   | Check a finished book and record the run                                        | High     | audit       | -      | -          | -         |
| T28 | Done   | close fourteen review defects                         | Correctness fixes across the CLI, worker and artifact chain                     | High     | cli         | -      | -          | -         |
| T29 | Done   | add job timeout, safe sync and a loss mask            | Per-job timeout kills the process group; sync cannot leave data/                | High     | worker      | -      | -          | -         |
| T30 | Done   | unwrap the processor before tokenising a chat row     | Tokenise the chat row through the unwrapped processor                           | Medium   | train       | -      | -          | -         |
| T31 | Done   | train on the reply only and stamp provenance          | Completion-only loss and a provenance sidecar for every GGUF                    | High     | train       | -      | -          | -         |
| T32 | Done   | add three remote run modes                            | Foreground, detached tmux and the worker queue side by side                     | High     | remote      | -      | -          | -         |
| T33 | Done   | launch a script in detached tmux                      | A detached tmux pane has no tty, so launch a script instead of a quoted command | Medium   | remote      | -      | -          | -         |
| T34 | Done   | resolve a relative repo_dir against $HOME             | A relative repo_dir means $HOME, not the shell's cwd                            | Medium   | remote      | -      | -          | -         |
| T35 | Done   | build the daily material research state machine       | The daily state machine behind \`msft material\`                                | High     | material    | -      | -          | -         |
| T36 | Done   | retry an errored frontier node                        | Retry an errored frontier node before giving up                                 | Medium   | material    | -      | -          | -         |
| T37 | Done   | treat a zero-result search as done                    | A zero-result search is a done candidate, not a failed run                      | Medium   | material    | -      | -          | -         |
| T38 | Done   | read one page on purpose with material add            | \`msft material add\` reads one page deliberately                               | Medium   | material    | -      | -          | -         |
| T39 | Done   | compare models on the same prompts                    | Score the same prompts with separate framework scorers                          | High     | arena       | -      | -          | -         |
| T40 | Done   | add multi-line samples and the two v2 arms            | Multi-line samples plus the v2 arms the arena will judge                        | Medium   | data        | -      | -          | -         |
| T41 | Done   | add a heretic twin of the merged checkpoint           | A refusal-ablated model without OBLITERATUS                                     | High     | abliterate  | -      | -          | -         |
| T42 | Done   | load the fast tokenizer directly                      | AutoTokenizer recurses on this multimodal checkpoint                            | Medium   | abliterate  | -      | -          | -         |
| T43 | Done   | place the model across both cards                     | Each direction moves to its matrix's device                                     | Medium   | abliterate  | -      | -          | -         |
| T44 | Done   | read the text config through getattr                  | The loader already unwrapped the multimodal model                               | Medium   | abliterate  | -      | -          | -         |
| T45 | Done   | accept wrapped and unwrapped text models              | Handle both shapes of the abliteration input                                    | Medium   | abliterate  | -      | -          | -         |
| T46 | Done   | measure refusals with thinking disabled               | Measure refusals the way the serving path runs                                  | Medium   | abliterate  | -      | -          | -         |
| T47 | Done   | document the four machinum tags                       | What the second run actually proved                                             | Low      | docs        | -      | -          | -         |
| T48 | Done   | make the daily loop re-ask and widen                  | The daily loop re-asks its questions and widens the pool                        | Medium   | material    | -      | -          | -         |
| T49 | Done   | book timed-out distill notes and give it time         | A timed-out distill agent's notes are still booked                              | Medium   | material    | -      | -          | -         |
| T50 | Done   | add a costed experiment queue                         | A costed GPU experiment queue and two new arena axes                            | High     | experiments | -      | -          | -         |
| T51 | Done   | pass --execute from the cron entrypoint               | A quiet day and a missing --execute both had no work                            | Medium   | experiments | -      | -          | -         |
| T52 | Done   | fold the catalog fallback check                       | One condition instead of two (SIM102)                                           | Low      | experiments | -      | -          | -         |
| T53 | Done   | read comments, follow links, re-target                | Read the comments, follow the links and re-target the research                  | Medium   | material    | -      | -          | -         |
| T54 | Done   | read a tweet as material with links                   | A tweet becomes material, links and all                                         | Medium   | material    | -      | -          | -         |
| T55 | Done   | read a vendor docs site via its markdown twin         | Read a vendor docs site through its markdown twin                               | Medium   | material    | -      | -          | -         |
| T56 | Done   | survive an unreadable link and fit argv               | One unreadable link must not fail the daily run                                 | Medium   | material    | -      | -          | -         |
| T57 | Done   | read hft.md and promote a dead hint                   | An explicit add promotes a dead hint                                            | Medium   | material    | -      | -          | -         |
| T58 | Done   | share the pack text budget                            | Share the pack's budget and keep a whole catalogue page                         | Medium   | material    | -      | -          | -         |
| T59 | Done   | re-read a page on re-queue                            | Re-queuing an item re-reads the page, never its own summary                     | Medium   | material    | -      | -          | -         |
| T60 | Done   | close the terms block and count clean lines           | A terms block must be closed and non-empty; parity counts cleaned lines         | Medium   | evaluate    | -      | -          | -         |
| T61 | Done   | split train/val by passage intervals                  | Never split by per-record digest; use merged passage intervals                  | High     | data        | -      | -          | -         |
| T62 | Done   | state the eval predictions file                       | The eval recipe states the file it cannot produce                               | Low      | experiments | -      | -          | -         |
| T63 | Done   | install a git-ignored training config                 | Install the local training config and prove the box has it                      | Medium   | configs     | -      | -          | -         |
| T64 | Done   | join a tweet's replies into the material              | Replies join the raw text, the count and the link net                           | Medium   | material    | -      | -          | -         |
