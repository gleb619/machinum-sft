# Project memory: one container, so the next session starts where this one stopped

The 2026-11-08 run cost a day to discoveries that were already in someone's head twice over:
transformers ≥ 5.2, hand-written chat templates, `think: false`, the ephemeral container layer.
None of that is model knowledge — it is project knowledge, and it belongs in a store the next
agent reads at session start rather than in a transcript nobody scrolls.

## What is running

| | |
| --- | --- |
| Service | [ai-memory](https://github.com/akitaonrails/ai-memory) 3.4.4, `akitaonrails/ai-memory` |
| Containers | **1** (no database container — embedded store, markdown wiki) |
| Port | `127.0.0.1:49374` — MCP at `/mcp`, browser at `/web`, health at `/healthz` |
| Storage | `deploy/memory/data/` (bind mount), wiki is plain markdown under `data/wiki/` |
| Cost of a write | zero LLM calls, zero API keys |
| Compose | `deploy/memory/docker-compose.yml` |

```bash
cd deploy/memory
cp .env.example .env
docker run --rm akitaonrails/ai-memory:3.4.4 generate-auth-token   # paste into .env
mkdir -p data && docker compose up -d
curl -s localhost:49374/healthz            # {"status":"ok"}
```

The data directory must be writable by uid 1000, which is the image's user and usually you.
If Docker created it as root, remove the empty directory and create it yourself before the
first `up` — that is the one failure this stack has produced so far (`Permission denied:
opening serve lock /data/.serve.lock`).

## Seeding it from this repo

The wiki is seeded from the documents that already exist, so nothing is written twice:

```bash
scripts/journal-seed.sh --dry-run     # what would be written
scripts/journal-seed.sh               # write/update every page
```

| Page | Source | Kind |
| --- | --- | --- |
| `projects/machinum-sft/gotchas.md` | `docs/memory.md` | gotcha |
| `projects/machinum-sft/handoff.md` | `docs/handoff.md` | rule |
| `projects/machinum-sft/agents-rules.md` | `AGENTS.md` | rule |
| `projects/machinum-sft/run-2026-11-08.md` | `docs/run-2026-11-08.md` | fact |
| `projects/machinum-sft/cli-audit.md` | `docs/cli-audit.md` | decision |
| `projects/machinum-sft/model-choice.md` | `docs/model-choice.md` | decision |

`write-page` in the image is an HTTP client, not a local writer — the script runs the helper
container with `--network host` so `127.0.0.1:49374` reaches the published port. Verified
retrieval:

```
$ ai-memory search "transformers 5.2"
5 result(s) for "transformers 5.2":
  projects/machinum-sft/gotchas.md   …
```

Re-run the script after editing any of those documents; pages are replaced by path, so it is
idempotent.

## Reading it at the start of the next session

The server speaks MCP over streamable HTTP:

```bash
# from the repo, with the token from deploy/memory/.env
export AI_MEMORY_SERVER_URL=http://127.0.0.1:49374
export AI_MEMORY_AUTH_TOKEN=$(grep -m1 AI_MEMORY_AUTH_TOKEN deploy/memory/.env | cut -d= -f2-)
ai-memory search "why did the 4B job die"      # or via the image, as the seed script does
```

It is **already registered** with Hermes as the MCP server `ai-memory` (27 tools:
`memory_search`, `memory_write_page`, `memory_read_page`, `memory_status`, handoffs, …), so a
new session can query the journal directly:

```bash
hermes mcp list        # ai-memory  http://127.0.0.1:49374/mcp  all  enabled
hermes mcp test ai-memory
```

Existing sessions pick the tools up with `/reload-mcp`. The registration is one line in
`~/.hermes/config.yaml`, and it was made non-interactively:

```bash
TOKEN=$(grep -m1 AI_MEMORY_AUTH_TOKEN deploy/memory/.env | cut -d= -f2-)
printf 'y\n%s\ny\n' "$TOKEN" | hermes mcp add ai-memory --url http://127.0.0.1:49374/mcp --auth header
```

The browser view at <http://127.0.0.1:49374/web> is the fastest way for a human to check what
the agent will recall, and it is read-only.

## What it is not

- **Not a transcript archive.** It stores pages we choose to write, not every turn. That is the
  point: `docs/memory.md` is 7 KB of decisions, while the session that produced it is megabytes
  of dead ends.
- **Not a vector database.** Retrieval is FTS5 + entities + links; embeddings are optional and
  off. Good enough for "what did we learn about transformers versions", and it needs no GPU —
  which matters on a box whose two cards are busy training.
- **Not exposed beyond loopback.** The container binds `127.0.0.1` only. If it ever needs to be
  reachable from another machine, put the Caddy TLS template from the project's `docker/`
  directory in front of it and set `AI_MEMORY_AUTH__SECURE_COOKIE=true`.

## Why this one, out of the thirteen considered

A decision table was produced by checking each project's repo and compose files, not its
marketing page. The shortlist:

| | containers | LLM key at write | REST | verdict |
| --- | --- | --- | --- | --- |
| **ai-memory** | **1** | no | MCP-over-HTTP + web | chosen |
| Hindsight | 1 (embedded Postgres) | yes, fact extraction | OpenAPI | fallback: LLM competes with training for VRAM |
| Cognee (full) | 4–5 | yes | yes | over budget |
| Graphiti + FalkorDB | 2 | yes | reference only | over budget, needs OpenAI key |
| Mem0 | 2 (server + pgvector) | yes | yes | an extra container for facts we can write as prose |
| Letta, LangMem, LlamaIndex Memory | 0 | yes | no | libraries or runtimes, not a service |
| Zep (self-host) | — | — | — | the OSS server no longer exists in that repo |
| HydraDB | 2+ | no | DB protocol | a graph database, not a memory layer |
| OpenViking | 2 | yes | yes | AGPL-3.0, VLM in the loop |
| Evermind | 1 | yes | yes | no shipped image; you build it |
| MemoryPalace, MemoClaw | — | — | — | no canonical public repo |

The deciding property is not retrieval quality — every vendor publishes a benchmark it wins.
It is that a journal written by an agent at 23:00 must not need an API key, must not take a GPU,
and must be greppable when the agent is gone. ai-memory is markdown + SQLite in one binary, so
`cp -a deploy/memory/data /somewhere` is the backup and `grep -r "transformers" data/wiki` works
without the service running.

## Operating it

```bash
cd deploy/memory
docker compose ps          # expect: Up (healthy)
docker compose logs -f     # MCP ready on /mcp, /web mounted
docker compose down        # data/ survives; rm -rf data wipes the journal
```

Backup: `tar czf memory-$(date +%F).tgz -C deploy/memory data` (the wiki is the source of
truth; the SQLite index can be rebuilt).
