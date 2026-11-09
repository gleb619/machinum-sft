#!/usr/bin/env bash
# Seed the project journal from the repository's own notes.
#
#   scripts/journal-seed.sh                 # write/update every page
#   scripts/journal-seed.sh --dry-run       # print what it would write
#
# The point: the next session should read the wiki, not re-derive the traps from a failed
# run. Each source document becomes one page under projects/machinum-sft/, with the kind in
# frontmatter so retrieval can prefer decisions and gotchas.
#
# Requires the stack from deploy/memory/docker-compose.yml to be running: `write-page` in the
# image is a thin HTTP client, not a local writer (it posts to /admin/write-page). The helper
# container therefore runs with --network host so 127.0.0.1 reaches the published port, and
# carries the token from deploy/memory/.env.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STACK="$REPO/deploy/memory"
ENV_FILE="$STACK/.env"
IMAGE="akitaonrails/ai-memory:3.4.4"
SERVER="${AI_MEMORY_SERVER_URL:-http://127.0.0.1:49374}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# source file | wiki page | kind | tags
PAGES=(
  "$REPO/docs/memory.md|projects/machinum-sft/gotchas.md|gotcha|gotcha,caveats,training"
  "$REPO/docs/handoff.md|projects/machinum-sft/handoff.md|rule|handoff,workflow,remote"
  "$REPO/docs/run-2026-11-08.md|projects/machinum-sft/run-2026-11-08.md|fact|run,numbers,qwen3.5-4b"
  "$REPO/docs/cli-audit.md|projects/machinum-sft/cli-audit.md|decision|cli,audit,automation"
  "$REPO/docs/model-choice.md|projects/machinum-sft/model-choice.md|decision|model,qwen,gemma,vram"
  "$REPO/AGENTS.md|projects/machinum-sft/agents-rules.md|rule|agents,rules,guard"
  "$REPO/docs/pipeline-check.md|projects/machinum-sft/pipeline-check.md|fact|mt,pipeline,quality,eval"
  "$REPO/docs/model-candidates.md|projects/machinum-sft/model-candidates.md|decision|models,vram,registry"
)

if [ "$DRY_RUN" -eq 1 ]; then
  for entry in "${PAGES[@]}"; do
    IFS='|' read -r src path kind tags <<<"$entry"
    printf 'would write %-46s <- %s (%s bytes, kind=%s)\n' "$path" "${src#$REPO/}" "$(wc -c <"$src")" "$kind"
  done
  exit 0
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "missing $ENV_FILE — copy .env.example and put a token in it" >&2
  exit 2
fi
TOKEN=$(grep -m1 '^AI_MEMORY_AUTH_TOKEN=' "$ENV_FILE" | cut -d= -f2-)
if [ -z "$TOKEN" ]; then
  echo "$ENV_FILE has no AI_MEMORY_AUTH_TOKEN" >&2
  exit 2
fi
curl -fsS -m 5 "$SERVER/healthz" >/dev/null || {
  echo "$SERVER is not answering; start it: (cd deploy/memory && docker compose up -d)" >&2
  exit 2
}

written=0
for entry in "${PAGES[@]}"; do
  IFS='|' read -r src path kind tags <<<"$entry"
  if [ ! -f "$src" ]; then
    echo "skip (missing): ${src#$REPO/}" >&2
    continue
  fi
  tag_args=()
  IFS=',' read -ra tag_list <<<"$tags"
  for tag in "${tag_list[@]}"; do tag_args+=(--tag "$tag"); done

  docker run --rm -i --network host \
    -e AI_MEMORY_SERVER_URL="$SERVER" -e AI_MEMORY_AUTH_TOKEN="$TOKEN" \
    "$IMAGE" write-page --path "$path" --kind "$kind" "${tag_args[@]}" --body - <"$src" \
    >/dev/null
  printf 'wrote %s (%s)\n' "$path" "$(wc -c <"$src") bytes"
  written=$((written + 1))
done

echo
echo "$written page(s) written — verify with:"
echo "  curl -s -H \"Authorization: Bearer \$AI_MEMORY_AUTH_TOKEN\" $SERVER/web"
