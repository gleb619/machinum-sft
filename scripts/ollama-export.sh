#!/bin/bash
# Create (and optionally push) an Ollama model from a GGUF produced by `msft quantize`.
#
#   scripts/ollama-export.sh out/model.gguf machinum-book-ru
#   scripts/ollama-export.sh out/model.gguf machinum-book-ru --push
#
# Works with a host `ollama` binary or with an ollama container named OLLAMA_CONTAINER
# (default: ollama): in container mode the GGUF is copied in and `ollama create` runs inside.
# Only run this after training + quantisation actually succeeded.
set -euo pipefail

GGUF=${1:?usage: ollama-export.sh <model.gguf> <name> [--push]}
NAME=${2:?usage: ollama-export.sh <model.gguf> <name> [--push]}
PUSH=${3:-}
OLLAMA_CONTAINER=${OLLAMA_CONTAINER:-ollama}

[ -f "$GGUF" ] || { echo "no such gguf: $GGUF" >&2; exit 2; }
SIZE=$(stat -c%s "$GGUF")
if [ "$SIZE" -lt 500000000 ]; then
    echo "refusing: $GGUF is only $((SIZE/1024/1024)) MB — that is not a finished quant" >&2
    exit 2
fi

if command -v ollama >/dev/null 2>&1; then
    MODE=host
elif docker ps --format '{{.Names}}' | grep -qx "$OLLAMA_CONTAINER"; then
    MODE=container
else
    echo "no ollama: neither a host binary nor a container named $OLLAMA_CONTAINER" >&2
    exit 2
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Gemma 4 chat format: <|turn>role … <turn|>. The worker never thinks — the pipeline wants
# the reply only — so no <|think|> marker here.
cat >"$WORK/Modelfile" <<EOF
FROM /models/$(basename "$GGUF")

PARAMETER num_ctx 8192
PARAMETER temperature 0
PARAMETER top_p 0.95
PARAMETER top_k 64
PARAMETER repeat_penalty 1.05

TEMPLATE """{{ if .System }}<bos><|turn>system
{{ .System }}<turn|>
{{ end }}<|turn>user
{{ .Prompt }}<turn|>
<|turn>model
"""

PARAMETER stop "<turn|>"
PARAMETER stop "<|turn>"
EOF

echo "== mode: $MODE =="
cat "$WORK/Modelfile"

if [ "$MODE" = container ]; then
    docker exec "$OLLAMA_CONTAINER" mkdir -p /models
    docker cp "$GGUF" "$OLLAMA_CONTAINER":/models/"$(basename "$GGUF")"
    docker cp "$WORK/Modelfile" "$OLLAMA_CONTAINER":/tmp/Modelfile."$NAME"
    echo "== ollama create $NAME (in $OLLAMA_CONTAINER) =="
    docker exec "$OLLAMA_CONTAINER" ollama create "$NAME" -f /tmp/Modelfile."$NAME"
    if [ "$PUSH" = "--push" ]; then
        echo "== ollama push $NAME (needs an ollama.com login inside the container) =="
        docker exec "$OLLAMA_CONTAINER" ollama push "$NAME"
    fi
    echo "== done =="
    docker exec "$OLLAMA_CONTAINER" ollama list | grep -E "$NAME|NAME" || true
else
    sed -i "s|FROM /models/|FROM |" "$WORK/Modelfile"
    echo "== ollama create $NAME =="
    ollama create "$NAME" -f "$WORK/Modelfile"
    if [ "$PUSH" = "--push" ]; then
        echo "== ollama push $NAME (requires an ollama.com login) =="
        ollama push "$NAME"
    fi
    echo "== done =="
    ollama list | grep -E "$NAME|NAME" || true
fi
