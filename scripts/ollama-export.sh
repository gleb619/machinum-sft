#!/bin/bash
# Create (and optionally push) an Ollama model from a GGUF produced by `msft quantize`.
#
#   scripts/ollama-export.sh out/model.gguf machinum-book-ru
#   scripts/ollama-export.sh out/model.gguf machinum-book-ru --push
#
# Only run this after training + quantisation actually succeeded: the script refuses to
# create a model from a missing or tiny file, and `--push` is opt-in.
set -euo pipefail

GGUF=${1:?usage: ollama-export.sh <model.gguf> <name> [--push]}
NAME=${2:?usage: ollama-export.sh <model.gguf> <name> [--push]}
PUSH=${3:-}

[ -f "$GGUF" ] || { echo "no such gguf: $GGUF" >&2; exit 2; }
SIZE=$(stat -c%s "$GGUF")
if [ "$SIZE" -lt 500000000 ]; then
    echo "refusing: $GGUF is only $((SIZE/1024/1024)) MB — that is not a finished quant" >&2
    exit 2
fi
command -v ollama >/dev/null || { echo "ollama is not on PATH" >&2; exit 2; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Gemma 4 chat format: <|turn>role … <turn|>, thinking via <|think|> in the system prompt.
# The worker model never thinks — the pipeline wants the reply only — so no <|think|> here.
cat >"$WORK/Modelfile" <<EOF
FROM $(realpath "$GGUF")

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

echo "== Modelfile =="
cat "$WORK/Modelfile"
echo "== ollama create $NAME =="
ollama create "$NAME" -f "$WORK/Modelfile"

if [ "$PUSH" = "--push" ]; then
    echo "== ollama push $NAME (requires an ollama.com login) =="
    ollama push "$NAME"
fi

echo "== done =="
ollama list | grep -E "$NAME|NAME" || true
