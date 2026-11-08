#!/bin/bash
# Container entrypoint: make sure the bind-mounted repo is importable, then exec the command.
set -euo pipefail

REPO=${MSFT_REPO_DIR:-/workspace}

if [ ! -d "$REPO" ]; then
    echo "worker-entrypoint: $REPO is not mounted; nothing to run" >&2
    exit 1
fi

cd "$REPO"

# Install the package itself in editable mode the first time (and after dependency changes).
# `--no-deps` keeps it instant: torch/unsloth live in the image.
if ! python -c "import msft" 2>/dev/null; then
    echo "worker-entrypoint: installing msft from $REPO"
    pip install --no-deps -e . >/dev/null
fi

mkdir -p "$REPO/.worker" "${HF_HOME:-/hf}"

exec "$@"
