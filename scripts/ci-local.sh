#!/usr/bin/env bash
# scripts/ci-local.sh — the guard's checks for unattended runs (cron, systemd timer).
#
# Identical to scripts/guard.sh except that it never refuses a dirty tree: a scheduled
# job must report on whatever the checkout happens to hold. All flags pass through.
#
#   ./scripts/ci-local.sh --json                  # nightly, CI emulation best effort
#   ./scripts/ci-local.sh --json --require-ci    # fail when wrkflw is missing
#
# Exit codes and the JSON contract are the guard's: see docs/guard.md.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec "$here/guard.sh" --allow-dirty "$@"
