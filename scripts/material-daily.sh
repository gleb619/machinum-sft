#!/usr/bin/env bash
# The daily research loop behind `msft material`, in one unattended command:
#
#   collect (run)  ->  distill (the at agent)  ->  memorize (ai-memory wiki)  ->  report
#
#   scripts/material-daily.sh                  # a real run
#   scripts/material-daily.sh --dry-run        # every step in plan mode: nothing written,
#                                              #   nothing posted, `at` never called
#   LIMIT=10 scripts/material-daily.sh         # more queries in this run (default 6)
#   MATERIAL_SKIP_DISTILL=1 scripts/material-daily.sh   # skip the agent step, collect only
#
# stdout carries exactly one line — the report — so a cron DM/mail is readable at a glance.
# Progress, warnings and the reason a step died go to stderr. Exit codes follow the CLI:
#   0 clean · 1 finished with item-level failures · 2 config/env error (bad YAML, no token,
#   no state directory). A locked run prints one line and exits 0: cron must not retry-storm.
#
# The whole design is in docs/material-pipeline.md; the contract is in the spec that shipped
# with `src/msft/material.py`.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

MATERIAL_DIR="${MSFT_MATERIAL_DIR:-$REPO/docs/material}"
STATE_DIR="$MATERIAL_DIR/.state"
LOCK="$STATE_DIR/lock"
LIMIT="${LIMIT:-6}"

# Per-step budgets, seconds. These are ceilings for an unattended run, not measurements: a
# stuck `webbridge.py` or a hung agent must not hold the lock (or the cron slot) until
# tomorrow. They sum to 3360 s so they land inside Hermes cron's 3600 s script timeout
# (`cron.script_timeout_seconds`); the outer kill is then a backstop, not the usual end.
T_COLLECT="${MATERIAL_TIMEOUT_COLLECT:-1800}"
T_DISTILL="${MATERIAL_TIMEOUT_DISTILL:-1200}"
T_MEMORIZE="${MATERIAL_TIMEOUT_MEMORIZE:-300}"
T_REPORT="${MATERIAL_TIMEOUT_REPORT:-60}"

DRY_RUN=0

note() { printf 'material: %s\n' "$*" >&2; }
die() { note "$*"; exit 2; }

usage() {
  cat <<'EOF'
usage: scripts/material-daily.sh [--dry-run] [--limit N]

  --dry-run    pass --dry-run to every step; nothing is written or posted
  --limit N    queries this run (default: $LIMIT, i.e. 6)

env: LIMIT, MATERIAL_SKIP_DISTILL=1, MATERIAL_TIMEOUT_{COLLECT,DISTILL,MEMORIZE,REPORT},
     MSFT_MATERIAL_DIR, MSFT_MATERIAL_CONFIG, MSFT_TUNNELSTACK, MSFT_MEMORY_URL,
     MSFT_MEMORY_TOKEN (the CLI reads these itself; this script only honours the path ones).
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --limit)
      [ "$#" -ge 2 ] || die "--limit needs a number"
      LIMIT="$2"
      shift
      ;;
    --limit=*) LIMIT="${1#*=}" ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

case "$LIMIT" in
  '' | *[!0-9]*) die "LIMIT must be a whole number, got '$LIMIT'" ;;
esac

# --- the CLI ------------------------------------------------------------------------------
# Prefer the repo venv's console script: it is the thing `scripts/guard.sh` tests. Fall back
# to a `python -m msft` that actually imports, then stop with instructions rather than
# letting cron quietly discover the problem a day later.
if [ -x "$REPO/.venv/bin/msft" ]; then
  MSFT=("$REPO/.venv/bin/msft")
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import msft' >/dev/null 2>&1; then
  note "no $REPO/.venv/bin/msft — falling back to python3 -m msft"
  MSFT=(python3 -m msft)
else
  die "no msft CLI: create the venv first — python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'"
fi

# --- one run at a time --------------------------------------------------------------------
# A cron run and a hand-started run share one SQLite file and one pack; overlapping them
# would double-fetch and interleave the `at` agent. `-n` (non-blocking) on purpose: waiting
# would pile up cron processes behind a slow agent.
if [ "$DRY_RUN" -eq 1 ]; then
  # A dry run writes no lock file of its own. If a real run is in flight, join its lock
  # read-only so a "plan" does not read half-updated state; if there is no lock, nothing is
  # running and there is nothing to wait for.
  if [ -f "$LOCK" ]; then
    exec 9<"$LOCK" || die "cannot open $LOCK"
    if ! flock -n 9; then
      printf 'material: another run holds the lock\n' >&2
      exit 0
    fi
  fi
else
  mkdir -p "$STATE_DIR" || die "cannot create $STATE_DIR"
  exec 9>"$LOCK" || die "cannot open $LOCK"
  if ! flock -n 9; then
    printf 'material: another run holds the lock\n' >&2
    exit 0
  fi
fi

# --- step runner --------------------------------------------------------------------------
FAILED=0
ENVERR=0
FAILURES=()

run_step() { # name budget argv...
  local name="$1" budget="$2"
  shift 2
  local rc=0
  note "→ $name: ${MSFT[*]} material $*"
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after=30 "$budget" "${MSFT[@]}" material "$@" >&2 || rc=$?
  else
    "${MSFT[@]}" material "$@" >&2 || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    FAILED=1
    [ "$rc" -eq 2 ] && ENVERR=1
    FAILURES+=("$(printf '%s(rc=%s)' "$name" "$rc")")
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      note "✗ $name hit its ${budget}s budget and was killed"
    else
      note "✗ $name failed (rc=$rc)"
    fi
  else
    note "✓ $name"
  fi
  return "$rc"
}

PASS=()
[ "$DRY_RUN" -eq 1 ] && PASS=(--dry-run)

# 1. collect — the only step that touches the network aside from the agent's own reads.
collected=0
if run_step collect "$T_COLLECT" run --limit "$LIMIT" "${PASS[@]}"; then
  collected=1
fi

# 2. distill — delegated to `at`, expensive, and pointless on a pack built from a broken
# collect. Never run it after a failed collect: the items stay `packed` for tomorrow.
if [ "$collected" -eq 0 ]; then
  note "→ distill: skipped (collect failed; packed items wait for the next run)"
elif [ "${MATERIAL_SKIP_DISTILL:-0}" = "1" ]; then
  note "→ distill: skipped (MATERIAL_SKIP_DISTILL=1)"
elif run_step distill "$T_DISTILL" distill "${PASS[@]}"; then
  :
else
  : # recorded by run_step; memorizing the *previous* days' notes is still safe
fi

# 3. memorize — cheap, idempotent, and independent of this run's collect: it walks every
# `distilled` item, so a failed distill still lets an earlier backlog reach the wiki.
run_step memorize "$T_MEMORIZE" memorize "${PASS[@]}" || true

# 4. report — read-only, so no --dry-run flag (the CLI's `report` does not take one).
REPORT=""
report_rc=0
if command -v timeout >/dev/null 2>&1; then
  REPORT=$(timeout --kill-after=10 "$T_REPORT" "${MSFT[@]}" material report) || report_rc=$?
else
  REPORT=$("${MSFT[@]}" material report) || report_rc=$?
fi
if [ "$report_rc" -ne 0 ]; then
  FAILED=1
  [ "$report_rc" -eq 2 ] && ENVERR=1
  FAILURES+=("report(rc=$report_rc)")
  note "✗ report failed (rc=$report_rc)"
fi

if [ -n "$REPORT" ]; then
  REPORT=$(printf '%s\n' "$REPORT" | awk 'NF { line = $0 } END { print line }')
fi
if [ -z "$REPORT" ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    REPORT="material: dry run finished, report unavailable (state written by a real run only)"
  else
    REPORT="material: report unavailable (rc=$report_rc) — run: .venv/bin/msft material report"
  fi
fi

# stdout: exactly one line.
printf '%s\n' "$REPORT"

if [ "$FAILED" -ne 0 ]; then
  note "failed steps: ${FAILURES[*]}"
  [ "$ENVERR" -eq 1 ] && exit 2
  exit 1
fi
exit 0
