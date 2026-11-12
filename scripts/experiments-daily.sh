#!/usr/bin/env bash
# The experiment queue behind `msft experiments`, in one unattended command:
#
#   sync (today's notes -> validated, costed jobs)  ->  tick (start the cheapest, watch it)
#
#   scripts/experiments-daily.sh               # a real run
#   scripts/experiments-daily.sh --dry-run     # plan only: no queue write, no ssh, no docker
#
# WHY it is a separate script from material-daily.sh: the research loop and the GPU box are
# different failures with different budgets. `material-daily` must finish inside an hour and
# owns the agent; this one waits for a training run and must not be killed halfway through
# its own watch. They share the tree, not the lock — one lock per queue, so a slow distill
# cannot block tonight's job and a slow job cannot block tomorrow's notes.
#
# The weekend rule lives in `msft.experiments.schedule_budget`, reached through
# `tick --budget auto`: Mon-Fri one job with a 90-minute watch, Sat-Sun the whole backlog
# with six hours. One implementation, so this script cannot drift from the tests.
#
# stdout carries exactly ONE line — the tick summary, which folds in the queue counters — so
# a cron DM is readable at a glance and a green day costs no model tokens. Progress, the
# catalog error, the ssh failure all go to stderr.
#
# Exit codes: 0 nothing due, work done, or the box is unreachable (the line says how many are
# queued); 1 a job actually failed; 2 config/environment (bad catalog, no CLI, lock held).
#
# The cron slot must be long enough for the watch: `cron.script_timeout_seconds` above
# 21600 + 600 for a weekend drain, or the outer kill lands mid-job and the queue sees a
# `running` row with no verdict. `tick --time-budget` is the inner ceiling.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

MATERIAL_DIR="${MSFT_MATERIAL_DIR:-$REPO/docs/material}"
STATE_DIR="$MATERIAL_DIR/.state"
LOCK="$STATE_DIR/experiments.lock"
CATALOG="${MSFT_EXPERIMENTS_CATALOG:-$REPO/configs/experiments/catalog.yaml}"

DRY_RUN=0
note() { printf 'experiments: %s\n' "$*" >&2; }
die() { note "$*"; exit 2; }

usage() {
  cat <<'EOF'
usage: scripts/experiments-daily.sh [--dry-run]

  --dry-run    sync and tick in plan mode; nothing is queued, signalled or run

env: MSFT_MATERIAL_DIR, MSFT_EXPERIMENTS_CATALOG (both read by the CLI as well),
     MSFT_EXPERIMENTS_BUDGET (default: auto — the weekday/weekend rule),
     MSFT_EXPERIMENTS_TIME_BUDGET (seconds; default: 5400 weekday / 21600 weekend).
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

[ -f "$CATALOG" ] || die "no catalog at $CATALOG (the allow-list is required)"

# --- the CLI ------------------------------------------------------------------------------
# Same rule as material-daily.sh: the repo venv's console script is the thing guard.sh tests;
# a `python -m msft` that imports is the fallback. Cron must not discover a broken install by
# failing silently a day later.
if [ -x "$REPO/.venv/bin/msft" ]; then
  MSFT=("$REPO/.venv/bin/msft")
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import msft' >/dev/null 2>&1; then
  note "no $REPO/.venv/bin/msft — falling back to python3 -m msft"
  MSFT=(python3 -m msft)
else
  die "no msft CLI: create the venv first — python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'"
fi

# --- one run at a time --------------------------------------------------------------------
# Its own lock, not material's: the two queues write different tables (material.db vs
# experiments.db) and a held agent must not stop the GPU job of the day. `-n` because a cron
# run that waits behind a six-hour weekend drain is worse than one that reports and leaves.
if [ "$DRY_RUN" -eq 1 ]; then
  if [ -f "$LOCK" ]; then
    exec 9<"$LOCK" || die "cannot open $LOCK"
    if ! flock -n 9; then
      printf 'experiments: another run holds the lock\n' >&2
      exit 0
    fi
  fi
else
  mkdir -p "$STATE_DIR" || die "cannot create $STATE_DIR"
  exec 9>"$LOCK" || die "cannot open $LOCK"
  if ! flock -n 9; then
    printf 'experiments: another run holds the lock\n' >&2
    exit 0
  fi
fi

# --- today's shape, for the progress line only --------------------------------------------
DOW=$(date +%u)
case "$DOW" in
  6 | 7) note "day $DOW: weekend — draining the backlog" ;;
  *) note "day $DOW: weekday — at most one job" ;;
esac

PASS=()
[ "$DRY_RUN" -eq 1 ] && PASS=(--dry-run)

# --- 1. note -> queue ---------------------------------------------------------------------
# A failed sync is a config problem (bad catalog, unreadable notes), not a red experiment:
# exit 2 so it is not confused with "the job failed". The already-queued rows stay runnable,
# so a sync failure on a weekend still lets the backlog drain — hence nothing is aborted.
sync_rc=0
"${MSFT[@]}" experiments sync "${PASS[@]}" >&2 || sync_rc=$?
if [ "$sync_rc" -ne 0 ]; then
  note "sync failed (rc=$sync_rc); the existing queue was left alone"
  exit 2
fi

# --- 2. queue -> box ----------------------------------------------------------------------
# The one line comes from tick, not from `report`: only tick knows whether the box was
# reachable, and "box unreachable: 3 queued" is exactly the fact a silent 13:00 cron would
# otherwise hide. `report` prints the same counters offline for a human.
BUDGET="${MSFT_EXPERIMENTS_BUDGET:-auto}"
TIME_ARGS=()
if [ -n "${MSFT_EXPERIMENTS_TIME_BUDGET:-}" ]; then
  TIME_ARGS=(--time-budget "$MSFT_EXPERIMENTS_TIME_BUDGET")
fi

LINE=""
tick_rc=0
LINE=$("${MSFT[@]}" experiments tick --budget "$BUDGET" "${TIME_ARGS[@]}" "${PASS[@]}") || tick_rc=$?

if [ -z "$LINE" ]; then
  report_rc=0
  LINE=$("${MSFT[@]}" experiments report) || report_rc=$?
  if [ -z "$LINE" ]; then
    LINE="experiments: no summary (tick rc=$tick_rc, report rc=$report_rc)"
  fi
fi

# stdout: exactly one line, whatever the CLI printed above it.
LINE=$(printf '%s\n' "$LINE" | awk 'NF { line = $0 } END { print line }')
printf '%s\n' "$LINE"

case "$tick_rc" in
  0) exit 0 ;;
  4) note "box unreachable — nothing started, the queue keeps its rows"; exit 0 ;;
  1) note "a job failed; see msft experiments list --status failed"; exit 1 ;;
  *)
    note "tick could not run (rc=$tick_rc)"
    exit 2
    ;;
esac
