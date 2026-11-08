#!/usr/bin/env bash
# scripts/guard.sh — the pre-merge gate: run what CI runs, before you merge.
#
#   ./scripts/guard.sh                  # refuses a dirty tree, then ruff -> pytest -> CI
#   ./scripts/guard.sh --allow-dirty    # same, for work in progress
#   ./scripts/guard.sh --json           # one JSON object on stdout, banners on stderr
#   ./scripts/guard.sh --require-ci     # a missing wrkflw is a failure, not a skip
#
# Steps, in order, stopping at the first failure:
#   ruff check .
#   pytest -q
#   wrkflw validate .github/workflows/ci.yml
#   wrkflw run --runtime emulation .github/workflows/ci.yml
#
# Exit codes: 0 ok · 1 a step failed · 2 usage or missing tool ·
#             3 dirty working tree · 4 --require-ci without wrkflw.
# Never prints secrets, environment dumps or the git remote URL. See docs/guard.md.
set -euo pipefail

ALLOW_DIRTY=0
JSON=0
REQUIRE_CI=0

WRKFLW_URL="https://github.com/bahdotsh/wrkflw/releases/download/v0.8.0/wrkflw-x86_64-unknown-linux-musl.tar.gz"

usage() {
    cat <<'EOF'
usage: scripts/guard.sh [--allow-dirty] [--json] [--require-ci]

  --allow-dirty   run even when `git status` is not empty (the files are listed anyway)
  --json          print exactly one JSON object on stdout:
                  {"steps":[{"name","cmd","status","seconds"}],"ok":bool}
                  every human-readable line goes to stderr instead
  --require-ci    exit 4 when wrkflw is not installed, instead of skipping the CI steps

exit codes: 0 ok, 1 step failed, 2 usage/missing tool, 3 dirty tree, 4 CI required.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --allow-dirty) ALLOW_DIRTY=1 ;;
        --json) JSON=1 ;;
        --require-ci) REQUIRE_CI=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            printf 'guard: unknown option: %s\n\n' "$arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Colour only when a human is watching; --json keeps stdout machine-readable.
if [ "$JSON" -eq 0 ] && [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    RED=$'\033[31m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    BLUE=$'\033[34m'
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    RESET=$'\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' DIM='' RESET=''
fi

# In --json mode every human line moves to stderr so stdout holds one JSON object.
log() {
    if [ "$JSON" -eq 1 ]; then
        printf '%s\n' "$*" >&2
    else
        printf '%s\n' "$*"
    fi
}

if ! REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null); then
    printf 'guard: not inside a git repository\n' >&2
    exit 2
fi
cd -- "$REPO_ROOT"

PY=${PYTHON:-python3}
if [ "$JSON" -eq 1 ] && ! command -v "$PY" >/dev/null 2>&1; then
    printf 'guard: --json needs python3 on PATH (set PYTHON=/path/to/python)\n' >&2
    exit 2
fi

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/msft-guard.XXXXXX") || {
    printf 'guard: cannot create a temporary directory\n' >&2
    exit 2
}
CLEAN_WORK=1
trap 'if [ "$CLEAN_WORK" -eq 1 ]; then rm -rf -- "$WORK_DIR"; fi' EXIT

STEP_NAME=()
STEP_CMD=()
STEP_STATUS=()
STEP_SECS=()
# Env assignments for the next step, space separated (word splitting is intended and
# keeps this working on bash 3.2, where an empty array trips `set -u`).
STEP_ENV=()

now_ms() {
    local ns
    ns=$(date +%s%N 2>/dev/null || true)
    case "$ns" in
        '' | *[!0-9]*) printf '%s' $(($(date +%s) * 1000)) ;;
        *) printf '%s' $((ns / 1000000)) ;;
    esac
}

fmt_secs() { printf '%d.%03d' $(($1 / 1000)) $(($1 % 1000)); }

# Mask credentials in URLs and token/key query parameters before printing any log.
scrub() {
    sed -E \
        -e 's#(https?://)[^/[:space:]@]+:[^/[:space:]@]+@#\1***@#g' \
        -e 's#([?&](token|access_token|api_key|key|password|secret)=)[^&[:space:]]+#\1***#g'
}

record_step() { # name cmd status seconds_ms
    STEP_NAME+=("$1")
    STEP_CMD+=("$2")
    STEP_STATUS+=("$3")
    STEP_SECS+=("$(fmt_secs "$4")")
}

run_quiet() { # cmd logfile
    env "${STEP_ENV[@]}" bash -c "$1" >>"$2" 2>&1
}

run_stream() { # cmd logfile
    env "${STEP_ENV[@]}" bash -c "$1" 2>&1 | tee -a "$2"
}

run_step() { # name cmd -> exit status of cmd; appends to the step table
    local name=$1 cmd=$2 log_file start ms rc status
    log_file="$WORK_DIR/$name.log"
    printf '# %s\n$ %s\n\n' "$name" "$cmd" >"$log_file"
    log ""
    log "${BLUE}${BOLD}> ${name}${RESET} ${DIM}${cmd}${RESET}"
    start=$(now_ms)
    rc=0
    if [ "$JSON" -eq 1 ]; then
        run_quiet "$cmd" "$log_file" || rc=$?
    else
        run_stream "$cmd" "$log_file" || rc=$?
    fi
    ms=$(($(now_ms) - start))
    if [ "$rc" -eq 0 ]; then
        status=ok
        log "  ${GREEN}ok${RESET} ${name} ${DIM}($(fmt_secs "$ms")s)${RESET}"
    else
        status=failed
        log "  ${RED}FAILED${RESET} ${name} ${DIM}($(fmt_secs "$ms")s, exit $rc)${RESET}"
    fi
    record_step "$name" "$cmd" "$status" "$ms"
    return "$rc"
}

show_failure() { # name rc
    local name=$1 rc=$2 log_file="$WORK_DIR/$1.log" line
    log ""
    log "${RED}${BOLD}GUARD FAILED: ${name} (exit ${rc})${RESET}"
    # Banner mode already streamed the output; only --json needs the excerpt reprinted.
    if [ "$JSON" -eq 1 ] && [ -s "$log_file" ]; then
        log "${DIM}--- last 40 lines of ${name} ---${RESET}"
        while IFS= read -r line; do
            log "$line"
        done < <(tail -n 40 "$log_file" | scrub)
        log "${DIM}--- end ---${RESET}"
    fi
    log "${RED}full log: ${log_file}${RESET}"
    CLEAN_WORK=0
}

emit_json() { # ok(true|false)
    [ "$JSON" -eq 1 ] || return 0
    local i tsv="$WORK_DIR/steps.tsv"
    : >"$tsv"
    for i in "${!STEP_NAME[@]}"; do
        printf '%s\t%s\t%s\t%s\n' \
            "${STEP_NAME[$i]}" "${STEP_CMD[$i]}" "${STEP_STATUS[$i]}" "${STEP_SECS[$i]}" >>"$tsv"
    done
    "$PY" - "$tsv" "$1" <<'PYEOF'
import json
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        name, cmd, status, seconds = line.rstrip("\n").split("\t", 3)
        rows.append({"name": name, "cmd": cmd, "status": status, "seconds": float(seconds)})
json.dump({"steps": rows, "ok": sys.argv[2] == "true"}, sys.stdout)
sys.stdout.write("\n")
PYEOF
}

finish_ok() {
    emit_json true
    log ""
    log "${GREEN}${BOLD}GUARD OK${RESET}"
    exit 0
}

finish_fail() { # exit code
    emit_json false
    exit "$1"
}

pick_tool() { # name -> repo venv first, then PATH
    local name=$1
    if [ -x "$REPO_ROOT/.venv/bin/$name" ]; then
        printf '%s' ".venv/bin/$name"
        return 0
    fi
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    return 1
}

DIRTY=$(git status --porcelain=v1 --untracked-files=all || true)

log ""
log "${BOLD}machinum-sft guard${RESET} ${DIM}$(git rev-parse --short HEAD 2>/dev/null || echo no-commit) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)${RESET}"

if [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" -eq 0 ]; then
    record_step "dirty-tree" "git status --porcelain" failed 0
    log ""
    log "${RED}${BOLD}GUARD REFUSED: the working tree is dirty${RESET}"
    log "${RED}commit or stash these files first, or re-run with --allow-dirty:${RESET}"
    while IFS= read -r line; do
        log "    $line"
    done < <(printf '%s\n' "$DIRTY" | scrub)
    finish_fail 3
fi

if [ -n "$DIRTY" ]; then
    log ""
    log "${YELLOW}note: dirty tree allowed (--allow-dirty), $(printf '%s\n' "$DIRTY" | grep -c .) file(s) uncommitted${RESET}"
fi

if ! RUFF=$(pick_tool ruff); then
    record_step "ruff" "ruff check ." failed 0
    log ""
    log "${RED}${BOLD}GUARD FAILED: ruff is not installed${RESET}"
    log "${RED}run: . .venv/bin/activate && pip install -e '.[dev]'${RESET}"
    finish_fail 2
fi

if ! PYTEST=$(pick_tool pytest); then
    record_step "pytest" "pytest -q" failed 0
    log ""
    log "${RED}${BOLD}GUARD FAILED: pytest is not installed${RESET}"
    log "${RED}run: . .venv/bin/activate && pip install -e '.[dev]'${RESET}"
    finish_fail 2
fi

rc=0
run_step "ruff" "$RUFF check ." || rc=$?
if [ "$rc" -ne 0 ]; then
    show_failure "ruff" "$rc"
    finish_fail 1
fi

rc=0
run_step "pytest" "$PYTEST -q" || rc=$?
if [ "$rc" -ne 0 ]; then
    show_failure "pytest" "$rc"
    finish_fail 1
fi

if WRKFLW=$(pick_tool wrkflw); then
    rc=0
    run_step "ci-validate" "$WRKFLW validate .github/workflows/ci.yml" || rc=$?
    if [ "$rc" -ne 0 ]; then
        show_failure "ci-validate" "$rc"
        finish_fail 1
    fi
    # Emulation runs the workflow on this host. Two things must be pinned or the result
    # depends on the caller's shell: PATH, so the workflow's `python`/`pip` are the project
    # venv rather than whatever the login shell has; and PIP_TARGET, so its
    # `pip install -e .` cannot overwrite the venv's editable install.
    STEP_ENV=(
        "PATH=$REPO_ROOT/.venv/bin:$PATH"
        "PIP_TARGET=$WORK_DIR/pip-target"
        "PIP_DISABLE_PIP_VERSION_CHECK=1"
    )
    rc=0
    run_step "ci-run" "$WRKFLW run --runtime emulation .github/workflows/ci.yml" || rc=$?
    STEP_ENV=()
    if [ "$rc" -ne 0 ]; then
        show_failure "ci-run" "$rc"
        finish_fail 1
    fi
else
    record_step "ci-validate" "wrkflw validate .github/workflows/ci.yml" skipped 0
    record_step "ci-run" "wrkflw run --runtime emulation .github/workflows/ci.yml" skipped 0
    log ""
    log "${YELLOW}SKIP ci: wrkflw is not on PATH, the GitHub workflow was NOT run locally${RESET}"
    log "${YELLOW}install it (static musl build, no sudo):${RESET}"
    log "${YELLOW}  mkdir -p ~/.local/bin && curl -fsSL ${WRKFLW_URL} | tar -xz -C ~/.local/bin${RESET}"
    if [ "$REQUIRE_CI" -eq 1 ]; then
        log ""
        log "${RED}${BOLD}GUARD FAILED: --require-ci was given but wrkflw is missing${RESET}"
        finish_fail 4
    fi
fi

finish_ok
