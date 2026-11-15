#!/usr/bin/env bash
# Local-only training configs: install them without ever committing them.
#
# `configs/*.local.yaml` is git-ignored, so a config with a machine name, a private path or a
# private dataset reference never reaches git. This script moves a YAML into that slot and pushes
# it to the worker's checkout, where the container mounts the repo at /workspace.
#
#   scripts/configs.sh put <file.yaml> [--name <stem>]   copy into configs/<stem>.local.yaml
#   scripts/configs.sh push [<stem> ...] [--dry-run]     scp to <repo>/configs on the box
#   scripts/configs.sh list                              what is installed and where it would go
#
# Host and repo dir come from the git-ignored `configs/worker.local.yaml` (see
# configs/worker.example.yaml) or `MSFT_REMOTE_HOST`. Nothing here prints config contents.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

py=".venv/bin/python"
[ -x "$py" ] || py="python3"

die() { printf 'configs.sh: %s\n' "$*" >&2; exit 1; }

# Read host + repo from the same place deploy.py does, so there is one source of truth.
spec() {
  "$py" - <<'PY'
from msft.deploy import load_spec

s = load_spec()
print(s.host)
print(s.repo)
PY
}

cmd=${1:-}
shift || true

case "$cmd" in
  put)
    src=${1:-}
    [ -n "$src" ] || die "put needs a YAML file"
    shift || true
    name=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --name) name=${2:-}; shift 2 || die "--name needs a value" ;;
        *) die "unknown argument: $1" ;;
      esac
    done
    [ -f "$src" ] || die "no such file: $src"
    [ -n "$name" ] || name=$(basename "$src" .yaml)
    case "$name" in
      */*) die "--name must be a stem, not a path" ;;
    esac
    dest="configs/${name}.local.yaml"
    git check-ignore -q "$dest" || die "$dest is not git-ignored; refusing to write a tracked path"
    "$py" - "$src" <<'PY'
import sys

try:
    import yaml
except ImportError:  # yaml ships in the venv; a bare python3 may lack it
    yaml = None

path = sys.argv[1]
raw = open(path, encoding="utf-8").read()
if yaml is not None:
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"configs.sh: {path} is not a non-empty YAML mapping")
PY
    install -m 0644 "$src" "$dest"
    printf 'installed %s (%s bytes, git-ignored)\n' "$dest" "$(wc -c < "$dest")"
    printf 'on the box it must exist as %s after `configs.sh push %s`\n' "configs/${name}.local.yaml" "$name"
    ;;

  push)
    dry=0
    names=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --dry-run) dry=1; shift ;;
        *) names+=("$1"); shift ;;
      esac
    done
    if [ ${#names[@]} -eq 0 ]; then
      while IFS= read -r f; do names+=("$(basename "$f" .local.yaml)"); done \
        < <(ls -1 configs/*.local.yaml 2>/dev/null || true)
    fi
    [ ${#names[@]} -gt 0 ] || die "no configs/*.local.yaml to push"

    { read -r host; read -r repo; } < <(spec)
    [ -n "${host:-}" ] || die "no remote host"
    printf 'target: %s:%s/configs\n' "$(printf '%s' "$host" | sed -E 's/^[^@]*@//')" "$repo"

    ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
    for name in "${names[@]}"; do
      file="configs/${name}.local.yaml"
      [ -f "$file" ] || die "missing $file"
      if [ "$dry" = 1 ]; then
        printf 'would copy %s (%s bytes)\n' "$file" "$(wc -c < "$file")"
        continue
      fi
      scp -q "${ssh_opts[@]}" "$file" "$host:$repo/configs/" || die "scp failed for $file"
      size=$(ssh "${ssh_opts[@]}" "$host" "wc -c < $repo/configs/${name}.local.yaml")
      printf 'pushed %s -> %s/configs/%s.local.yaml (%s bytes)\n' "$file" "$repo" "$name" "$size"
    done
    ;;

  list)
    { read -r host; read -r repo; } < <(spec) || true
    printf 'remote: %s:%s/configs\n' "$(printf '%s' "${host:-<unset>}" | sed -E 's/^[^@]*@//')" "${repo:-?}"
    shopt -s nullglob
    found=0
    for f in configs/*.local.yaml; do
      found=1
      printf '  %-44s %8s bytes  %s\n' "$f" "$(wc -c < "$f")" "$(git check-ignore -q "$f" && echo git-ignored || echo NOT-IGNORED)"
    done
    [ "$found" = 1 ] || printf '  (none)\n'
    ;;

  *)
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
