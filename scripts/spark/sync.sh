#!/usr/bin/env bash
# Push the working tree to one or both DGX Sparks and prove they match.
#
#   scripts/spark/sync.sh            # both nodes
#   scripts/spark/sync.sh spark-1    # one node
#   scripts/spark/sync.sh --check    # compare only, sync nothing
#   scripts/spark/sync.sh --dry-run  # rsync -n
#
# Sync is rsync, not `git pull`: the remote needs the *working* tree including
# uncommitted changes, and rsync is one hop with no auth dance.
#
# After every sync both nodes are verified byte-identical to the control node
# over the synced file set. That check exists because the ggml RPC HELLO
# handshake compares only major/minor and ignores patch, so mismatched builds
# connect and then silently disagree about op codes.

set -euo pipefail

# shellcheck source=scripts/spark/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

check_only=0
dry_run=0
target=all

while [ $# -gt 0 ]; do
  case "$1" in
    --check) check_only=1; shift ;;
    --dry-run | -n) dry_run=1; shift ;;
    -h | --help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) spark_die "unknown option '$1' (try --help)" ;;
    *) target=$1; shift ;;
  esac
done

root=$(spark_repo_root)
cd "$root"

# Read into an array without `mapfile`: /bin/bash on macOS is still 3.2.
nodes=()
while IFS= read -r _n; do nodes+=("$_n"); done < <(spark_resolve_nodes "$target")

digest_cmd=$(spark_tree_digest_cmd)
local_digest=$(bash -c "$digest_cmd")
head_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)
dirty=$(git status --short 2>/dev/null || true)

if [ "$check_only" -eq 0 ]; then
  rsync_opts=(-az --delete --human-readable)
  [ "$dry_run" -eq 1 ] && rsync_opts+=(-n --itemize-changes)
  for excl in "${SPARK_SYNC_EXCLUDES[@]}"; do rsync_opts+=(--exclude "/$excl"); done

  for node in "${nodes[@]}"; do
    spark_log "sync -> $node:~/$SPARK_REMOTE_DIR"
    ssh "$(spark_host "$node")" "mkdir -p ~/$SPARK_REMOTE_DIR"
    rsync "${rsync_opts[@]}" ./ "$(spark_host "$node"):$SPARK_REMOTE_DIR/"
  done

  [ "$dry_run" -eq 1 ] && exit 0

  # The stamp is written after the tree lands so a half-finished sync leaves the
  # previous stamp in place rather than a lying new one.
  stamp=$(
    printf 'head=%s\n' "$head_sha"
    printf 'digest=%s\n' "$local_digest"
    printf 'synced=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'from=%s\n' "$(hostname)"
    printf 'dirty=%s\n' "$(printf '%s' "$dirty" | grep -c . || true)"
    if [ -n "$dirty" ]; then printf -- '--- git status --short ---\n%s\n' "$dirty"; fi
  )
  for node in "${nodes[@]}"; do
    printf '%s\n' "$stamp" | ssh "$(spark_host "$node")" "cat > ~/$SPARK_REMOTE_DIR/.sync-stamp"
  done
fi

# --- Verification ------------------------------------------------------------

fail=0

for node in "${nodes[@]}"; do
  remote_digest=$(ssh "$(spark_host "$node")" "cd ~/$SPARK_REMOTE_DIR 2>/dev/null && bash -c $(spark_shquote "$digest_cmd")" || echo MISSING)
  if [ "$remote_digest" = "$local_digest" ]; then
    spark_log "$node tree digest ${remote_digest:0:12} — matches control node"
  else
    spark_warn "$node tree digest $remote_digest != local $local_digest"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  spark_die "nodes are NOT byte-identical. Re-run without --check, or investigate before building:
  a drifted tree still passes the RPC HELLO handshake and then corrupts silently."
fi

# Makefile:29 resolves LLAMA_SHA from vendor/llama.cpp/.git when it is present
# and falls back to LLAMA_COMMIT when it is not. We exclude that .git, so this
# is the fallback path in production on every node — verified, not assumed.
local_sha=$(MIX_APP_PATH=/tmp/.spark-sha-probe make -n -p 2>/dev/null | sed -n 's/^LLAMA_SHA :*= *//p' | head -1)
for node in "${nodes[@]}"; do
  remote_sha=$(ssh "$(spark_host "$node")" "cd ~/$SPARK_REMOTE_DIR && MIX_APP_PATH=/tmp/.spark-sha-probe make -n -p 2>/dev/null | sed -n 's/^LLAMA_SHA :*= *//p' | head -1" || true)
  if [ -z "$remote_sha" ]; then
    spark_die "$node: LLAMA_SHA did not resolve. The vendor/llama.cpp/.git exclusion broke Makefile:29."
  elif [ "$remote_sha" != "$local_sha" ]; then
    spark_die "$node: LLAMA_SHA is $remote_sha, control node says $local_sha."
  fi
  spark_log "$node LLAMA_SHA ${remote_sha:0:12} — resolves without vendor/llama.cpp/.git"
done

spark_log "in sync at ${head_sha:0:12}${dirty:+ (+$(printf '%s' "$dirty" | grep -c .) local changes)}"
