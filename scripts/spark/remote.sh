#!/usr/bin/env bash
# Run a command on a DGX Spark with the right environment.
#
#   scripts/spark/remote.sh spark-1 mix test
#   scripts/spark/remote.sh --big-cores spark-1 mix run bench/single_generate.exs
#   scripts/spark/remote.sh --env MIX_ENV=bench spark-1 mix run bench/spark_baseline.exs
#   scripts/spark/remote.sh --print spark-2 mix compile     # show, do not run
#
# The default is a NON-login shell, because that is what `ssh host cmd`, a
# systemd unit and any CI runner get, and it is what the build spike validated.
# --login exists for the rare thing that genuinely needs
# /etc/profile.d/nv_paths.sh; if a *build* needs it, the bug is in the
# environment contract, not in the shell.

set -euo pipefail

# shellcheck source=scripts/spark/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'
Options:
  --login          run through `bash -lc` instead of a non-login shell
  --dir <path>     remote working directory, relative to $HOME
                   (default: the synced repo; "-" for $HOME)
  --env K=V        extra environment variable, repeatable
  --big-cores      wrap the command in `taskset -c 5-9,15-19` (Cortex-X925)
  --tty            allocate a remote tty
  --print          print the remote script instead of running it
USAGE
}

login=0
tty=0
print_only=0
big_cores=0
forward_agent=0
dir="$SPARK_REMOTE_DIR"
extra_env=()

while [ $# -gt 0 ]; do
  case "$1" in
    --login) login=1; shift ;;
    --tty) tty=1; shift ;;
    --print) print_only=1; shift ;;
    --big-cores) big_cores=1; shift ;;
    --forward-agent) forward_agent=1; shift ;;
    --dir) dir="${2:?--dir needs a path}"; shift 2 ;;
    --env) extra_env+=("${2:?--env needs K=V}"); shift 2 ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    -*) spark_die "unknown option '$1' (try --help)" ;;
    *) break ;;
  esac
done

[ $# -ge 2 ] || { usage >&2; exit 2; }

node=$1; shift
spark_is_node "$node" || spark_die "unknown node '$node' (known: ${SPARK_NODES[*]})"

prefix=()
[ "$big_cores" -eq 1 ] && prefix=(taskset -c "$SPARK_BIG_CORES")

# Assemble the remote script. Every word is quoted here rather than left to
# ssh's own re-splitting, which joins its arguments with spaces and would mangle
# any argument containing one.
build_script() {
  printf 'set -euo pipefail\n'
  spark_remote_env_preamble

  local kv
  for kv in ${extra_env[@]+"${extra_env[@]}"}; do
    case "$kv" in
      *=*) printf 'export %s=%s\n' "${kv%%=*}" "$(spark_shquote "${kv#*=}")" ;;
      *) spark_die "--env expects K=V, got '$kv'" ;;
    esac
  done

  # Forward MIX_ENV from the caller: `MIX_ENV=bench remote.sh ...` should mean
  # what it says.
  [ -n "${MIX_ENV:-}" ] && printf 'export MIX_ENV=%s\n' "$(spark_shquote "$MIX_ENV")"

  if [ "$dir" = "-" ]; then
    printf 'cd "$HOME"\n'
  else
    printf 'cd "$HOME/%s" 2>/dev/null || { echo "remote: ~/%s is missing; run scripts/spark/sync.sh first" >&2; exit 1; }\n' \
      "$dir" "$dir"
  fi

  local arg
  printf 'exec'
  for arg in ${prefix[@]+"${prefix[@]}"} "$@"; do printf ' %s' "$(spark_shquote "$arg")"; done
  printf '\n'
}

script=$(build_script "$@")

if [ "$print_only" -eq 1 ]; then
  printf '%s\n' "$script"
  exit 0
fi

ssh_opts=()
[ "$tty" -eq 1 ] && ssh_opts+=(-t)

# The nodes hold no keys for each other and the plan forbids adding any, so a
# command that needs to reach the *other* Spark (reading the RPC worker's RSS,
# say) rides a forwarded agent. ControlPath=none is required: the multiplexed
# master was opened without -A and reusing it would silently drop the
# forwarding.
if [ "$forward_agent" -eq 1 ]; then
  ssh_opts+=(-A -o ControlPath=none)
fi
[ "$login" -eq 1 ] && shell="bash -lc" || shell="bash -c"

exec ssh ${ssh_opts[@]+"${ssh_opts[@]}"} "$(spark_host "$node")" "$shell $(spark_shquote "$script")"
