#!/usr/bin/env bash
# Measure decode latency under four process-placement conditions.
#
#   scripts/spark/cpuidle-matrix.sh spark-1
#
# The conditions, and why each is here:
#
#   a-default       no pinning, no poller. The number you get by doing nothing.
#   b-poller        a `nice -19` busy loop pinned to every cpu, holding the
#                   cores out of deep idle. This is the sudo-free stand-in for
#                   `idle=poll` on the kernel cmdline. On these boxes it takes
#                   ICMP RTT from 1.2 ms to 0.028 ms — a 43x effect on the
#                   network path. Whether it buys anything for *decode* is the
#                   question; if it does, that is the argument for asking the
#                   user to set cpuidle limits with their password.
#   c-beam-busy     BEAM scheduler busy-wait: +sbwt very_long and friends. Keeps
#                   Erlang's own schedulers spinning instead of sleeping, which
#                   is the in-VM equivalent of the poller and costs no cores.
#   d-big-cores     bound to the Cortex-X925 cluster, 5-9,15-19. NOT 0-9: the
#                   clusters are interleaved and 0-9 is mostly little cores.
#   e-big+beam      d plus c, since they address different sleeps.
#
# Output is a markdown table, ready to paste into bench/results/.

set -euo pipefail

# shellcheck source=scripts/spark/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

node=${1:-spark-1}
spark_is_node "$node" || spark_die "unknown node '$node'"
shift || true

here=$(dirname "${BASH_SOURCE[0]}")

run() {
  local label=$1 prefix=$2 erl_flags=$3
  spark_log "$label"

  "$here/remote.sh" --env MIX_ENV=bench --env "SPARK_COND=$label" \
    ${erl_flags:+--env "ELIXIR_ERL_OPTIONS=$erl_flags"} \
    "$node" bash -c "$prefix mix run bench/spark_cpuidle.exs" 2>&1 |
    grep '^RESULT' || spark_warn "$label produced no result"
}

# The poller: one `nice -19` spinner per cpu. nice -19 means it yields to
# anything real, so it costs throughput ~nothing while still preventing the
# core from entering a deep C-state. Started, measured against, and reaped —
# `pkill -f` would match this ssh session's own command line, so the pattern is
# escaped and the pids are tracked instead.
poller_start() {
  ssh "$(spark_host "$node")" 'bash -s' <<'POLLER'
ncpu=$(nproc)
mkdir -p ~/.cache/spark-poller
: > ~/.cache/spark-poller/pids
for cpu in $(seq 0 $((ncpu - 1))); do
  setsid nice -n 19 taskset -c "$cpu" bash -c 'while :; do :; done' >/dev/null 2>&1 &
  echo $! >> ~/.cache/spark-poller/pids
done
echo "started $ncpu pollers"
POLLER
}

poller_stop() {
  ssh "$(spark_host "$node")" 'while read -r pid; do kill "$pid" 2>/dev/null || true; done < ~/.cache/spark-poller/pids; \
               rm -f ~/.cache/spark-poller/pids; echo "pollers stopped"'
}

# Registered before any poller starts, and in the parent shell rather than inside
# the subshell below, so it survives the subshell dying. `run` can fail under
# `set -euo pipefail` (its `grep` tolerates no match, but remote.sh itself can
# fail), and without this the `b-poller` branch would leave one spinning process
# per cpu behind — silently contaminating every later measurement on the box.
# poller_stop is idempotent, so running it on the happy path too is free.
trap 'poller_stop >&2' EXIT

results=$(
  run "a-default" "" ""

  poller_start >&2
  # Give the pollers a moment to actually be scheduled everywhere.
  sleep 2
  run "b-poller" "" ""
  poller_stop >&2

  run "c-beam-busy" "" "+sbwt very_long +sbwtdcpu very_long +sbwtdio very_long"
  run "d-big-cores" "taskset -c $SPARK_BIG_CORES" ""
  run "e-big+beam" "taskset -c $SPARK_BIG_CORES" "+sbwt very_long +sbwtdcpu very_long +sbwtdio very_long"
)

printf '\n| condition | TTFT median ms | TTFT worst ms | decode t/s | ms/token |\n'
printf -- '|---|---|---|---|---|\n'
printf '%s\n' "$results" | awk -F'\t' '{ printf "| %s | %s | %s | %s | %s |\n", $2, $3, $4, $5, $6 }'
