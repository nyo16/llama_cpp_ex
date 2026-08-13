#!/usr/bin/env bash
# B5 — does holding the cores out of deep idle help a TWO-NODE run?
#
#   scripts/spark/cpuidle-two-node.sh
#
# Phase 3 measured this single-node and found nothing: every condition landed
# within 2%, and the poller actively hurt TTFT. That is the expected answer for
# a GPU-bound decode loop on a CPU that never gets deeply idle anyway.
#
# Two nodes is a different question. Every token now involves a network wake on
# the far side, and LPI-3 exit latency on these boxes is 433 us — the effect
# that makes ICMP read 1.2 ms on a link whose real RTT is 1.39 us. If cpuidle
# costs anything anywhere, it costs it here.
#
# Pollers run on BOTH nodes, because the wake that matters is the worker's.
# `nice -19` so they yield to the real work; this is the sudo-free stand-in for
# `idle=poll`, which would need the kernel cmdline and a password.

set -euo pipefail

# shellcheck source=scripts/spark/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

here=$(dirname "${BASH_SOURCE[0]}")
client=${SPARK_CLIENT_NODE:-spark-1}
worker=${SPARK_WORKER_NODE:-spark-2}

poller_start() {
  local node=$1
  ssh "$(spark_host "$node")" 'bash -s' <<'POLLER' >/dev/null
mkdir -p ~/.cache/spark-poller
: > ~/.cache/spark-poller/pids
for cpu in $(seq 0 $(($(nproc) - 1))); do
  setsid nice -n 19 taskset -c "$cpu" bash -c 'while :; do :; done' >/dev/null 2>&1 &
  echo $! >> ~/.cache/spark-poller/pids
done
POLLER
  spark_log "$node: pollers up"
}

poller_stop() {
  local node=$1
  ssh "$(spark_host "$node")" 'while read -r pid; do kill "$pid" 2>/dev/null || true; done < ~/.cache/spark-poller/pids 2>/dev/null; \
               rm -f ~/.cache/spark-poller/pids' || true
  spark_log "$node: pollers down"
}

# Restart the worker between conditions: upstream's worker leaks, and a run that
# starts from a different RSS is not the same run.
measure() {
  local label=$1
  "$here/rpc-worker.sh" restart "$worker" >/dev/null 2>&1
  spark_log "measuring $label"

  "$here/remote.sh" --env MIX_ENV=bench --env LLAMA_RPC=1 --big-cores --forward-agent \
    "$client" mix run bench/spark_two_node.exs b3 2>&1 |
    grep '^| 120b' | sed "s/^| 120b two-node over RDMA/| $label/"
}

a=$(measure "a-default")

# The pollers are `setsid`'d so they survive the ssh session, and nothing reaps
# them but poller_stop. Under `set -euo pipefail` this script can die before
# reaching it in at least two ordinary ways -- rpc-worker.sh failing inside
# measure(), or the `grep` at the tail of that pipeline matching nothing -- and
# the result would be 20 spinning processes per node, on two machines,
# indefinitely, quietly contaminating every later measurement on those boxes.
# The trap is registered before the first poller starts and is idempotent.
trap 'poller_stop "$client"; poller_stop "$worker"' EXIT

poller_start "$client"
poller_start "$worker"
sleep 2
b=$(measure "b-pollers on both nodes")
poller_stop "$client"
poller_stop "$worker"
trap - EXIT

printf '\n| condition | load s | prompt | TTFT ms | prefill t/s | decode t/s | worker RSS MiB |\n'
printf -- '|---|---|---|---|---|---|---|\n'
printf '%s\n%s\n' "$a" "$b"
