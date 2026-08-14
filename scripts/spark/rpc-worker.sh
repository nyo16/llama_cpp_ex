#!/usr/bin/env bash
# Manage the ggml RPC worker on a Spark.
#
#   scripts/spark/rpc-worker.sh start spark-2
#   scripts/spark/rpc-worker.sh start spark-2 --upstream --debug
#   scripts/spark/rpc-worker.sh status spark-2
#   scripts/spark/rpc-worker.sh logs spark-2
#   scripts/spark/rpc-worker.sh rss spark-2
#   scripts/spark/rpc-worker.sh stop spark-2
#
# Supervision is `systemd --user` via systemd-run. `loginctl enable-linger`
# succeeds without a password on these boxes (verified), so a user manager
# survives logout and journald captures the worker's output — no nohup, no
# stray log files, and `systemctl --user restart` is the restart story.
#
# ## Restart the worker between runs
#
# Two independent reporters describe upstream's RPC worker growing RSS during
# inference and never releasing it; killing the client does not free it. So the
# `rss` subcommand exists, `start` launches a sampler alongside the worker, and
# the sampler stops the worker rather than letting the node OOM.
#
# ## Bind to the fabric address
#
# Upstream defaults to 127.0.0.1, which is wrong here twice: a remote client
# cannot reach it, and RDMA can never engage on it because the transport picks
# an HCA by matching a GID against the socket's *local* address. This script
# always binds a fabric address and has no option to bind loopback.
#
# Nothing about this port is authenticated. It accepts commands to allocate
# memory and execute compute graphs. The fabric is point-to-point; keep it there.

set -euo pipefail

# shellcheck source=scripts/spark/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

UNIT=llama-rpc-worker
RSS_UNIT=llama-rpc-rss

usage() { sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

[ $# -ge 2 ] || { usage >&2; exit 2; }

command=$1; shift
node=$1; shift
spark_is_node "$node" || spark_die "unknown node '$node' (known: ${SPARK_NODES[*]})"

port=50052
link=0
threads=$SPARK_BIG_CORE_COUNT
upstream=0
debug=0
cache=1
# Fraction of total memory at which the sampler stops the worker. The worker
# legitimately holds tens of GB of weights, so an absolute default would be
# either useless or a footgun; this catches the leak, not the workload.
rss_fraction=0.92
# Force the TCP path. There is no runtime switch for the transport: selection is
# silent auto-negotiation, and GGML_RDMA_DEV pointing at a device that does not
# exist is the only lever short of a -DGGML_RPC_RDMA=OFF build. Both ends have
# to agree, so pass --tcp here and GGML_RDMA_DEV to the client too.
tcp=0
# `logs` takes a bare line count.
lines=100

while [ $# -gt 0 ]; do
  case "$1" in
    --port) port="${2:?}"; shift 2 ;;
    --link) link="${2:?}"; shift 2 ;;
    --threads) threads="${2:?}"; shift 2 ;;
    --rss-fraction) rss_fraction="${2:?}"; shift 2 ;;
    --upstream) upstream=1; shift ;;
    --debug) debug=1; shift ;;
    --no-cache) cache=0; shift ;;
    --tcp) tcp=1; shift ;;
    -h | --help) usage; exit 0 ;;
    [0-9]*) lines=$1; shift ;;
    *) spark_die "unknown option '$1' (try --help)" ;;
  esac
done

ip=$(spark_fabric_ip "$node" "$link") || spark_die "no link $link on $node"
endpoint="$ip:$port"

remote() { ssh "$(spark_host "$node")" "bash -c $(spark_shquote "$1")"; }

start() {
  # Linger keeps the user manager alive after the ssh session ends. It is
  # idempotent and, on these boxes, needs no password.
  remote "loginctl enable-linger >/dev/null 2>&1 || true"

  if remote "systemctl --user is-active --quiet $UNIT"; then
    spark_die "$node already has a worker running. Stop it first — and do stop it
  between benchmark runs, because upstream's worker leaks."
  fi

  # The sampler outlives a worker that dies on its own (it waits for a MainPID),
  # so a previous failed start leaves it running and systemd-run then refuses
  # the name. Clear both before claiming them.
  remote "systemctl --user stop $RSS_UNIT >/dev/null 2>&1 || true; \
          systemctl --user reset-failed $UNIT $RSS_UNIT >/dev/null 2>&1 || true"

  local env_lines cache_dir inner
  cache_dir="\$HOME/$SPARK_RPC_CACHE_DIR"
  env_lines=$(spark_remote_env_preamble)

  # The `|| true` inside both substitutions below is load-bearing, not noise. An
  # assignment whose value contains a command substitution takes that
  # substitution's exit status, and `set -euo pipefail` above then kills the
  # shell -- so with `--no-cache` (cache=0) the `[ ... -eq 1 ]` test failed, the
  # assignment returned 1, and `start` exited silently *after* having already
  # stopped the running worker. Do not remove them.
  if [ "$upstream" -eq 1 ]; then
    # Upstream's standalone binary: the A/B reference. When a two-node run
    # misbehaves the first question is whether the fault is ours or upstream's,
    # and this is how that gets answered.
    inner="BIN=\$(ls -d _build/*/lib/llama_cpp_ex/obj/rpc_server_build/bin/ggml-rpc-server 2>/dev/null | head -1)
[ -n \"\$BIN\" ] || { echo 'no ggml-rpc-server; run: LLAMA_RPC=1 make rpc-server' >&2; exit 1; }
exec taskset -c $SPARK_BIG_CORES \"\$BIN\" -H $ip -p $port -t $threads$([ "$cache" -eq 1 ] && echo ' -c' || true)"
  else
    inner="export SPARK_RPC_ENDPOINT=$endpoint
export SPARK_RPC_THREADS=$threads
$([ "$cache" -eq 1 ] && echo "export SPARK_RPC_CACHE=$cache_dir" || true)
mkdir -p $cache_dir
exec taskset -c $SPARK_BIG_CORES mix run scripts/spark/rpc_worker.exs"
  fi

  local script
  script=$(
    printf 'set -euo pipefail\n'
    printf '%s\n' "$env_lines"
    # Not optional for a worker, and it has to be in the unit's own environment:
    # LLAMA_BACKEND is set, so mix force-builds, and without this the worker
    # would happily rebuild the *non*-RPC tree and then fail at start_link with
    # :rpc_unsupported. It is also part of the build-directory key, so the two
    # trees coexist and switching back is a cache hit.
    printf 'export LLAMA_RPC=1\n'
    [ "$debug" -eq 1 ] && printf 'export GGML_RPC_DEBUG=1\n'
    [ "$tcp" -eq 1 ] && printf 'export GGML_RDMA_DEV=no-such-hca\n'
    printf 'cd "$HOME/%s"\n' "$SPARK_REMOTE_DIR"
    printf '%s\n' "$inner"
  )

  # Build before the unit exists, not inside it. `mix run` compiles on demand,
  # and the readiness loop below cannot tell "still compiling" from "never going
  # to listen": it only knows the unit is active and the port is silent. After a
  # submodule bump that compile is a 2-3 minute llama.cpp rebuild, so the 120s
  # budget expired, this reported a failure, and the worker then came up fine on
  # its own -- an error message for a working worker. Building here also makes a
  # compile error arrive as a compile error instead of as a readiness timeout.
  # The upstream branch runs a prebuilt binary and has nothing to compile.
  if [ "$upstream" -eq 0 ]; then
    local prep
    prep=$(
      printf 'set -euo pipefail\n'
      printf '%s\n' "$env_lines"
      printf 'export LLAMA_RPC=1\n'
      printf 'cd "$HOME/%s"\n' "$SPARK_REMOTE_DIR"
      printf 'mix compile\n'
    )
    spark_log "$node: building the NIF first (minutes, after a llama.cpp bump)"
    remote "/bin/bash -c $(spark_shquote "$prep")" >/dev/null ||
      spark_die "$node: the worker's build failed; fix that before starting it"
  fi

  spark_log "$node: starting worker on $endpoint (${threads} threads on cores $SPARK_BIG_CORES$([ "$debug" -eq 1 ] && echo ", GGML_RPC_DEBUG=1")$([ "$tcp" -eq 1 ] && echo ", TCP forced"))"

  remote "systemd-run --user --unit=$UNIT --collect --description='llama.cpp RPC worker' \
    /bin/bash -c $(spark_shquote "$script") >/dev/null"

  start_rss_sampler

  # systemd-run returns as soon as the unit is queued, which says nothing about
  # whether anything is listening. Wait for the port.
  local waited=0
  while [ "$waited" -lt 120 ]; do
    if remote "exec 3<>/dev/tcp/$ip/$port" 2>/dev/null; then
      spark_log "$node: listening on $endpoint"
      status
      return 0
    fi
    if ! remote "systemctl --user is-active --quiet $UNIT"; then
      spark_warn "$node: the worker unit exited. Last output:"
      logs 40
      spark_die "worker failed to start"
    fi
    sleep 1
    waited=$((waited + 1))
  done

  logs 40
  spark_die "$node: nothing listening on $endpoint after 120s"
}

# Samples the worker's RSS into journald and stops it before the node OOMs.
# A watchdog that only logs is not a watchdog: an unbounded leak on a box with
# no swap takes the whole machine, and losing the worker is the cheaper failure.
start_rss_sampler() {
  local sampler
  sampler=$(cat <<SAMPLER
total_kb=\$(awk '/MemTotal/{print \$2}' /proc/meminfo)
limit_kb=\$(awk -v t="\$total_kb" 'BEGIN{printf "%d", t * $rss_fraction}')
echo "rss sampler: limit \$((limit_kb / 1024)) MiB (${rss_fraction} of total)"
while true; do
  pid=\$(systemctl --user show $UNIT --property=MainPID --value 2>/dev/null)
  [ -n "\$pid" ] && [ "\$pid" != 0 ] || { sleep 5; continue; }
  rss_kb=\$(awk '/^VmRSS:/{print \$2}' /proc/\$pid/status 2>/dev/null || echo 0)
  [ -n "\$rss_kb" ] && [ "\$rss_kb" -gt 0 ] || { sleep 5; continue; }
  echo "rss \$((rss_kb / 1024)) MiB"
  if [ "\$rss_kb" -gt "\$limit_kb" ]; then
    echo "rss \$((rss_kb / 1024)) MiB exceeds \$((limit_kb / 1024)) MiB - stopping the worker"
    systemctl --user stop $UNIT
    exit 1
  fi
  sleep 5
done
SAMPLER
  )

  remote "systemd-run --user --unit=$RSS_UNIT --collect --description='llama.cpp RPC worker RSS sampler' \
    /bin/bash -c $(spark_shquote "$sampler") >/dev/null"
}

stop() {
  remote "systemctl --user stop $RSS_UNIT $UNIT 2>/dev/null || true; \
          systemctl --user reset-failed $RSS_UNIT $UNIT 2>/dev/null || true"
  spark_log "$node: worker stopped (the port is released only because the VM exits;
  the native accept loop has no shutdown hook)"
}

status() {
  remote "systemctl --user is-active $UNIT 2>/dev/null || echo inactive" | sed "s/^/  unit    /"
  rss
}

rss() {
  remote "pid=\$(systemctl --user show $UNIT --property=MainPID --value 2>/dev/null); \
    if [ -n \"\$pid\" ] && [ \"\$pid\" != 0 ]; then \
      awk '/^VmRSS:/{printf \"  rss     %d MiB\\n\", \$2/1024}' /proc/\$pid/status; \
    else echo '  rss     n/a'; fi"
}

logs() {
  remote "journalctl --user -u $UNIT --no-pager -n ${1:-100}"
}

case "$command" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  rss) rss ;;
  logs) logs "$lines" ;;
  *) usage >&2; exit 2 ;;
esac
