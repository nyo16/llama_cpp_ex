#!/usr/bin/env bash
# Shared configuration and helpers for scripts/spark/*. Sourced, never executed.
#
# Every constant below is a measured fact about the two DGX Sparks rather than a
# guess, and it lives here so a hardware change is a one-line edit instead of a
# grep. The measurements behind them are in
# .claude/plans/dgx-spark-2node/research/ and docs/dgx-spark.md.

# --- Nodes -------------------------------------------------------------------

# ssh aliases, configured on the control node. The scripts never assume more
# than "ssh <alias> works and lands in the same $HOME on both".
SPARK_NODES=(spark-1 spark-2)

# Fabric addresses. Two independent point-to-point ConnectX-7 links, MTU 9000,
# RoCE v2. They are separate subnets, not a bond: ggml-rpc opens one queue pair
# per socket (ggml/src/ggml-rpc/transport.cpp:292) and picks the HCA by matching
# a GID against the socket's local address, so one peer connection can only ever
# use one link. Link 0 is the one everything uses; link 1 is listed so the fact
# sheet can prove both trained.
spark_fabric_ip() {
  case "$1:${2:-0}" in
    spark-1:0) printf '10.100.64.1' ;;
    spark-1:1) printf '10.100.65.1' ;;
    spark-2:0) printf '10.100.64.2' ;;
    spark-2:1) printf '10.100.65.2' ;;
    *) return 1 ;;
  esac
}

# The ssh target for a node. Normally the alias, which resolves through mDNS
# (`spark-f11e.local`).
#
# mDNS is the least reliable thing in this setup: it went away for good on a box
# that was otherwise perfectly healthy — 10-day uptime, 117 GiB free, answering
# pings from its neighbour — after a heavy run, and no amount of cache flushing
# brought it back. The override exists so a name-resolution problem is a
# one-variable fix instead of a dead afternoon:
#
#   export SPARK_HOST_SPARK_1=192.168.0.164
#
# The value is passed to ssh and rsync in place of the alias, so it can be an IP,
# another alias, or a user@host.
spark_host() {
  case "$1" in
    spark-1) printf '%s' "${SPARK_HOST_SPARK_1:-spark-1}" ;;
    spark-2) printf '%s' "${SPARK_HOST_SPARK_2:-spark-2}" ;;
    *) printf '%s' "$1" ;;
  esac
}

# The node that owns a working asdf install and acts as the toolchain donor.
SPARK_TOOLCHAIN_DONOR="${SPARK_TOOLCHAIN_DONOR:-spark-1}"

# --- CPU topology ------------------------------------------------------------

# The clusters are INTERLEAVED. X925 (performance) is 5-9,15-19 and A725
# (efficiency) is 0-4,10-14, so the obvious `taskset -c 0-9` pins a job entirely
# to little cores. Measured: 1-byte RDMA ping-pong p50 21.5 us on cpu19 versus
# 29.1 us on cpu0.
SPARK_BIG_CORES="5-9,15-19"
SPARK_LITTLE_CORES="0-4,10-14"
SPARK_BIG_CORE_COUNT=10

# --- Build flags -------------------------------------------------------------

# GB10 is sm_121a. GCC 13.3 predates Cortex-X925/A725 and rejects
# -mcpu=cortex-x925 outright, so ggml's -mcpu=native probe degrades to base
# ARMv8-A *silently* (soft CMake warning, exit 0) and libggml-cpu.a comes out
# with 0 sdot, 0 smmla and no SVE — i.e. no quantized matmul kernels at all.
# Naming the architecture is the only way to get them, it requires
# GGML_NATIVE=OFF, and GGML_NATIVE=OFF in turn drags CUDA from one arch into a
# 7-arch fat binary unless the CUDA arch is pinned too. The Makefile enforces
# that pairing with an $(error); these are the values it wants.
SPARK_CPU_ARM_ARCH="${SPARK_CPU_ARM_ARCH:-armv9.2-a+dotprod+i8mm+fp16+bf16+sve2}"
SPARK_CUDA_ARCH="${SPARK_CUDA_ARCH:-121a-real}"
SPARK_LLAMA_BACKEND="${SPARK_LLAMA_BACKEND:-cuda}"

# --- Remote layout -----------------------------------------------------------
# All relative to the remote $HOME so they survive tilde expansion in ssh.

SPARK_REMOTE_DIR="${SPARK_REMOTE_DIR:-src/llama_cpp_ex}"
SPARK_MODELS_DIR="${SPARK_MODELS_DIR:-models}"
SPARK_RPC_CACHE_DIR="${SPARK_RPC_CACHE_DIR:-.cache/llama.cpp/rpc}"

# --- What the working tree sync carries --------------------------------------
#
# Each exclusion has a reason, and the last one is the only non-obvious entry:
#
#   .git                       history, not sources
#   _build                     the control node's tree is aarch64-apple-darwin;
#                              sharing it poisons the remote build
#   deps                       fetched remotely — hex.pm is reachable from both
#   doc, .elixir_ls, .expert   editor and docs output
#   priv/llama_cpp_ex_nif.so   the mac's NIF; the remote builds its own
#   priv/plts                  dialyzer PLTs are OTP+arch specific
#   tmp, erl_crash.dump        scratch
#   vendor/llama.cpp/.git      36 MB of history the build does not need.
#                              Makefile:29 falls back to LLAMA_COMMIT when
#                              vendor/llama.cpp/.git is absent, so LLAMA_SHA
#                              still resolves — sync.sh verifies that on every
#                              run rather than trusting it.
#   models                     .gitignore treats ./models as the local GGUF
#                              directory and the nodes keep their own at
#                              ~/models. Without this, a developer with models
#                              checked out locally rsyncs tens or hundreds of GB
#                              to both Sparks on every sync — over the LAN, not
#                              the fabric — and --delete makes the interaction
#                              with the nodes' own ~/models worth not finding out
#                              about.
SPARK_SYNC_EXCLUDES=(
  .git
  _build
  deps
  doc
  models
  .elixir_ls
  .expert
  priv/llama_cpp_ex_nif.so
  priv/.llama_cpp_ex_nif.built
  priv/plts
  tmp
  erl_crash.dump
  vendor/llama.cpp/.git
  .sync-stamp
)

# A content digest over exactly the synced file set, computed identically on the
# control node and on each Spark. Both nodes must end byte-identical: the RPC
# HELLO handshake compares only major/minor and ignores patch
# (ggml/src/ggml-rpc/ggml-rpc.h:9-15), so two builds from drifted trees connect
# happily and then misinterpret each other's op codes. This check is
# load-bearing, not hygiene.
spark_tree_digest_cmd() {
  local prunes='' excl
  for excl in "${SPARK_SYNC_EXCLUDES[@]}"; do
    prunes+=" -path ./${excl} -o"
  done
  cat <<DIGEST
H=\$(command -v sha256sum >/dev/null 2>&1 && echo sha256sum || echo 'shasum -a 256')
find . \\( ${prunes% -o} \\) -prune -o -type f -print0 \\
  | LC_ALL=C sort -z \\
  | xargs -0 \$H \\
  | \$H \\
  | cut -d' ' -f1
DIGEST
}

# --- Toolchain ---------------------------------------------------------------

# asdf 0.19's shims are `exec asdf exec <tool>`, and the asdf binary is in
# /usr/bin on spark-1 and absent on spark-2 — no sudo, no apt, so it stays
# absent. Putting the real install bin directories on PATH drops the dependency
# on the CLI entirely and behaves identically on both nodes.
SPARK_ERLANG_VERSION="${SPARK_ERLANG_VERSION:-29.0.2}"
SPARK_ELIXIR_VERSION="${SPARK_ELIXIR_VERSION:-1.20.2-otp-29}"

SPARK_CUDA_HOME="${SPARK_CUDA_HOME:-/usr/local/cuda}"

# --- Models ------------------------------------------------------------------
# LLAMA_CACHE_DIR points Hub.download at ~/models, and its layout is
# <cache_dir>/<repo_id>/<revision>/<filename> (hub.ex:415-427) — the revision
# segment is part of the cache key, so "main" is in the path. remote.sh exports
# the two env vars the test suite and the benches read, but only for files that
# are actually present.
#
# scripts/spark/fetch_models.exs is the fetcher and the single source of truth
# for repo/filename pairs; these two are the ones the default workflow wants.
SPARK_SMOKE_GEN_MODEL="${SPARK_SMOKE_GEN_MODEL:-Qwen/Qwen3-8B-GGUF/main/Qwen3-8B-Q4_K_M.gguf}"
SPARK_BENCH_MODEL="${SPARK_BENCH_MODEL:-Qwen/Qwen3-8B-GGUF/main/Qwen3-8B-Q4_K_M.gguf}"

if [ -t 2 ]; then
  _spark_dim=$'\033[2m'; _spark_red=$'\033[31m'; _spark_yel=$'\033[33m'; _spark_rst=$'\033[0m'
else
  _spark_dim=''; _spark_red=''; _spark_yel=''; _spark_rst=''
fi

spark_log()  { printf '%s==>%s %s\n' "$_spark_dim" "$_spark_rst" "$*" >&2; }
spark_warn() { printf '%swarning:%s %s\n' "$_spark_yel" "$_spark_rst" "$*" >&2; }
spark_die()  { printf '%serror:%s %s\n' "$_spark_red" "$_spark_rst" "$*" >&2; exit 1; }

# --- Helpers -----------------------------------------------------------------

# Repository root, derived from this file's location rather than $PWD so the
# scripts work from anywhere.
spark_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

spark_shquote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

spark_is_node() {
  local candidate=$1 node
  for node in "${SPARK_NODES[@]}"; do
    [ "$node" = "$candidate" ] && return 0
  done
  return 1
}

# Resolve a node argument to a list of nodes. "all" or "both" means every node.
spark_resolve_nodes() {
  case "$1" in
    all | both) printf '%s\n' "${SPARK_NODES[@]}" ;;
    *)
      spark_is_node "$1" || spark_die "unknown node '$1' (known: ${SPARK_NODES[*]}, all)"
      printf '%s\n' "$1"
      ;;
  esac
}

# The environment preamble every remote command runs under. Emitted as shell
# source text so ssh, rsync-driven bootstrap and the worker scripts all agree.
#
# Deliberately a NON-login shell contract: this is what `ssh host cmd`, a
# systemd unit and any future CI runner get, and it is what the build spike
# validated. DGX OS puts nvcc on the login PATH only, via
# /etc/profile.d/nv_paths.sh, which is exactly the blind spot the Makefile's
# CUDA_HOME discovery exists to cover — so we set CUDA_HOME rather than relying
# on the profile script.
# The build flags are part of the contract too: LLAMA_BACKEND=cuda also flips
# mix.exs's make_force_build, which is what we want here — there is no
# aarch64-linux precompiled artifact, so the Spark always source-builds, and
# make itself stays incremental behind its stamp file.
#
# The two model exports are `|| true`-guarded because callers run under
# `set -e` and a missing model file is normal before Phase 3 provisioning.
spark_remote_env_preamble() {
  local models="\$HOME/${SPARK_MODELS_DIR}"
  cat <<PREAMBLE
export PATH="\$HOME/.asdf/installs/erlang/${SPARK_ERLANG_VERSION}/bin:\$HOME/.asdf/installs/elixir/${SPARK_ELIXIR_VERSION}/bin:${SPARK_CUDA_HOME}/bin:\$PATH"
export CUDA_HOME=${SPARK_CUDA_HOME}
export LLAMA_BACKEND=${SPARK_LLAMA_BACKEND}
export LLAMA_CPU_ARM_ARCH=${SPARK_CPU_ARM_ARCH}
export LLAMA_CUDA_ARCH=${SPARK_CUDA_ARCH}
export LLAMA_CACHE_DIR="${models}"
[ -f "${models}/${SPARK_SMOKE_GEN_MODEL}" ] && export LLAMA_SMOKE_GEN_MODEL="${models}/${SPARK_SMOKE_GEN_MODEL}" || true
[ -f "${models}/${SPARK_BENCH_MODEL}" ] && export LLAMA_MODEL_PATH="${models}/${SPARK_BENCH_MODEL}" || true
PREAMBLE
}
