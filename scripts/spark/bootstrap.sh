#!/usr/bin/env bash
# Idempotent per-node provisioning for the DGX Sparks. No sudo, ever.
#
#   scripts/spark/bootstrap.sh              # both nodes
#   scripts/spark/bootstrap.sh spark-2      # one node
#   scripts/spark/bootstrap.sh --facts-only # just print the fact sheet
#
# Neither box has passwordless sudo, so nothing here installs a package, writes
# a sysctl, touches the kernel cmdline or adds a system unit. It also does not
# touch ~/.ssh on either node.
#
# The one substantive job is the toolchain. spark-2 has no asdf at all and we
# cannot `apt install` the OTP build dependencies, so `asdf install erlang`
# there is not a plan. The two boxes are identical — same distro, kernel, arch —
# and kerl-built OTP links only against libraries present on both, so the
# primary path is to copy spark-1's ~/.asdf across the fabric. The fallback is
# reported for the user to action with their password rather than silently
# degrading to Ubuntu's Elixir 1.14 (mix.exs:143 requires ~> 1.18).

set -euo pipefail

# shellcheck source=scripts/spark/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

facts_only=0
target=all

while [ $# -gt 0 ]; do
  case "$1" in
    --facts-only) facts_only=1; shift ;;
    -h | --help) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) spark_die "unknown option '$1' (try --help)" ;;
    *) target=$1; shift ;;
  esac
done

nodes=()
while IFS= read -r _n; do nodes+=("$_n"); done < <(spark_resolve_nodes "$target")

erlang_bin=".asdf/installs/erlang/${SPARK_ERLANG_VERSION}/bin"
elixir_bin=".asdf/installs/elixir/${SPARK_ELIXIR_VERSION}/bin"

# --- Toolchain ---------------------------------------------------------------

toolchain_present() {
  ssh "$(spark_host "$1")" "test -x ~/$erlang_bin/erl && test -x ~/$elixir_bin/elixir"
}

# Copy the donor's ~/.asdf over the 10.100.64 fabric. The nodes have no keys for
# each other and the plan forbids adding any, so authentication rides a
# forwarded agent. ControlPath=none is required: the multiplexed master was
# opened without -A, and -A on a session that reuses it does nothing.
copy_toolchain_over_fabric() {
  local donor=$1 target_node=$2 donor_ip
  donor_ip=$(spark_fabric_ip "$donor" 0)

  ssh -o ControlPath=none -A "$(spark_host "$target_node")" \
    "rsync -a --delete -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
       '$donor_ip:.asdf/' '.asdf/' && rsync -a -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
       '$donor_ip:.tool-versions' '.tool-versions'"
}

# Fallback when the agent is not forwardable: relay through the control node.
# 431 MB, so it is the slow path, not the wrong one.
copy_toolchain_via_control_node() {
  local donor=$1 target_node=$2 tmp
  tmp=$(mktemp -d)
  # shellcheck disable=SC2064  # $tmp must expand now, not at trap time
  trap "rm -rf '$tmp'" RETURN
  rsync -a "$(spark_host "$donor"):.asdf/" "$tmp/asdf/"
  rsync -a "$(spark_host "$donor"):.tool-versions" "$tmp/tool-versions"
  rsync -a --delete "$tmp/asdf/" "$(spark_host "$target_node"):.asdf/"
  rsync -a "$tmp/tool-versions" "$(spark_host "$target_node"):.tool-versions"
}

verify_toolchain() {
  local node=$1 out
  out=$(ssh "$(spark_host "$node")" "PATH=\$HOME/$erlang_bin:\$HOME/$elixir_bin:\$PATH; \
    elixir -v 2>&1 | tail -1; \
    erl -noshell -eval 'io:format(\"otp=~s nif=~s~n\",[erlang:system_info(otp_release),erlang:system_info(nif_version)]),halt().'" 2>&1) || return 1
  printf '%s\n' "$out"
  printf '%s' "$out" | grep -q "Elixir ${SPARK_ELIXIR_VERSION%%-*}" || return 1
  printf '%s' "$out" | grep -q 'otp=29 nif=2.18' || return 1
}

provision_toolchain() {
  local node=$1

  if toolchain_present "$node"; then
    spark_log "$node: toolchain already installed"
  elif [ "$node" = "$SPARK_TOOLCHAIN_DONOR" ]; then
    spark_die "$node is the toolchain donor but has no erlang ${SPARK_ERLANG_VERSION} / elixir ${SPARK_ELIXIR_VERSION}.
  Install them there first, or point SPARK_TOOLCHAIN_DONOR at a node that has them."
  else
    toolchain_present "$SPARK_TOOLCHAIN_DONOR" ||
      spark_die "donor $SPARK_TOOLCHAIN_DONOR has no usable toolchain to copy"

    spark_log "$node: copying ~/.asdf from $SPARK_TOOLCHAIN_DONOR over the fabric"
    if ! copy_toolchain_over_fabric "$SPARK_TOOLCHAIN_DONOR" "$node"; then
      spark_warn "$node: fabric copy failed (no forwardable agent?), relaying through the control node"
      copy_toolchain_via_control_node "$SPARK_TOOLCHAIN_DONOR" "$node"
    fi
  fi

  if ! verify_toolchain "$node"; then
    spark_warn "$node: the copied toolchain does not run. Fallback path:

    ssh $node
    asdf plugin add erlang && asdf install erlang ${SPARK_ERLANG_VERSION}

  That needs the OTP build dependencies, which need your password:

    sudo apt install build-essential autoconf m4 libncurses-dev \\
      libssl-dev libwxgtk3.2-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev

  Do NOT fall back to the distro's Elixir: Ubuntu 24.04 ships 1.14 / OTP 25 and
  mix.exs:143 requires ~> 1.18."
    return 1
  fi
}

# --- Fact sheet --------------------------------------------------------------
#
# Printed so a later session can diff it. nvidia-smi cannot report GPU memory on
# GB10 (memory.total = [N/A], ATS addressing mode) — `free -h` is the number,
# and it is simultaneously host RAM and GPU memory.
fact_sheet() {
  local node=$1
  ssh "$(spark_host "$node")" "bash -s" <<'FACTS'
set -u
printf 'hostname      %s\n' "$(hostname)"
printf 'kernel        %s\n' "$(uname -r)"
printf 'distro        %s\n' "$(. /etc/os-release && echo "$PRETTY_NAME")"
printf 'driver        %s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
printf 'gpu           %s\n' "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
printf 'cuda          %s\n' "$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
printf 'memory        %s\n' "$(free -h | awk '/^Mem:/{print $2" total, "$7" available"}')"
printf 'disk          %s\n' "$(df -h "$HOME" | awk 'NR==2{print $4" free of "$2}')"
printf 'cores         %s online\n' "$(nproc)"
printf 'gcc           %s\n' "$(gcc -dumpfullversion 2>/dev/null)"
printf 'cmake         %s\n' "$(cmake --version 2>/dev/null | head -1 | awk '{print $3}')"
for iface in $(ip -o -4 addr show | awk '$4 ~ /^10\.100\./ {print $2}'); do
  addr=$(ip -o -4 addr show dev "$iface" | awk '{print $4}')
  mtu=$(cat "/sys/class/net/$iface/mtu")
  state=$(cat "/sys/class/net/$iface/operstate")
  printf 'fabric        %-16s %-16s mtu %s %s\n' "$iface" "$addr" "$mtu" "$state"
done
for dev in /sys/class/infiniband/*; do
  [ -e "$dev" ] || continue
  printf 'hca           %-10s width %s rate %s\n' "$(basename "$dev")" \
    "$(cat "$dev/ports/1/rate" 2>/dev/null | tr -s ' ')" \
    "$(cat "$dev/ports/1/phys_state" 2>/dev/null)"
done
printf 'toolchain     %s\n' "$(ls -d "$HOME"/.asdf/installs/*/* 2>/dev/null | sed "s|$HOME/.asdf/installs/||" | tr '\n' ' ')"
FACTS
  # Cluster topology comes from the local table, not from a probe: the point of
  # printing it is that `taskset -c 0-9` is a trap, and that is a fact about the
  # chip rather than about this boot.
  printf 'cpu clusters  X925 (big) %s, A725 (little) %s\n' "$SPARK_BIG_CORES" "$SPARK_LITTLE_CORES"
}

# --- Main --------------------------------------------------------------------

status=0

for node in "${nodes[@]}"; do
  printf '\n===== %s =====\n' "$node"

  ssh -o BatchMode=yes -o ConnectTimeout=10 "$(spark_host "$node")" true ||
    spark_die "$node is not reachable over ssh"

  if [ "$facts_only" -eq 0 ]; then
    provision_toolchain "$node" || status=1

    spark_log "$node: creating ~/$SPARK_MODELS_DIR, ~/$SPARK_RPC_CACHE_DIR, ~/$SPARK_REMOTE_DIR"
    ssh "$(spark_host "$node")" "mkdir -p ~/$SPARK_MODELS_DIR ~/$SPARK_RPC_CACHE_DIR ~/$SPARK_REMOTE_DIR"
  fi

  fact_sheet "$node"
done

exit "$status"
