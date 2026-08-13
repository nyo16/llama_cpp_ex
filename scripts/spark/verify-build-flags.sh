#!/usr/bin/env bash
# Prove that a built tree actually got the architecture flags it was asked for.
#
# Run it ON a node, inside the synced repo, after a build:
#
#   scripts/spark/remote.sh spark-1 scripts/spark/verify-build-flags.sh
#
# Three assertions, because each one catches a different silent failure:
#
#  1. ggml-cuda compiles for exactly ONE architecture. GGML_NATIVE=OFF is
#     required to reach the ARM CPU flag and it drops ggml-cuda out of `native`
#     into a seven-architecture fat binary. That is a ~6x build-time regression
#     with no runtime benefit and nothing warns about it.
#     Asserted from flags.make, NOT from CMakeCache.txt: CMAKE_CUDA_ARCHITECTURES
#     is an ordinary variable and never appears in the cache.
#
#  2. ggml-cpu compiles at the named architecture rather than -mcpu=native.
#
#  3. The emitted archive actually contains the quantized matmul instructions.
#     (2) can pass while (3) fails if the compiler accepts a flag and then
#     declines to vectorize, and (3) is the thing anybody actually cares about.
#     GCC 13.3 on GB10 fails (2) and (3) together, silently: it predates
#     Cortex-X925, rejects -mcpu=cortex-x925, and degrades native to base
#     ARMv8-A behind a soft CMake warning and exit 0.

set -uo pipefail

# shellcheck source=scripts/spark/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

build_dir=""
cpu_arch="$SPARK_CPU_ARM_ARCH"
cuda_arch="$SPARK_CUDA_ARCH"

while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) build_dir="${2:?--build-dir needs a path}"; shift 2 ;;
    --cpu-arch) cpu_arch="${2:?--cpu-arch needs a value}"; shift 2 ;;
    --cuda-arch) cuda_arch="${2:?--cuda-arch needs a value}"; shift 2 ;;
    -h | --help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) spark_die "unknown argument '$1' (try --help)" ;;
  esac
done

if [ -z "$build_dir" ]; then
  # Newest first, so a fresh build wins over a stale one from another flag set.
  build_dir=$(ls -dt _build/*/lib/llama_cpp_ex/obj/llama_build-* 2>/dev/null | head -1)
fi
[ -n "$build_dir" ] && [ -d "$build_dir" ] ||
  spark_die "no llama.cpp build tree found. Build first, or pass --build-dir."

spark_log "verifying $build_dir"

fails=0
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
skip() { printf '  SKIP  %s\n' "$*"; }

# --- 1. one CUDA architecture, and the right one -----------------------------

# 121a-real and 121a-virtual both compile for compute_121a; the suffix only
# selects which of cubin/PTX is embedded.
want_cc=${cuda_arch%-real}
want_cc=${want_cc%-virtual}

cuda_flags=$(find "$build_dir" -path '*ggml-cuda.dir/flags.make' -print -quit 2>/dev/null)

if [ -z "$cuda_flags" ]; then
  skip "ggml-cuda: not a CUDA build"
else
  archs=$(grep -oE 'arch=compute_[0-9a-z]+' "$cuda_flags" | sort -u | sed 's/arch=compute_//')
  n=$(printf '%s\n' "$archs" | grep -c .)
  if [ "$n" -ne 1 ]; then
    fail "ggml-cuda compiles $n architectures ($(printf '%s' "$archs" | tr '\n' ' ')) — fat binary, LLAMA_CUDA_ARCH is not reaching cmake"
  elif [ "$archs" != "$want_cc" ]; then
    fail "ggml-cuda compiles compute_$archs, expected compute_$want_cc"
  else
    pass "ggml-cuda: one architecture, compute_$want_cc"
  fi
fi

# --- 2. the CPU architecture flag --------------------------------------------

cpu_flags=$(find "$build_dir" -path '*ggml-cpu.dir/flags.make' -print -quit 2>/dev/null)

if [ -z "$cpu_flags" ]; then
  fail "ggml-cpu: no flags.make under $build_dir"
else
  if grep -q -- "-march=$cpu_arch" "$cpu_flags"; then
    pass "ggml-cpu: -march=$cpu_arch"
  else
    got=$(grep -oE -- '-m(arch|cpu)=[^ ]+' "$cpu_flags" | sort -u | tr '\n' ' ')
    fail "ggml-cpu: expected -march=$cpu_arch, got ${got:-nothing}"
  fi
fi

# --- 3. the instructions are actually in the archive -------------------------

archive=$(find "$build_dir" -name 'libggml-cpu.a' -print -quit 2>/dev/null)

if [ -z "$archive" ]; then
  fail "libggml-cpu.a not found under $build_dir"
elif ! command -v objdump >/dev/null 2>&1; then
  skip "objdump unavailable; cannot inspect $archive"
else
  disasm=$(objdump -d "$archive" 2>/dev/null)
  sdot=$(printf '%s' "$disasm" | grep -cE '\bsdot\b')
  smmla=$(printf '%s' "$disasm" | grep -cE '\bsmmla\b')
  sve=$(printf '%s' "$disasm" | grep -cE '\bz[0-9]+\.[bhsd]\b')

  if [ "$sdot" -gt 0 ] && [ "$smmla" -gt 0 ]; then
    pass "libggml-cpu.a: $sdot sdot, $smmla smmla, $sve SVE operands"
  else
    fail "libggml-cpu.a: $sdot sdot, $smmla smmla — the Q4/Q8 matmul kernels are
        compiled at base ARMv8-A. The -march flag did not survive to the compiler."
  fi
fi

if [ "$fails" -ne 0 ]; then
  spark_die "$fails check(s) failed"
fi
spark_log "all checks passed"
