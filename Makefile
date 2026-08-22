# Makefile for llama_cpp_ex NIF
# Called by elixir_make during `mix compile`

# elixir_make always sets MIX_APP_PATH. A human running `make` directly does not,
# and an empty value is silently destructive rather than an error: PREFIX becomes
# /priv and BUILD becomes /obj, so `make clean` would try to rm -rf /obj and
# `make rpc-server` cmake-configures into it (observed: "Unable to (re)create the
# private pkgRedirects directory: /obj/rpc_server_build/CMakeFiles/pkgRedirects").
# Falling back keeps every path inside the project. The directory name is not one
# of mix's environments on purpose -- it must not collide with a real build tree,
# and scripts/spark/rpc-worker.sh globs _build/*/lib/llama_cpp_ex so it is still
# found. Reading a variable (`make print-VAR`) never needed this.
# `?=` would miss the defined-but-empty case, which is the dangerous one.
ifeq ($(strip $(MIX_APP_PATH)),)
MIX_APP_PATH := _build/standalone/lib/llama_cpp_ex
endif

PREFIX = $(MIX_APP_PATH)/priv
BUILD  = $(MIX_APP_PATH)/obj
NIF_SO = $(PREFIX)/llama_cpp_ex_nif.so

# --- llama.cpp source tree ---------------------------------------------------
# A git checkout has this as the `vendor/llama.cpp` submodule. A Hex tarball does
# not: shipping the tree would add hundreds of MB to the package, so `files:` in
# mix.exs ships `.gitmodules` instead and the rule near the bottom of this file
# clones the tree on demand, pinned to LLAMA_COMMIT.
LLAMA_DIR = $(shell pwd)/vendor/llama.cpp

# Upstream URL, read from .gitmodules so there is a single source of truth. The
# literal is only a fallback for the case where .gitmodules is missing.
LLAMA_REPO := $(shell git config --file .gitmodules --get submodule.vendor/llama.cpp.url 2>/dev/null)
ifeq ($(strip $(LLAMA_REPO)),)
  LLAMA_REPO := https://github.com/ggml-org/llama.cpp
endif

# Pinned llama.cpp commit, used when vendor/llama.cpp has to be cloned. MUST
# match the vendor/llama.cpp submodule; bump both together, see
# docs/release-guide.md. Override to build the NIF against another revision.
LLAMA_COMMIT ?= e85caa81ea2b65797396018c179b87ad61fa38ab

# The commit actually on disk. A submodule can be bumped without LLAMA_COMMIT
# following it, and the build has to key off what is really there.
LLAMA_SHA := $(shell test -e $(LLAMA_DIR)/.git && git -C $(LLAMA_DIR) rev-parse HEAD 2>/dev/null || echo $(LLAMA_COMMIT))
LLAMA_SHA_SHORT := $(shell echo $(LLAMA_SHA) | cut -c1-12)

# Compiler
CXX      ?= c++
# -DNDEBUG matches the llama.cpp libraries this object links against: they are
# configured with CMAKE_BUILD_TYPE=Release, whose CMAKE_CXX_FLAGS_RELEASE is
# "-O3 -DNDEBUG". Without it the llama.cpp/ggml/common headers inlined into this
# translation unit keep their debug assertions live, and an assert() firing
# inside a NIF takes down the whole VM. Build consistency is the entire reason.
#
# It does NOT disarm GGML_ASSERT. That expands to an unconditional ggml_abort
# (vendor/llama.cpp/ggml/include/ggml.h) and was observed aborting the VM with
# NDEBUG active; only explicit validation at the NIF boundary prevents those.
CXXFLAGS  = -std=c++17 -O2 -DNDEBUG -fPIC -fvisibility=hidden -Wall -Wno-unused-parameter -Wno-unused-function
CXXFLAGS += -I$(ERTS_INCLUDE_DIR)
CXXFLAGS += -I$(FINE_INCLUDE_DIR)
CXXFLAGS += -I$(LLAMA_DIR)/include
CXXFLAGS += -I$(LLAMA_DIR)/ggml/include
CXXFLAGS += -I$(LLAMA_DIR)/common
CXXFLAGS += -I$(LLAMA_DIR)/vendor

# Linker
LDFLAGS = -shared

# Platform detection
UNAME_S := $(shell uname -s)

# --- CUDA toolkit discovery --------------------------------------------------
# Resolved once, up here, because two separate decisions depend on it: `auto`
# uses it to decide whether CUDA is available at all, and the Linux link line
# below needs the library directory.
#
# Probing only `which nvcc` was wrong in both places. nvcc is routinely absent
# from a *non-login* PATH -- DGX OS installs it via /etc/profile.d/nv_paths.sh,
# which `ssh host make`, systemd units and most CI shells never source -- so on
# a machine with a complete toolkit `auto` silently produced a CPU-only build,
# and an explicit LLAMA_BACKEND=cuda produced no -L flags and failed the link on
# libcudart. Honour CUDA_HOME and CUDA_PATH first (both are conventional and
# either may be exported by a module system), then nvcc on PATH, then the
# standard install locations.
ifeq ($(strip $(CUDA_HOME)),)
  CUDA_HOME := $(strip $(CUDA_PATH))
endif
ifeq ($(strip $(CUDA_HOME)),)
  CUDA_HOME := $(patsubst %/bin/nvcc,%,$(shell command -v nvcc 2>/dev/null))
endif
ifeq ($(strip $(CUDA_HOME)),)
  CUDA_HOME := $(firstword $(wildcard /usr/local/cuda /opt/cuda) \
                 $(shell ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1))
endif

# Presence of the compiler, not of the directory: /usr/local/cuda survives a
# partial uninstall, and a runtime-only install has libraries but cannot build.
NVCC := $(wildcard $(CUDA_HOME)/bin/nvcc)

# x86_64 and sbsa toolkits both expose lib64 (a symlink to targets/<triple>/lib
# on sbsa); Debian's packaged nvidia-cuda-toolkit only has lib.
CUDA_LIBDIR := $(firstword $(wildcard $(CUDA_HOME)/lib64 $(CUDA_HOME)/lib))

# ggml's own GGML_CUDA_NCCL defaults to ON and quietly links libnccl through
# cmake whenever NCCL happens to be installed on the build host. cmake's
# target_link_libraries is invisible to this file -- the link line below is
# assembled by hand from the static archives ggml leaves behind -- so on a host
# with NCCL (every DGX, most multi-GPU boxes) ggml-cuda.a came out carrying
# undefined nccl* symbols and the resulting NIF failed to load with
# `undefined symbol: ncclAllReduce`.
#
# So whether NCCL is used is declared here rather than discovered, and it is off
# by default. It only accelerates collectives across multiple GPUs, while
# linking it makes libnccl.so.2 a hard load-time requirement of the artifact --
# the same trade as -march=native, and the same answer.
LLAMA_CUDA_NCCL ?= 0

# The same discover-vs-declare hazard, one layer down and with a second flag
# chained to it.
#
# ggml probes the host CPU with -mcpu=native (ggml/src/ggml-cpu/CMakeLists.txt)
# whenever GGML_NATIVE is ON, which is the default. On GB10 with GCC 13.3 that
# probe fails *silently* -- the compiler predates Cortex-X925/A725 and rejects
# -mcpu=cortex-x925, cmake emits a soft warning and exits 0, and the CPU backend
# is quietly built at base ARMv8-A. Measured on the emitted libggml-cpu.a:
# 0 sdot, 0 smmla, no SVE, versus 1134 sdot + 370 smmla + SVE once the
# architecture is named. Those are the Q4/Q8 quantized matmul kernels, so this
# is not a rounding error.
#
# Naming it requires GGML_NATIVE=OFF, and that is the trap: with GGML_NATIVE
# off, ggml-cuda stops using `native` and falls back to a seven-architecture fat
# binary (ggml/src/ggml-cuda/CMakeLists.txt:27-28), turning a 1m50s build into a
# ~6x one for no benefit on a machine with exactly one known GPU. So the two
# move together or not at all, and setting one without the other is an error
# rather than a surprise six minutes later.
#
# DGX Spark (GB10):
#   LLAMA_CPU_ARM_ARCH=armv9.2-a+dotprod+i8mm+fp16+bf16+sve2
#   LLAMA_CUDA_ARCH=121a-real
LLAMA_CPU_ARM_ARCH ?=
LLAMA_CUDA_ARCH ?=

# The ggml RPC backend, which lets a model's layers live on another host. Off by
# default: it is a networked attack surface and a protocol version coupling, and
# neither belongs in a build nobody asked for.
#
# GGML_RPC_RDMA gets the same declare-don't-discover treatment as NCCL, for
# exactly the same reason. ggml/src/ggml-rpc/CMakeLists.txt:11-22 turns it ON
# whenever libibverbs happens to be installed on the build host, so the same
# source produces a different artifact on a DGX than on a laptop, silently. And
# because that file links ibverbs with target_link_libraries -- invisible to the
# hand-assembled link line below -- libggml-rpc.a comes out carrying undefined
# ibv_* symbols and the NIF fails to load. Same failure mode as
# `undefined symbol: ncclAllReduce`, same fix: declare it here and pair the link
# flag.
LLAMA_RPC ?= 0
LLAMA_RPC_RDMA ?= 1

# Backend selection (auto, metal, cuda, vulkan, cpu)
LLAMA_BACKEND ?= auto

# CMake flags
CMAKE_FLAGS  = -DCMAKE_BUILD_TYPE=Release
CMAKE_FLAGS += -DBUILD_SHARED_LIBS=OFF
CMAKE_FLAGS += -DLLAMA_BUILD_EXAMPLES=OFF
CMAKE_FLAGS += -DLLAMA_BUILD_TESTS=OFF
CMAKE_FLAGS += -DLLAMA_BUILD_SERVER=OFF
CMAKE_FLAGS += -DLLAMA_BUILD_TOOLS=OFF
CMAKE_FLAGS += -DLLAMA_BUILD_APP=OFF
CMAKE_FLAGS += -DLLAMA_OPENSSL=OFF
CMAKE_FLAGS += -DCMAKE_POSITION_INDEPENDENT_CODE=ON

# Backend configuration
ifeq ($(LLAMA_BACKEND),auto)
  ifeq ($(UNAME_S),Darwin)
    CMAKE_FLAGS += -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
  else
    ifneq ($(NVCC),)
      CMAKE_FLAGS += -DGGML_CUDA=ON
    endif
  endif
else ifeq ($(LLAMA_BACKEND),metal)
  CMAKE_FLAGS += -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
else ifeq ($(LLAMA_BACKEND),cuda)
  CMAKE_FLAGS += -DGGML_CUDA=ON
else ifeq ($(LLAMA_BACKEND),vulkan)
  CMAKE_FLAGS += -DGGML_VULKAN=ON
else ifeq ($(LLAMA_BACKEND),cpu)
  CMAKE_FLAGS += -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF
endif

# cmake locates the toolkit by searching PATH for nvcc, so it has the same blind
# spot the discovery block above exists to cover: without this, a build that
# correctly selected CUDA still fails in `find_package(CUDAToolkit)`.
ifneq (,$(filter -DGGML_CUDA=ON,$(CMAKE_FLAGS)))
  ifneq ($(NVCC),)
    CMAKE_FLAGS += -DCMAKE_CUDA_COMPILER=$(NVCC)
  endif
  # Always stated, never left to ggml's default, so the archives cmake produces
  # and the link line assembled below cannot disagree about NCCL.
  ifneq ($(filter 1 true yes,$(LLAMA_CUDA_NCCL)),)
    CMAKE_FLAGS += -DGGML_CUDA_NCCL=ON
  else
    CMAKE_FLAGS += -DGGML_CUDA_NCCL=OFF
  endif
endif

# Outside the CUDA block on purpose: RPC is backend-independent, and a CPU-only
# worker node is a legitimate configuration. Both states are stated explicitly
# so the cmake configuration and the link line below provably agree.
ifneq ($(filter 1 true yes,$(LLAMA_RPC)),)
  CMAKE_FLAGS += -DGGML_RPC=ON
  # ggml/src/CMakeLists.txt:318-321 puts GGML_USE_RPC on the cmake `ggml` target
  # as a PUBLIC definition. The NIF is compiled by this file with a hand-written
  # g++ line that carries only -I flags, so it never inherits cmake target
  # properties and would compile its #ifdef GGML_USE_RPC blocks out while
  # linking against a libggml-rpc.a that is definitely there. State it.
  CXXFLAGS += -DGGML_USE_RPC
  # RDMA is Linux-only upstream (the CMakeLists forces it off elsewhere), so
  # asking for it on macOS would be a configure-time lie.
  ifeq ($(UNAME_S),Linux)
    ifneq ($(filter 1 true yes,$(LLAMA_RPC_RDMA)),)
      CMAKE_FLAGS += -DGGML_RPC_RDMA=ON
    else
      CMAKE_FLAGS += -DGGML_RPC_RDMA=OFF
    endif
  else
    CMAKE_FLAGS += -DGGML_RPC_RDMA=OFF
  endif
else
  CMAKE_FLAGS += -DGGML_RPC=OFF
endif

# Portable builds, for artifacts that leave this machine. ggml defaults
# GGML_NATIVE to ON unless cross-compiling (vendor/llama.cpp/ggml/CMakeLists.txt),
# which adds -march=native (ggml/src/ggml-cpu/CMakeLists.txt) and ties the binary
# to the build machine's CPU. That is free performance for a local build and a
# SIGILL waiting to happen in a published one, so the precompile workflow sets
# LLAMA_PORTABLE=1 while local builds keep -march=native.
ifneq ($(filter 1 true yes,$(LLAMA_PORTABLE)),)
  CMAKE_FLAGS += -DGGML_NATIVE=OFF
  LLAMA_PORTABLE_SUFFIX = -portable
endif

# The arch pair, emitted after LLAMA_PORTABLE so the two GGML_NATIVE=OFF
# sources are visible together. LLAMA_PORTABLE already turns native off for a
# different reason -- artifacts that leave this machine, on release runners with
# no GPU and therefore no CUDA arch to pin -- so the filter below is what keeps
# the two from double-emitting the flag.
ifneq ($(strip $(LLAMA_CPU_ARM_ARCH)),)
  ifeq ($(strip $(LLAMA_CUDA_ARCH)),)
    ifneq (,$(filter -DGGML_CUDA=ON,$(CMAKE_FLAGS)))
      $(error LLAMA_CPU_ARM_ARCH requires GGML_NATIVE=OFF, which drops ggml-cuda \
        from one architecture to a seven-architecture fat binary. Set \
        LLAMA_CUDA_ARCH too (DGX Spark GB10: LLAMA_CUDA_ARCH=121a-real), or \
        unset LLAMA_CPU_ARM_ARCH)
    endif
  endif
  ifeq (,$(filter -DGGML_NATIVE=OFF,$(CMAKE_FLAGS)))
    CMAKE_FLAGS += -DGGML_NATIVE=OFF
  endif
  CMAKE_FLAGS += -DGGML_CPU_ARM_ARCH=$(LLAMA_CPU_ARM_ARCH)
endif

ifneq ($(strip $(LLAMA_CUDA_ARCH)),)
  ifneq (,$(filter -DGGML_CUDA=ON,$(CMAKE_FLAGS)))
    CMAKE_FLAGS += -DCMAKE_CUDA_ARCHITECTURES=$(LLAMA_CUDA_ARCH)
  endif
endif

# Both of the above select a different set of emitted instructions from the same
# sources, so they belong in the build-directory key alongside the backend --
# otherwise toggling one reuses the previous CMakeCache.txt and silently
# no-ops, which is the exact failure the key exists to prevent. A short hash
# keeps the directory name readable; empty values hash to nothing so today's
# unflagged builds keep their existing directory and stay cache hits.
# Immediate assignment: a recursive one would re-run the shell on every
# expansion of LLAMA_BUILD.
ifneq ($(strip $(LLAMA_CPU_ARM_ARCH))$(strip $(LLAMA_CUDA_ARCH)),)
  LLAMA_ARCH_SUFFIX := -$(shell printf '%s|%s' '$(LLAMA_CPU_ARM_ARCH)' '$(LLAMA_CUDA_ARCH)' \
                         | { shasum -a 256 2>/dev/null || sha256sum; } | cut -c1-8)
endif

# RPC adds a whole backend library plus the public GGML_USE_RPC define, and the
# RDMA toggle changes the code inside it, so both are part of the key too.
# Spelled out rather than hashed: these two show up in every two-node runbook
# and a readable directory name is worth more than eight characters.
ifneq ($(filter 1 true yes,$(LLAMA_RPC)),)
  ifneq (,$(filter -DGGML_RPC_RDMA=ON,$(CMAKE_FLAGS)))
    LLAMA_RPC_SUFFIX = -rpc
  else
    LLAMA_RPC_SUFFIX = -rpc-tcp
  endif
endif

# Custom CMake args
ifdef LLAMA_CMAKE_ARGS
  CMAKE_FLAGS += $(LLAMA_CMAKE_ARGS)
endif

# Build layout. Every key here is load-bearing, because each one selects a
# different cmake configuration or a different set of sources: reusing one build
# tree across them is what made submodule bumps and backend switches silently
# no-op. The directory carries the backend, portability, the architecture flags
# and RPC, so a switch gets a clean CMakeCache.txt (and switching back is still
# a cache hit); the stamp carries the llama.cpp commit, so a bump forces a
# rebuild in place.
LLAMA_BUILD = $(BUILD)/llama_build-$(LLAMA_BACKEND)$(LLAMA_PORTABLE_SUFFIX)$(LLAMA_ARCH_SUFFIX)$(LLAMA_RPC_SUFFIX)
LLAMA_STAMP = $(LLAMA_BUILD)/.built-$(LLAMA_SHA_SHORT)

# Platform-specific linker flags
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -undefined dynamic_lookup
  LDFLAGS += -framework Foundation -framework Accelerate
  ifneq (,$(filter -DGGML_METAL=ON,$(CMAKE_FLAGS)))
    LDFLAGS += -framework Metal -framework MetalKit
  endif
else
  LDFLAGS += -lstdc++ -lm -lpthread
  # ggml-cpu uses OpenMP on Linux when available
  ifneq ($(shell $(CXX) -fopenmp -E - < /dev/null 2>/dev/null && echo yes),)
    LDFLAGS += -lgomp
  endif
  # Pairs with -DGGML_RPC_RDMA=ON above. ggml-rpc's CMakeLists links ibverbs
  # with target_link_libraries, which this hand-assembled link line never sees,
  # so without this libggml-rpc.a's undefined ibv_* symbols surface as a NIF
  # that will not dlopen.
  ifneq (,$(filter -DGGML_RPC_RDMA=ON,$(CMAKE_FLAGS)))
    LDFLAGS += -libverbs
  endif
  # ggml-cuda.a leaves the CUDA runtime, cuBLAS/cuBLASLt and the CUDA driver API
  # unresolved, but this line only ever added -lstdc++ -lm -lpthread. The .so
  # then linked and failed at load with `undefined symbol: cuMemCreate`, a
  # driver-API symbol ggml-cuda's VMM pool calls.
  ifneq (,$(filter -DGGML_CUDA=ON,$(CMAKE_FLAGS)))
    ifeq ($(strip $(CUDA_LIBDIR)),)
      $(error CUDA backend selected but no CUDA toolkit libraries were found. \
        Set CUDA_HOME to the toolkit root, the directory holding bin/nvcc.)
    endif
    LDFLAGS += -L$(CUDA_LIBDIR)
    # -lcuda is the driver API, which is shipped by the driver and not by the
    # toolkit, so it is missing on every GPU-less build host including the
    # release runners. The stub carries SONAME libcuda.so.1, so linking against
    # it resolves the symbols at build time and still loads the real driver
    # library at run time.
    ifneq ($(wildcard $(CUDA_LIBDIR)/stubs),)
      LDFLAGS += -L$(CUDA_LIBDIR)/stubs
    endif
    LDFLAGS += -lcudart -lcublas -lcublasLt -lcuda
    # Matches the -DGGML_CUDA_NCCL above. Opting in without this pairing is the
    # `undefined symbol: ncclAllReduce` load failure.
    ifneq ($(filter 1 true yes,$(LLAMA_CUDA_NCCL)),)
      LDFLAGS += -lnccl
    endif
  endif
endif

# CPU count for parallel builds
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Sources
NIF_SRC = c_src/llama_cpp_ex/llama_nif.cpp
NIF_OBJ = $(BUILD)/llama_nif.o

# Everything downstream depends on the build *configuration*, not only on file
# timestamps, and three separate rules were blind to it:
#
#   - Toggling LLAMA_RPC changes CXXFLAGS (-DGGML_USE_RPC) and the archive set
#     while touching no source file, so make kept a previously linked .so.
#     Observed twice: once as `Function not found model_load/11`, once as an RPC
#     build reporting :rpc_unsupported at runtime. Silent, and it looks like an
#     Elixir bug.
#   - LDFLAGS changes (-lnccl, -libverbs) are invisible to every source rule.
#   - A CMAKE_FLAGS change that does not alter the build directory -- NCCL, for
#     instance -- left $(LLAMA_STAMP) newer than CMakeLists.txt, so cmake was
#     never re-run and the archives kept the previous configuration.
#
# This stamp's *name* carries a hash of all three, so any change makes the
# prerequisite disappear and the cmake configure, the compile and the link all
# rerun. cmake is incremental, so the cost of a reconfigure is small and the
# alternative is a build that silently does not match its flags.
LLAMA_CONFIG_HASH := $(shell printf '%s|%s|%s|%s' \
                       '$(CXXFLAGS)' '$(LDFLAGS)' '$(CMAKE_FLAGS)' '$(LLAMA_BUILD)' \
                       | { shasum -a 256 2>/dev/null || sha256sum; } | cut -c1-12)
LLAMA_CONFIG_STAMP = $(BUILD)/.config-$(LLAMA_CONFIG_HASH)

# ...and one more marker, next to the artifact rather than next to the objects,
# because the artifact is SHARED and gets overwritten from two directions.
#
# Mix symlinks $(MIX_APP_PATH)/priv to the project's priv/ in every MIX_ENV, so
# dev, test and bench all write one llama_cpp_ex_nif.so while each keeps its own
# objects and its own llama.cpp tree. Two consequences, both observed:
#
#   1. Build test with LLAMA_RPC=1 and bench without it, and whichever ran last
#      is what every environment loads. The plain timestamp rule then sees a .so
#      newer than its own object and skips the relink, so it stays wrong --
#      a test suite reporting {:error, :rpc_unsupported} against a live RPC
#      worker the same tree had just talked to.
#   2. Mix restores a downloaded precompiled artifact into priv/ and copies it
#      over the source build, with a fresh mtime, so no timestamp comparison can
#      see it either.
#
# So the marker records what the artifact IS, not when it was written: the
# configuration hash plus a digest of the bytes that were linked. Anything that
# replaces the artifact -- another MIX_ENV, a downloaded release, a stray cp --
# breaks the digest and forces a relink.
NIF_LINK_STAMP = $(PREFIX)/.llama_cpp_ex_nif.built
# The file is an argument, not stdin: a redirect here would attach to `cut`.
NIF_DIGEST = { shasum -a 256 "$(NIF_SO)" 2>/dev/null || sha256sum "$(NIF_SO)"; } | cut -c1-32

# Targets
.PHONY: all clean rpc-server check-artifact

# Serial, so check-artifact provably runs before the link decision below.
# Nothing here benefits from make's -j: the heavy build is cmake's own
# `--build -j$(NPROC)`.
.NOTPARALLEL:

all: check-artifact $(NIF_LINK_STAMP)

check-artifact:
	@if [ -f "$(NIF_LINK_STAMP)" ]; then \
	  want="$(LLAMA_CONFIG_HASH) $$($(NIF_DIGEST))"; \
	  if [ "$$(cat "$(NIF_LINK_STAMP)")" != "$$want" ]; then \
	    echo "==> $(NIF_SO) does not match this build; relinking"; \
	    rm -f "$(NIF_LINK_STAMP)"; \
	  fi; \
	fi

# Upstream's standalone RPC worker. Not the production path -- LlamaCppEx.RPC.Server
# hosts the same server inside the NIF, under OTP supervision -- but when a
# two-node run misbehaves the first question is "is this us or upstream", and
# having both answers costs 1m32s of build time.
#
# Its own build directory, because it needs LLAMA_BUILD_TOOLS=ON and the NIF's
# tree deliberately does not build tools. The cmake target is `ggml-rpc-server`,
# not `rpc-server` (vendor/llama.cpp/tools/rpc/CMakeLists.txt:1).
RPC_SERVER_BUILD = $(BUILD)/rpc_server_build
RPC_SERVER_BIN = $(RPC_SERVER_BUILD)/bin/ggml-rpc-server

rpc-server: $(RPC_SERVER_BIN)
	@echo "built $(RPC_SERVER_BIN)"

$(RPC_SERVER_BIN): $(LLAMA_DIR)/CMakeLists.txt
	@test -n "$(filter 1 true yes,$(LLAMA_RPC))" || { \
	  echo "error: rpc-server needs LLAMA_RPC=1"; exit 1; }
	cmake -B $(RPC_SERVER_BUILD) -S $(LLAMA_DIR) $(CMAKE_FLAGS) -DLLAMA_BUILD_TOOLS=ON
	cmake --build $(RPC_SERVER_BUILD) --config Release -j$(NPROC) --target ggml-rpc-server

# `make print-CMAKE_FLAGS` echoes one variable, fully expanded. `make -p` cannot
# do this: CMAKE_FLAGS is a recursive variable, so the database dump shows the
# unexpanded text. test/makefile_arch_flags_test.exs asserts on what this build
# will really hand to cmake, and that needs the expansion.
print-%:
	@echo '$($*)'

# Materialize vendor/llama.cpp when it is absent, which is the Hex-tarball case:
# no vendor/ directory and no surrounding git repository. Pinned to LLAMA_COMMIT
# so the tree matches the submodule a git checkout would use. GitHub serves
# arbitrary SHAs, so `init` + `fetch --depth 1 <sha>` gets exactly that commit
# without cloning any history.
$(LLAMA_DIR)/CMakeLists.txt:
	@command -v git >/dev/null 2>&1 || { \
	  echo "error: vendor/llama.cpp is missing and git is not in PATH."; \
	  echo "       Install git, or place a llama.cpp checkout at vendor/llama.cpp."; \
	  exit 1; }
	@if [ -e $(LLAMA_DIR)/.git ]; then \
	  echo "error: $(LLAMA_DIR) exists as a git checkout but has no CMakeLists.txt."; \
	  echo "       In a git checkout the submodule is not fully initialized:"; \
	  echo "         git submodule update --init --recursive"; \
	  echo "       Otherwise a previous clone was interrupted; remove the tree and"; \
	  echo "       rebuild:  rm -rf $(LLAMA_DIR)"; \
	  exit 1; \
	fi
	@echo "==> vendor/llama.cpp not found; cloning $(LLAMA_REPO) at $(LLAMA_COMMIT)"
	rm -rf $(LLAMA_DIR)
	@mkdir -p $(dir $(LLAMA_DIR))
	git -c init.defaultBranch=master init -q $(LLAMA_DIR)
	git -C $(LLAMA_DIR) remote add origin $(LLAMA_REPO)
	git -C $(LLAMA_DIR) fetch --depth 1 --quiet origin $(LLAMA_COMMIT)
	git -C $(LLAMA_DIR) checkout --quiet FETCH_HEAD
	@test -f $@ || { echo "error: clone finished but $@ is still missing"; exit 1; }

# Build llama.cpp static libraries
$(LLAMA_STAMP): $(LLAMA_DIR)/CMakeLists.txt $(LLAMA_CONFIG_STAMP)
	@mkdir -p $(LLAMA_BUILD)
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) $(CMAKE_FLAGS)
	cmake --build $(LLAMA_BUILD) --config Release -j$(NPROC)
	@rm -f $(LLAMA_BUILD)/.built-*
	@touch $@

# Regenerated whenever the configuration hash changes, which is what forces the
# compile and the link below to rerun. Old stamps are swept so $(BUILD) does not
# accumulate one per configuration ever tried.
$(LLAMA_CONFIG_STAMP):
	@mkdir -p $(dir $@)
	@rm -f $(BUILD)/.config-*
	@touch $@

# Compile NIF
$(NIF_OBJ): $(NIF_SRC) c_src/llama_cpp_ex/llama_nif.h $(LLAMA_STAMP) $(LLAMA_CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $(NIF_SRC) -o $@

# Link NIF - find all static libs from llama.cpp build.
# The target is the marker, not the .so: see NIF_LINK_STAMP above for why the
# artifact's own timestamp cannot be trusted.
$(NIF_LINK_STAMP): $(NIF_OBJ) $(LLAMA_STAMP) $(LLAMA_CONFIG_STAMP)
	@mkdir -p $(PREFIX)
	@LIBS=$$(find $(LLAMA_BUILD) -name '*.a' \
		! -path '*/CMakeFiles/*' \
		! -path '*/examples/*' \
		! -path '*/tests/*' \
		| sort); \
	if [ "$(UNAME_S)" = "Linux" ]; then \
		$(CXX) $(NIF_OBJ) -Wl,--start-group $$LIBS -Wl,--end-group $(LDFLAGS) -o $(NIF_SO); \
	else \
		$(CXX) $(NIF_OBJ) $$LIBS $(LDFLAGS) -o $(NIF_SO); \
	fi
	@printf '%s %s' '$(LLAMA_CONFIG_HASH)' "$$($(NIF_DIGEST))" > $@

clean:
	rm -rf $(BUILD) $(PREFIX)/llama_cpp_ex_nif.so $(NIF_LINK_STAMP)
