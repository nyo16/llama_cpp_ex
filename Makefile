# Makefile for llama_cpp_ex NIF
# Called by elixir_make during `mix compile`

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
LLAMA_COMMIT ?= 61881b1f7f0b13d9e46d561fc25afcd6bbaec479

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
    ifneq ($(shell which nvcc 2>/dev/null),)
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

# Custom CMake args
ifdef LLAMA_CMAKE_ARGS
  CMAKE_FLAGS += $(LLAMA_CMAKE_ARGS)
endif

# Build layout. Every key here is load-bearing, because each one selects a
# different cmake configuration or a different set of sources: reusing one build
# tree across them is what made submodule bumps and backend switches silently
# no-op. The directory carries the backend and portability, so a switch gets a
# clean CMakeCache.txt (and switching back is still a cache hit); the stamp
# carries the llama.cpp commit, so a bump forces a rebuild in place.
LLAMA_BUILD = $(BUILD)/llama_build-$(LLAMA_BACKEND)$(LLAMA_PORTABLE_SUFFIX)
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
  # ggml-cuda.a leaves the CUDA runtime, cuBLAS, and driver API unresolved.
  # The stubs dir lets -lcuda link on hosts without a driver (e.g. release CI);
  # the real libcuda.so.1 is picked up from the driver at load time.
  ifneq (,$(filter -DGGML_CUDA=ON,$(CMAKE_FLAGS)))
    CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell which nvcc 2>/dev/null))
    ifneq ($(CUDA_HOME),)
      LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs
    endif
    LDFLAGS += -lcudart -lcublas -lcublasLt -lcuda
  endif
endif

# CPU count for parallel builds
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Sources
NIF_SRC = c_src/llama_cpp_ex/llama_nif.cpp
NIF_OBJ = $(BUILD)/llama_nif.o

# Targets
.PHONY: all clean

all: $(NIF_SO)

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
$(LLAMA_STAMP): $(LLAMA_DIR)/CMakeLists.txt
	@mkdir -p $(LLAMA_BUILD)
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) $(CMAKE_FLAGS)
	cmake --build $(LLAMA_BUILD) --config Release -j$(NPROC)
	@rm -f $(LLAMA_BUILD)/.built-*
	@touch $@

# Compile NIF
$(NIF_OBJ): $(NIF_SRC) c_src/llama_cpp_ex/llama_nif.h $(LLAMA_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $(NIF_SRC) -o $@

# Link NIF - find all static libs from llama.cpp build
$(NIF_SO): $(NIF_OBJ) $(LLAMA_STAMP)
	@mkdir -p $(PREFIX)
	@LIBS=$$(find $(LLAMA_BUILD) -name '*.a' \
		! -path '*/CMakeFiles/*' \
		! -path '*/examples/*' \
		! -path '*/tests/*' \
		| sort); \
	if [ "$(UNAME_S)" = "Linux" ]; then \
		$(CXX) $(NIF_OBJ) -Wl,--start-group $$LIBS -Wl,--end-group $(LDFLAGS) -o $@; \
	else \
		$(CXX) $(NIF_OBJ) $$LIBS $(LDFLAGS) -o $@; \
	fi

clean:
	rm -rf $(BUILD) $(PREFIX)/llama_cpp_ex_nif.so
