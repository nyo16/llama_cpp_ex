# Makefile for llama_cpp_ex NIF
# Called by elixir_make during `mix compile`

PREFIX = $(MIX_APP_PATH)/priv
BUILD  = $(MIX_APP_PATH)/obj
NIF_SO = $(PREFIX)/llama_cpp_ex_nif.so

LLAMA_DIR   = $(shell pwd)/vendor/llama.cpp
LLAMA_BUILD = $(BUILD)/llama_build

# Compiler
CXX      ?= c++
CXXFLAGS  = -std=c++17 -O2 -fPIC -fvisibility=hidden -Wall -Wno-unused-parameter
CXXFLAGS += -I$(ERTS_INCLUDE_DIR)
CXXFLAGS += -I$(FINE_INCLUDE_DIR)
CXXFLAGS += -I$(LLAMA_DIR)/include
CXXFLAGS += -I$(LLAMA_DIR)/ggml/include

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

# Custom CMake args
ifdef LLAMA_CMAKE_ARGS
  CMAKE_FLAGS += $(LLAMA_CMAKE_ARGS)
endif

# Platform-specific linker flags
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -undefined dynamic_lookup
  LDFLAGS += -framework Foundation -framework Accelerate
  ifneq (,$(filter -DGGML_METAL=ON,$(CMAKE_FLAGS)))
    LDFLAGS += -framework Metal -framework MetalKit
  endif
else
  LDFLAGS += -lstdc++ -lm -lpthread
endif

# CPU count for parallel builds
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Sources
NIF_SRC = c_src/llama_cpp_ex/llama_nif.cpp
NIF_OBJ = $(BUILD)/llama_nif.o

# Targets
.PHONY: all clean

all: $(NIF_SO)

# Build llama.cpp static libraries
$(LLAMA_BUILD)/.built: $(LLAMA_DIR)/CMakeLists.txt
	@mkdir -p $(LLAMA_BUILD)
	cmake -B $(LLAMA_BUILD) -S $(LLAMA_DIR) $(CMAKE_FLAGS)
	cmake --build $(LLAMA_BUILD) --config Release -j$(NPROC)
	@touch $@

# Compile NIF
$(NIF_OBJ): $(NIF_SRC) c_src/llama_cpp_ex/llama_nif.h $(LLAMA_BUILD)/.built
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $(NIF_SRC) -o $@

# Link NIF - find all static libs from llama.cpp build
$(NIF_SO): $(NIF_OBJ) $(LLAMA_BUILD)/.built
	@mkdir -p $(PREFIX)
	@LIBS=$$(find $(LLAMA_BUILD) -name '*.a' \
		! -path '*/CMakeFiles/*' \
		! -path '*/examples/*' \
		! -path '*/tests/*' \
		! -path '*/common/*' \
		| sort); \
	if [ "$(UNAME_S)" = "Linux" ]; then \
		$(CXX) $(NIF_OBJ) -Wl,--start-group $$LIBS -Wl,--end-group $(LDFLAGS) -o $@; \
	else \
		$(CXX) $(NIF_OBJ) $$LIBS $(LDFLAGS) -o $@; \
	fi

clean:
	rm -rf $(BUILD) $(PREFIX)/llama_cpp_ex_nif.so
