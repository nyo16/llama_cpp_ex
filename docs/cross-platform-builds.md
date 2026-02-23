# Cross-Platform Builds

## Supported Platforms

| Platform | Backend | Auto-detected? | Status |
|---|---|---|---|
| macOS (Apple Silicon) | Metal | Yes | Tested |
| macOS (Intel) | CPU | Yes | Supported |
| Linux (x86_64) | CPU | Yes (fallback) | Supported |
| Linux (x86_64) + NVIDIA | CUDA | Yes (if `nvcc` found) | Supported |
| Linux (x86_64) + AMD | Vulkan | No (manual) | Supported |
| Windows (WSL2) | CPU/CUDA | Same as Linux | Supported |

## Backend Selection

### Auto-Detection (Default)

```bash
mix compile
```

The Makefile auto-detects the best backend:
1. **macOS** → Metal
2. **Linux with `nvcc` in PATH** → CUDA
3. **Otherwise** → CPU

### Explicit Backend

```bash
LLAMA_BACKEND=metal  mix compile   # Apple Silicon GPU
LLAMA_BACKEND=cuda   mix compile   # NVIDIA GPU (requires CUDA toolkit)
LLAMA_BACKEND=vulkan mix compile   # Vulkan (requires Vulkan SDK)
LLAMA_BACKEND=cpu    mix compile   # CPU only (no GPU acceleration)
```

### Custom CMake Flags

For advanced users who need fine-grained control over the llama.cpp build:

```bash
# Force cuBLAS for CUDA
LLAMA_CMAKE_ARGS="-DGGML_CUDA_FORCE_CUBLAS=ON" mix compile

# Enable specific CUDA architectures
LLAMA_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" mix compile

# Combine backend + custom flags
LLAMA_BACKEND=cuda LLAMA_CMAKE_ARGS="-DGGML_CUDA_F16=ON" mix compile
```

## Platform-Specific Instructions

### macOS (Apple Silicon)

No additional setup required. Metal is auto-detected.

```bash
# Prerequisites
brew install cmake

# Build (Metal auto-detected)
mix deps.get
mix compile

# Verify Metal is active (look for "Metal" in model load logs)
```

**Performance tips:**
- Use `n_gpu_layers: -1` to offload all layers to GPU
- Apple Silicon unified memory means no CPU-GPU transfer overhead

### macOS (Intel)

Falls back to CPU. No GPU acceleration available (Metal requires Apple Silicon).

```bash
mix deps.get
mix compile
```

### Linux (CPU)

```bash
# Prerequisites
sudo apt-get install build-essential cmake

# Build
mix deps.get
mix compile
```

### Linux (NVIDIA CUDA)

```bash
# Prerequisites
# Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
# Ensure nvcc is in PATH:
nvcc --version

# Build (CUDA auto-detected if nvcc found)
mix deps.get
mix compile

# Or explicitly:
LLAMA_BACKEND=cuda mix compile
```

**CUDA version compatibility:**
- CUDA 11.7+ recommended
- CUDA 12.x preferred for latest GPU architectures
- The build uses static CUDA libraries by default

### Linux (Vulkan)

```bash
# Prerequisites
# Install Vulkan SDK: https://vulkan.lunarg.com/sdk/home
sudo apt-get install libvulkan-dev vulkan-tools

# Build
LLAMA_BACKEND=vulkan mix deps.get
LLAMA_BACKEND=vulkan mix compile
```

### Windows (via WSL2)

Run inside WSL2 with a Linux distribution. Follow the Linux instructions above.

For CUDA on WSL2, install the [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) driver on the Windows host.

## Docker

### CPU-Only

```dockerfile
FROM elixir:1.17-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY mix.exs mix.lock ./
RUN mix deps.get && mix deps.compile

COPY . .
RUN LLAMA_BACKEND=cpu mix compile
```

### NVIDIA CUDA

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install Erlang/Elixir
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget \
    && rm -rf /var/lib/apt/lists/*

# Install Erlang + Elixir (via asdf, mise, or system packages)
# ...

WORKDIR /app
COPY mix.exs mix.lock ./
RUN mix deps.get && mix deps.compile

COPY . .
RUN LLAMA_BACKEND=cuda mix compile
```

Run with GPU access:
```bash
docker run --gpus all my-llama-app
```

## Troubleshooting

### Build Fails: "Cannot find llama.h"

The llama.cpp submodule may not be initialized:

```bash
git submodule update --init --recursive
mix compile
```

### Build Fails: "CMake Error"

Ensure CMake 3.14+ is installed:

```bash
cmake --version
```

### CUDA Not Detected

Verify `nvcc` is in your PATH:

```bash
which nvcc
nvcc --version
```

If using a non-standard CUDA installation path:

```bash
LLAMA_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc" mix compile
```

### Metal Errors on macOS

Ensure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

### Linking Errors on Linux

If you see undefined symbol errors during linking, ensure all required system libraries are present:

```bash
# For CPU builds
sudo apt-get install build-essential

# For Vulkan builds
sudo apt-get install libvulkan-dev
```

### Recompiling After Backend Change

The build system caches the CMake output. To switch backends, clean first:

```bash
mix clean
LLAMA_BACKEND=cuda mix compile
```

## Build Internals

The build process has three stages:

1. **CMake configures llama.cpp** with the selected backend flags
2. **CMake builds static libraries** (`libllama.a`, `libggml.a`, `libggml-base.a`, and backend-specific libs)
3. **The NIF is compiled and linked** against these static libraries plus `erl_nif.h` and `fine.hpp`

Static linking produces a self-contained `.so`/`.dylib` with no runtime dependencies on llama.cpp (only on system libraries like CUDA runtime or Metal framework).

### Build Output

```
priv/
└── llama_cpp_ex_nif.so    # (or .dylib on macOS)
```

This single file contains all of llama.cpp statically linked in. No `RPATH` or `LD_LIBRARY_PATH` configuration needed.
