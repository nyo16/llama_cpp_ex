# Cross-Platform Builds

## Supported Platforms

Two axes matter here and they are easy to conflate: whether a **precompiled
artifact** exists, and what a **source build** auto-detects. A precompiled
artifact never runs the Makefile, so it can only contain the backend it was built
with.

| Platform | Precompiled artifact | Source-build default | Status |
|---|---|---|---|
| macOS (Apple Silicon) | Yes — Metal | Metal | Tested |
| macOS (Intel) | No | CPU | Supported, source build |
| Linux (x86_64, glibc) | Yes — **CPU only** | CUDA if `nvcc` found, else CPU | Supported |
| Linux (x86_64) + NVIDIA | No | CUDA if `nvcc` found | Supported via `LLAMA_BACKEND=cuda` |
| Linux (x86_64) + AMD | No | CPU | Supported via `LLAMA_BACKEND=vulkan` |
| Linux (aarch64), musl | No | as above | Supported, source build |
| Windows (WSL2) | No | Same as Linux | Supported, source build |

Artifacts are published for NIF 2.17 and 2.18, which means Erlang/OTP 26 or
newer. Anything else — including OTP 25, which reports NIF 2.16 — falls back to a
source build. The NIF-version-to-OTP mapping is verified against
`erts/emulator/beam/erl_nif.h` in the OTP source; see the comment on
`make_precompiler_nif_versions` in `mix.exs`, which also explains why no "2.16"
entry is declared even though OTP 25 works.

## Backend Selection

### Default: precompiled first, source build second

```bash
mix compile
```

If a precompiled artifact matches this OS, architecture and NIF version,
`mix compile` downloads it and stops. **The Makefile does not run, so nothing is
auto-detected**, and the artifact's backend is whatever it was built with: Metal
on Apple Silicon, CPU on `x86_64-linux-gnu`. A Linux user with a working CUDA
toolkit still gets a CPU-only binary from this path.

Only when no artifact matches does the Makefile run, and only then is a backend
detected:

1. **macOS** → Metal
2. **Linux with `nvcc` in PATH** → CUDA
3. **Otherwise** → CPU

### Explicit Backend

Setting `LLAMA_BACKEND` forces a source build (`make_force_build` in `mix.exs`),
which is the supported way to get CUDA or Vulkan:

```bash
LLAMA_BACKEND=metal  mix compile   # Apple Silicon GPU
LLAMA_BACKEND=cuda   mix compile   # NVIDIA GPU (requires CUDA toolkit)
LLAMA_BACKEND=vulkan mix compile   # Vulkan (requires Vulkan SDK)
LLAMA_BACKEND=cpu    mix compile   # CPU only (no GPU acceleration)
```

A source build needs the llama.cpp sources. In a git checkout those are the
`vendor/llama.cpp` submodule. The Hex package does not ship the tree — it would
add hundreds of megabytes to every release — and ships `.gitmodules` instead, so
the Makefile clones the pinned `LLAMA_COMMIT` into `vendor/llama.cpp` on the
first build. That needs `git` and network access.

### Portable Builds

`LLAMA_PORTABLE=1` adds `-DGGML_NATIVE=OFF`:

```bash
LLAMA_PORTABLE=1 LLAMA_BACKEND=cpu mix compile
```

ggml defaults `GGML_NATIVE` to `ON`, which adds `-march=native` and ties the
resulting binary to the exact CPU it was built on. That is free performance for a
build that never leaves the machine and a `SIGILL` on a user's older CPU for one
that does, so the release workflow sets `LLAMA_PORTABLE=1` for every published
artifact. Set it yourself whenever you build a binary you intend to copy
elsewhere — into a container image that runs on heterogeneous hosts, for example.

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

Nothing to install: `mix compile` downloads the precompiled Metal artifact.

```bash
mix deps.get
mix compile

# Verify Metal is active (look for "Metal" in model load logs)
```

To build from source instead — a local llama.cpp patch, say — you need cmake, and
Metal is what the Makefile auto-detects on macOS:

```bash
brew install cmake
LLAMA_BACKEND=metal mix compile
```

**Performance tips:**
- Use `n_gpu_layers: -1` to offload all layers to GPU
- Apple Silicon unified memory means no CPU-GPU transfer overhead

### macOS (Intel)

No artifact is published for `x86_64-apple-darwin`, so this is always a source
build. It falls back to CPU — Metal requires Apple Silicon.

```bash
brew install cmake
mix deps.get
mix compile
```

### Linux (CPU)

`mix compile` downloads the precompiled CPU artifact on `x86_64-linux-gnu` with
OTP 26+. Other architectures, musl, or OTP 25 build from source and need
`build-essential`, `cmake` and `git`.

```bash
mix deps.get
mix compile
```

### Linux (NVIDIA CUDA)

**A plain `mix compile` will not give you CUDA.** The published
`x86_64-linux-gnu` artifact is a CPU build, and downloading it skips the Makefile
entirely, so `nvcc` is never looked for. CUDA requires an explicit source build:

```bash
# Prerequisites
sudo apt-get install build-essential cmake git
# Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
nvcc --version

mix deps.get
LLAMA_BACKEND=cuda mix compile
```

`LLAMA_BACKEND=cuda` forces the source build and selects CUDA explicitly. The
Makefile's `nvcc` auto-detection only ever applies to a source build that ran for
some other reason, so do not rely on it.

**CUDA version compatibility:**
- CUDA 11.7+ recommended
- CUDA 12.x preferred for latest GPU architectures
- The build uses static CUDA libraries by default

### Linux (Vulkan)

Same story as CUDA: there is no Vulkan artifact, so this is always an explicit
source build.

```bash
# Prerequisites
sudo apt-get install build-essential cmake git
# Install Vulkan SDK: https://vulkan.lunarg.com/sdk/home
sudo apt-get install libvulkan-dev vulkan-tools

mix deps.get
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

In a git checkout the llama.cpp submodule may not be initialized:

```bash
git submodule update --init --recursive
mix compile
```

Installed from Hex there is no submodule; the Makefile clones the pinned commit
itself. A failed clone stops the build with an explicit message rather than a
missing header, so check that `git` is installed and that
`https://github.com/ggml-org/llama.cpp` is reachable.

### Build Fails: "CMake Error"

Ensure CMake 3.14+ is installed:

```bash
cmake --version
```

### CUDA Not Detected

First check whether a source build ran at all. If `mix compile` downloaded a
precompiled artifact then the Makefile never executed and `nvcc` was never
consulted — the binary is CPU-only by construction. Force a source build:

```bash
LLAMA_BACKEND=cuda mix compile
```

Then verify `nvcc` is in your PATH:

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

Switching backends no longer needs a manual clean. The llama.cpp build directory
is keyed on `LLAMA_BACKEND` (`llama_build-<backend>` under `_build`) and the build
stamp is keyed on the llama.cpp commit, so a backend switch and a submodule bump
each force a fresh configure and build:

```bash
LLAMA_BACKEND=cuda mix compile
```

Every backend keeps its own build tree, so switching back is a cache hit. `mix
clean` removes all of them.

## Build Internals

A source build has four stages:

1. **The llama.cpp sources are located** — the `vendor/llama.cpp` submodule in a git checkout, otherwise a shallow clone of the pinned `LLAMA_COMMIT`
2. **CMake configures llama.cpp** with the selected backend flags
3. **CMake builds static libraries** (`libllama.a`, `libggml.a`, `libggml-base.a`, and backend-specific libs)
4. **The NIF is compiled and linked** against these static libraries plus `erl_nif.h` and `fine.hpp`

Static linking produces a self-contained `.so`/`.dylib` with no runtime dependencies on llama.cpp (only on system libraries like CUDA runtime or Metal framework).

### Build Output

```
priv/
└── llama_cpp_ex_nif.so    # (or .dylib on macOS)
```

This single file contains all of llama.cpp statically linked in. No `RPATH` or `LD_LIBRARY_PATH` configuration needed.
