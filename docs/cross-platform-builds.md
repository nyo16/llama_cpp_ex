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
| Linux (x86_64, glibc) | Yes — CPU | CUDA if a toolkit is found, else CPU | Supported |
| Linux (x86_64) + NVIDIA, CUDA 12 | Yes — **CUDA** (`-cu12`) | CUDA | Supported |
| Linux (x86_64) + NVIDIA, CUDA 13 | Yes — **CUDA** (`-cu13`) | CUDA | Supported |
| Linux (x86_64) + AMD | No | CPU | Supported via `LLAMA_BACKEND=vulkan` |
| Linux (aarch64) + NVIDIA | No | CUDA if a toolkit is found | Tested on DGX Spark (GB10), source build — see [DGX Spark](dgx-spark.md) |
| Linux (aarch64), musl | No | as above | Supported, source build |
| Windows (WSL2) | No | Same as Linux | Supported, source build |

### Why CUDA is split by major version

The NIF links `libcudart`, `libcublas` and `libcublasLt` dynamically, and those
sonames are major-versioned — `libcudart.so.12` and `libcudart.so.13` are
different files with no compatibility shim in either direction. One "Linux CUDA"
artifact therefore cannot exist; each CUDA major gets its own target name.

Running one needs the CUDA **runtime** libraries, not the toolkit: no `nvcc`
required.

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

If a precompiled artifact matches this OS, architecture, NIF version and CUDA
variant, `mix compile` downloads it and stops. **The Makefile does not run, so
nothing is auto-detected** — the artifact's backend is whatever it was built
with.

On `x86_64-linux-gnu` the choice between the CPU artifact and a CUDA one is made
by `LlamaCppEx.Precompiler` in `mix.exs`, which looks for two things and needs
both:

1. a driver, `libcuda.so.1`, and
2. a CUDA runtime, `libcudart.so.13` then `libcudart.so.12`, newest first.

The driver half is not belt-and-braces. A CUDA build links `-lcuda`, so on a
machine with a toolkit and no driver it cannot be `dlopen`ed at all — handing
that machine a CUDA artifact would turn a working CPU install into a NIF that
fails to load. Absent either half, the CPU artifact is chosen.

Both are looked up through `ldconfig -p` and, failing that, on disk under
`/usr/local/cuda*/lib64`, `/usr/local/cuda-*/targets/*/lib`,
`/usr/lib/x86_64-linux-gnu` and `/usr/lib64`.

`LLAMA_CUDA_VARIANT` overrides the result: `cu12`, `cu13`, or `none` to force the
CPU artifact.

Only when no artifact matches does the Makefile run, and only then is a backend
detected:

1. **macOS** → Metal
2. **Linux with a CUDA toolkit** → CUDA
3. **Otherwise** → CPU

"With a CUDA toolkit" means the Makefile found one, which is a wider test than
`nvcc` being on `PATH`: `CUDA_HOME`, then `CUDA_PATH`, then `nvcc` on `PATH`,
then `/usr/local/cuda`, `/opt/cuda`, then the newest `/usr/local/cuda-*`. DGX OS
puts `nvcc` on the login `PATH` only, via `/etc/profile.d/nv_paths.sh`, which
`ssh host mix compile`, systemd units and most CI shells never source — probing
`PATH` alone silently produced CPU-only builds on machines with a full toolkit.

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

### CPU and CUDA architecture flags

Two paired variables, for hosts where ggml's `-mcpu=native` probe gives the
wrong answer:

```bash
LLAMA_BACKEND=cuda \
LLAMA_CPU_ARM_ARCH=armv9.2-a+dotprod+i8mm+fp16+bf16+sve2 \
LLAMA_CUDA_ARCH=121a-real \
  mix compile
```

`LLAMA_CPU_ARM_ARCH` names the architecture for ggml's CPU backend instead of
letting it probe. On a DGX Spark (GB10, Cortex-X925 + A725) with GCC 13.3 the
probe fails **silently**: the compiler predates those cores, rejects
`-mcpu=cortex-x925`, and falls back to base ARMv8-A behind a soft CMake warning
and a zero exit status. The cost is not subtle — `objdump` on the emitted
`libggml-cpu.a` finds **0 `sdot`, 0 `smmla`, no SVE** either way, versus
**1134 `sdot`, 370 `smmla`** and SVE once the architecture is named. Those are
the Q4/Q8 quantized matmul kernels.

`LLAMA_CUDA_ARCH` sets `CMAKE_CUDA_ARCHITECTURES`, and on a CUDA build it is
**required** whenever `LLAMA_CPU_ARM_ARCH` is set — the Makefile errors out
otherwise. Reaching the CPU flag requires `GGML_NATIVE=OFF`, and with native off
ggml-cuda stops compiling for the GPU it can see and produces a seven-architecture
fat binary instead. That is a roughly 6× build-time regression for no runtime
benefit, and nothing warns about it, so the two variables move together.

Both are part of the build-directory key, so toggling either lands in a fresh
cmake tree rather than silently reusing a stale `CMakeCache.txt`. Switching back
is still a cache hit.

To check that the flags survived cmake all the way into the machine code:

```bash
scripts/spark/verify-build-flags.sh
```

`LLAMA_PORTABLE=1` also sets `GGML_NATIVE=OFF`, for the different reason above.
The two never double-emit it, and portable builds still need no CUDA
architecture — the release runners have no GPU to pin one for.

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

On x86_64 a plain `mix compile` now does give you CUDA, provided the machine has
a driver and a CUDA 12 or CUDA 13 runtime. No toolkit and no build tools are
needed for that path — the `-cu12`/`-cu13` artifact is downloaded like any other:

```bash
mix deps.get
mix compile
```

Confirm what you got:

```elixir
LlamaCppEx.devices()   # backend "CUDA" on a CUDA artifact
```

A source build is still required for aarch64 Linux, for a CUDA major with no
published artifact, and any time you want flags of your own:

```bash
# Prerequisites
sudo apt-get install build-essential cmake git
# Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads

mix deps.get
LLAMA_BACKEND=cuda mix compile
```

`LLAMA_BACKEND=cuda` forces the source build and selects CUDA explicitly. The
toolkit does not have to be on `PATH`; see the discovery order above, or set
`CUDA_HOME` to the directory holding `bin/nvcc`. If nothing is found the build
fails with that message rather than quietly linking against nothing.

**CUDA version compatibility:**
- CUDA 12 and CUDA 13 have published artifacts. Older majors build from source.
- Architectures come from ggml's portable default list under `LLAMA_PORTABLE=1`,
  which covers Turing through Blackwell and includes `sm_121a` for GB10; a local
  build without it compiles for the GPU actually present.
- NCCL is off unless `LLAMA_CUDA_NCCL=1`. See the README's build variables for
  why the default is not ggml's.
- On **aarch64** (DGX Spark and friends) also set `LLAMA_CPU_ARM_ARCH` and
  `LLAMA_CUDA_ARCH` — see "CPU and CUDA architecture flags" above, and
  [DGX Spark](dgx-spark.md) for the measured values. Without them the CPU
  backend is built at base ARMv8-A with no quantized matmul kernels, silently.

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

Find out which binary you are running before changing anything:

```elixir
LlamaCppEx.devices()   # backend "CUDA", or "CPU" if this is a CPU build
```

If it says CPU on a CUDA machine, the artifact probe declined. It requires both
a driver and a CUDA runtime:

```bash
ldconfig -p | grep -E 'libcuda\.so\.1|libcudart\.so\.(12|13)'
```

A missing `libcuda.so.1` means no driver, and a CUDA build could not have been
loaded anyway. A missing `libcudart.so.N` means no CUDA runtime for a published
major. Name the variant directly if your layout defeats the probe:

```bash
LLAMA_CUDA_VARIANT=cu13 mix deps.compile llama_cpp_ex --force
```

To build from source instead:

```bash
LLAMA_BACKEND=cuda mix compile
```

That fails loudly when no toolkit is found. `nvcc` does **not** need to be on
`PATH` — `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda` and
`/usr/local/cuda-*` are all checked — but if the toolkit lives somewhere else:

```bash
CUDA_HOME=/opt/nvidia/cuda-13.0 LLAMA_BACKEND=cuda mix compile
```

### `undefined symbol` when the NIF loads

A CUDA build that compiles and links but dies at load is a missing library on
the link line, not a broken toolkit. Two have bitten this project:
`cuMemCreate` (the driver API, fixed by linking `-lcuda`) and `ncclAllReduce`
(ggml enabling NCCL behind cmake's back, fixed by stating `-DGGML_CUDA_NCCL`
explicitly). Reproduce the diagnosis with:

```bash
ldd -r _build/dev/lib/llama_cpp_ex/priv/llama_cpp_ex_nif.so | grep undefined
```

Ignore `enif_*`: those are the NIF API and the BEAM resolves them when it loads
the library. Anything else unresolved is a real missing dependency. The
`cuda-link` job in `.github/workflows/ci.yml` runs exactly this check on every
pull request for both CUDA majors.

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
