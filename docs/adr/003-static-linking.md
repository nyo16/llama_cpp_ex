# ADR 003: Static Linking of llama.cpp

## Status

Accepted

## Context

When building a NIF that depends on external C/C++ libraries, there are two linking strategies:

1. **Dynamic linking** — the NIF `.so` references shared libraries (`.so`/`.dylib`) at runtime
2. **Static linking** — all library code is embedded in the NIF `.so` at build time

## Decision

We chose **static linking** — all llama.cpp libraries are statically linked into a single `priv/llama_cpp_ex_nif.so`.

## Rationale

### Deployment simplicity

With static linking, the NIF is a single self-contained file. No need to:
- Set `RPATH` or `LD_LIBRARY_PATH`
- Bundle additional `.so` files in `priv/`
- Handle library path differences across distros
- Worry about ABI compatibility with system-installed libraries

### Follows Elixir ML conventions

EXLA and Evision both statically link their C++ dependencies. This is the established pattern for distributing compiled Elixir packages.

### Enables precompiled binaries

Static linking makes `cc_precompiler` distribution straightforward — one binary per platform/architecture combination, with no external library dependencies.

### Build implementation

The Makefile uses CMake to build llama.cpp with `BUILD_SHARED_LIBS=OFF` (the default), producing:
- `libllama.a`
- `libggml.a`
- `libggml-base.a`
- Backend-specific: `libggml-metal.a`, `libggml-cuda.a`, etc.

At link time, all `.a` files are linked into the NIF. On Linux, `--start-group` / `--end-group` handles circular dependencies between the static libraries.

## Consequences

- Larger NIF binary (~10-50MB depending on backend and quantization support compiled in)
- Rebuilding llama.cpp is required when updating the submodule (no system library sharing)
- CUDA runtime is still dynamically linked (static CUDA linking is impractical and license-restricted)
- Metal framework is dynamically linked (system framework on macOS)
- Build time is longer on first compile (CMake builds entire llama.cpp), but subsequent builds are incremental
