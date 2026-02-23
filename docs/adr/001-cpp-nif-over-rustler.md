# ADR 001: C++ NIF Over Rustler

## Status

Accepted

## Context

We needed to choose a binding strategy for connecting Elixir to llama.cpp. The main candidates were:

1. **C++ NIF** with `elixir_make` and `fine`
2. **Rustler** (Rust NIF) via a Rust wrapper crate
3. **Port** (separate OS process communicating via stdio)
4. **Zigler** (Zig-based NIFs)

## Decision

We chose **C++ NIF with `fine` and `elixir_make`**.

## Rationale

### Against Rustler

- The Rust crates wrapping llama.cpp (`llama-cpp-2`, `llama_cpp_rs`) are alpha-quality and lag behind upstream releases
- Double indirection (Elixir → Rust → C++) adds maintenance burden — when llama.cpp changes its API, we'd need to wait for the Rust crate to update
- Requires Rust + Cargo + CMake toolchain vs just C++ + CMake
- Existing Rustler-based Elixir bindings (`ex_llama`, `llama_cpp` on Hex) are dormant/minimal

### Against Port

- No shared model state across Elixir processes — each Port would need its own model loaded
- Serialization overhead for embeddings and large token arrays
- Process lifecycle management complexity
- Cannot share KV cache between requests

### Against Zigler

- Less mature ecosystem
- Fewer production examples in the Elixir ML space
- Zig's C interop is excellent but the tooling is less proven for large C++ libraries

### For C++ NIF + fine

- **Proven pattern**: EXLA, Evision, and TorchX all use C++ NIFs with `elixir_make` — this is the standard for production Elixir ML
- **`fine`** (by the Nx team) provides Rustler-like ergonomics: auto type encoding/decoding, `ResourcePtr<T>` for GC-managed C++ objects, declarative NIF registration
- **Direct API access**: Call `llama.h` C API directly with no intermediate language layer
- **Simple toolchain**: C++17 compiler + CMake (already needed for llama.cpp itself)
- **Static linking**: Produce a self-contained `.so` with no runtime dependencies

## Consequences

- We must handle `erl_nif.h` concerns (dirty schedulers, `enif_send` for streaming, proper resource lifecycle)
- Build errors are C++ compiler errors, which can be harder to debug than Rust compiler errors
- No memory safety guarantees beyond our RAII wrappers — we must be careful with pointer lifetimes
- We benefit from `fine`'s type system for encoding/decoding Elixir terms to/from C++ types
