# ADR 002: fine for NIF Ergonomics

## Status

Accepted

## Context

Writing raw `erl_nif.h` NIFs in C++ involves significant boilerplate: manual `enif_get_*` / `enif_make_*` calls for every argument and return value, manual resource type registration, manual NIF function table management.

We evaluated two approaches:
1. **Raw erl_nif.h** — manual everything
2. **fine** (by the Nx/Elixir team) — declarative macros for NIF registration, automatic type encoding/decoding, RAII resource management

## Decision

We chose **fine** (v0.1.4).

## Rationale

`fine` provides three key macros that dramatically reduce boilerplate:

### `FINE_NIF(name, flags)`

Declares a NIF with automatic argument decoding and return value encoding:

```cpp
// Without fine: ~30 lines of enif_get_*/enif_make_* boilerplate
// With fine:
std::tuple<bool, std::string>
tokenize(ErlNifEnv* env, fine::ResourcePtr<LlamaModel> model, std::string text, bool add_special) {
    // Just use C++ types directly — fine handles encoding/decoding
}
FINE_NIF(tokenize, 0);
```

### `FINE_RESOURCE(Type)`

Registers a C++ class as an Erlang resource type. The destructor runs automatically when the BEAM garbage collects the reference:

```cpp
FINE_RESOURCE(LlamaModel);   // ~LlamaModel() called by GC
FINE_RESOURCE(LlamaContext);  // ~LlamaContext() called by GC
FINE_RESOURCE(LlamaSampler);  // ~LlamaSampler() called by GC
```

### `FINE_INIT("Module.Name")`

Generates the `ErlNifEntry` and `nif_init` function automatically from all declared NIFs and resources.

### Type Support

fine automatically handles encoding/decoding for:
- Primitives: `int`, `int64_t`, `uint32_t`, `double`, `bool`, `std::string`
- Containers: `std::vector<T>`, `std::tuple<T...>`, `std::optional<T>`
- Atoms: `fine::Ok`, `fine::Error`, `fine::Atom`
- Resources: `fine::ResourcePtr<T>`
- Erlang terms: `ERL_NIF_TERM` (passthrough)

## Consequences

- Depends on `fine` hex package (maintained by the Nx team — low risk)
- `fine` v0.1.4 has some limitations: no `std::pair` decoding (we use `std::tuple<A,B>` instead), no `std::map` support
- RAII via `ResourcePtr` prevents use-after-free and ensures cleanup — this is the primary safety mechanism
- When `fine` doesn't support a type, we fall back to raw `ERL_NIF_TERM` and manual encoding
