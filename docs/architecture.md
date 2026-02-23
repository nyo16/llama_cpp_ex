# Architecture

## Overview

LlamaCppEx provides Elixir bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) via C++ NIFs (Native Implemented Functions). The design follows the same pattern used by production Elixir ML libraries like EXLA and Evision.

## Layer Diagram

```mermaid
graph TD
    A[Elixir API<br/>LlamaCppEx] --> B[NIF Stubs<br/>LlamaCppEx.NIF]
    B --> C[C++ NIF Layer<br/>c_src/llama_nif.cpp]
    C --> D[fine.hpp<br/>Type Encoding + RAII]
    C --> E[llama.cpp Static Libs<br/>libllama.a + libggml.a]
    E --> F[Hardware Backend]
    F --> G[Metal<br/>macOS GPU]
    F --> H[CUDA<br/>NVIDIA GPU]
    F --> I[Vulkan<br/>Cross-platform GPU]
    F --> J[CPU<br/>Fallback]
```

## Module Structure

```mermaid
graph LR
    subgraph "High-Level API"
        A[LlamaCppEx]
    end

    subgraph "Core Modules"
        B[Model]
        C[Context]
        D[Sampler]
        E[Tokenizer]
        F[Chat]
    end

    subgraph "Internal"
        G[NIF]
    end

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    B --> G
    C --> G
    D --> G
    E --> G
    F --> G
```

## Resource Lifecycle

All C++ objects are wrapped in RAII classes registered with the BEAM via `fine`. When the Elixir process holding a reference is garbage collected, the C++ destructor runs automatically.

```mermaid
sequenceDiagram
    participant Elixir as Elixir Process
    participant NIF as C++ NIF
    participant BEAM as BEAM GC

    Elixir->>NIF: Model.load("model.gguf")
    NIF->>NIF: llama_model_load_from_file()
    NIF->>NIF: Wrap in ResourcePtr<LlamaModel>
    NIF-->>Elixir: {:ok, %Model{ref: resource}}

    Elixir->>NIF: Context.create(model, opts)
    NIF->>NIF: llama_init_from_model()
    NIF->>NIF: Wrap in ResourcePtr<LlamaContext>
    Note over NIF: Context holds ResourcePtr<LlamaModel><br/>preventing model GC
    NIF-->>Elixir: {:ok, %Context{ref: resource}}

    Note over Elixir: Context goes out of scope
    BEAM->>NIF: ~LlamaContext()
    NIF->>NIF: llama_free(ctx)
    Note over NIF: Model ref count drops

    Note over Elixir: Model goes out of scope
    BEAM->>NIF: ~LlamaModel()
    NIF->>NIF: llama_model_free(model)
```

## Resource Types

| C++ Wrapper | Wraps | Destructor | Prevents GC of |
|---|---|---|---|
| `LlamaModel` | `llama_model*` | `llama_model_free()` | - |
| `LlamaContext` | `llama_context*` | `llama_free()` | LlamaModel |
| `LlamaSampler` | `llama_sampler*` | `llama_sampler_free()` | - |

The Context holds a `ResourcePtr<LlamaModel>` to prevent the model from being garbage collected while the context is alive. This is critical since `llama_context` internally references the model's weights.

## NIF Scheduler Assignment

NIFs are assigned to the appropriate scheduler based on their execution characteristics:

| NIF | Scheduler | Reason |
|---|---|---|
| `model_load` | DirtyIO | Reads multi-GB file from disk |
| `context_create` | DirtyCPU | GPU memory allocation |
| `decode` | DirtyCPU | Forward pass (compute-heavy) |
| `generate` | DirtyCPU | Tight decode+sample loop |
| `generate_tokens` | DirtyCPU | Streaming decode+sample loop |
| `tokenize`, `detokenize` | Normal | Fast string operations |
| `sampler_*` | Normal | Lightweight operations |
| `model_*` (introspection) | Normal | Simple field reads |

**Why dirty schedulers?** Regular NIF calls must return within ~1ms to avoid blocking BEAM schedulers. Model loading and inference can take seconds to minutes. Dirty schedulers provide dedicated OS threads for these long-running operations without impacting BEAM responsiveness.

## Text Generation Flow

```mermaid
sequenceDiagram
    participant User as Elixir Caller
    participant API as LlamaCppEx
    participant Tok as Tokenizer
    participant Ctx as Context
    participant Sam as Sampler
    participant NIF as C++ NIF

    User->>API: generate(model, "Hello", max_tokens: 100)
    API->>Tok: encode(model, "Hello")
    Tok->>NIF: tokenize(vocab, "Hello")
    NIF-->>Tok: [token_ids]
    Tok-->>API: {:ok, [15496]}

    API->>Ctx: create(model, n_ctx: 2048)
    Ctx->>NIF: context_create(model, params)
    NIF-->>Ctx: {:ok, ctx_ref}

    API->>Sam: create(temp: 0.8)
    Sam->>NIF: sampler_init(params)
    NIF-->>Sam: {:ok, sampler_ref}

    API->>Ctx: generate(ctx, sampler, tokens, max_tokens: 100)
    Ctx->>NIF: generate(ctx, sampler, tokens, 100)

    loop For each token (on DirtyCPU scheduler)
        NIF->>NIF: llama_decode(batch)
        NIF->>NIF: llama_sampler_sample(sampler, ctx, -1)
        NIF->>NIF: llama_sampler_accept(sampler, token)
        Note over NIF: Check for EOG token
    end

    NIF->>NIF: Detokenize all generated tokens
    NIF-->>Ctx: {:ok, "world, how are you?"}
    Ctx-->>API: {:ok, "world, how are you?"}
    API-->>User: {:ok, "world, how are you?"}
```

## Streaming Flow

Streaming uses `enif_send` to send tokens from the dirty scheduler to the calling Elixir process:

```mermaid
sequenceDiagram
    participant User as Elixir Caller
    participant Stream as Stream.resource/3
    participant Gen as Generator (spawn_link)
    participant NIF as C++ NIF (DirtyCPU)

    User->>Stream: LlamaCppEx.stream(model, prompt)

    Stream->>Stream: Tokenize, create ctx + sampler
    Stream->>Gen: spawn_link(generate_tokens NIF)

    loop Token generation
        NIF->>NIF: llama_decode + llama_sampler_sample
        NIF-->>Stream: enif_send {ref, {:token, id, "text"}}
        Stream-->>User: "text" (via Enum.each)
    end

    alt End of generation
        NIF-->>Stream: enif_send {ref, :eog}
    else Max tokens reached
        NIF-->>Stream: enif_send {ref, :done}
    end

    Stream->>Gen: Process.exit(:kill)
    Stream->>Stream: Flush remaining messages
```

Key design decisions:
- Generator runs in a `spawn_link`ed process on a dirty scheduler
- Messages use a unique `ref` to prevent cross-stream interference
- `Stream.resource/3` provides lazy enumeration with proper cleanup
- Early termination (e.g., `Enum.take/2`) kills the generator and flushes messages

## Build System

```mermaid
graph TD
    A[mix compile] --> B[elixir_make]
    B --> C[Makefile]
    C --> D{Backend Detection}
    D -->|LLAMA_BACKEND=metal| E[CMake -DGGML_METAL=ON]
    D -->|LLAMA_BACKEND=cuda| F[CMake -DGGML_CUDA=ON]
    D -->|LLAMA_BACKEND=vulkan| G[CMake -DGGML_VULKAN=ON]
    D -->|LLAMA_BACKEND=cpu| H[CMake - CPU only]
    D -->|Auto-detect| I{Platform?}
    I -->|macOS| E
    I -->|nvcc found| F
    I -->|Otherwise| H

    E --> J[Build llama.cpp static libs]
    F --> J
    G --> J
    H --> J

    J --> K[Compile llama_nif.cpp]
    K --> L[Link into priv/llama_cpp_ex_nif.so]

    subgraph "Static Libraries"
        J --> M[libllama.a]
        J --> N[libggml.a]
        J --> O[libggml-base.a]
        J --> P[libggml-metal.a / libggml-cuda.a / ...]
    end
```

## Batching Architecture (Planned - Phase 3)

For serving multiple concurrent users, llama.cpp supports batched inference where multiple sequences share a single forward pass:

```mermaid
graph TD
    subgraph "Elixir Layer"
        A[Caller 1] --> D[LlamaCppEx.Server<br/>GenServer Batcher]
        B[Caller 2] --> D
        C[Caller 3] --> D
    end

    D -->|Flush on batch_size<br/>or batch_timeout| E[decode_batch NIF]

    subgraph "C++ Layer"
        E --> F[Build llama_batch<br/>with all sequences]
        F --> G[Single llama_decode call]
        G --> H[Sample per sequence]
    end

    H --> I[Reply to Caller 1]
    H --> J[Reply to Caller 2]
    H --> K[Reply to Caller 3]
```

### Why Batching Matters

- **Prefill** (prompt processing): Already GPU-efficient, compute-bound
- **Decode** (token generation): Memory-bandwidth-bound, GPU utilization 10-30%
- **Batching**: Converts N serial matrix-vector ops into one matrix-matrix multiply

### How `llama_batch` Enables Multi-Sequence Batching

Each token in a batch carries its sequence ID. Tokens only attend to tokens with the same sequence ID, enabling independent sequences to share a forward pass:

```
llama_batch:
  token[0] = {id: 1234, pos: 42, seq_id: [0], logits: true}   # Sequence 0
  token[1] = {id: 5678, pos: 18, seq_id: [1], logits: true}   # Sequence 1
  token[2] = {id: 9012, pos: 31, seq_id: [2], logits: true}   # Sequence 2
```

### Shared System Prompt

Tag prompt tokens with ALL sequence IDs to cache the system prompt once and share it across all sequences:

```mermaid
graph LR
    subgraph "KV Cache"
        A["System prompt tokens<br/>seq_id: [0,1,2,3]"] --> B["Seq 0 tokens"]
        A --> C["Seq 1 tokens"]
        A --> D["Seq 2 tokens"]
        A --> E["Seq 3 tokens"]
    end
```

### GenServer Batcher Design

```mermaid
sequenceDiagram
    participant C1 as Caller 1
    participant C2 as Caller 2
    participant S as Server (GenServer)
    participant NIF as decode_batch NIF

    C1->>S: GenServer.call({:generate, ...})
    Note over S: Start batch_timeout timer (20ms)
    C2->>S: GenServer.call({:generate, ...})

    Note over S: Timer fires or batch_size reached
    S->>NIF: decode_batch([{seq0, token, pos}, {seq1, token, pos}])
    NIF-->>S: [{seq0, next_token}, {seq1, next_token}]
    S-->>C1: GenServer.reply(from1, next_token)
    S-->>C2: GenServer.reply(from2, next_token)
```

## File Map

```
llama_cpp_ex/
├── mix.exs                          # Project config, deps, Hex package metadata
├── Makefile                         # CMake + NIF build system
├── vendor/llama.cpp/                # Git submodule (pinned to release)
├── c_src/llama_cpp_ex/
│   ├── llama_nif.h                  # RAII wrappers (LlamaModel, LlamaContext, LlamaSampler)
│   └── llama_nif.cpp                # All NIF implementations (~560 lines)
├── lib/
│   ├── llama_cpp_ex.ex              # High-level API: generate, stream, chat
│   └── llama_cpp_ex/
│       ├── nif.ex                   # @on_load + NIF stubs
│       ├── model.ex                 # Model loading + introspection
│       ├── context.ex               # Inference context with KV cache
│       ├── sampler.ex               # Sampling chain configuration
│       ├── tokenizer.ex             # Text <-> token conversion
│       └── chat.ex                  # Chat template formatting
├── priv/                            # Build output (.so / .dylib)
├── docs/                            # Architecture docs + ADRs
└── test/
    └── llama_cpp_ex_test.exs        # 11 tests (2 unit + 9 model-dependent)
```
