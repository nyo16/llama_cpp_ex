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
        H[Embedding]
        I[Server]
    end

    subgraph "Internal"
        G[NIF]
    end

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> H
    A --> I
    B --> G
    C --> G
    D --> G
    E --> G
    F --> G
    H --> G
    I --> G
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
| `prefill` | DirtyCPU | Prompt processing forward pass |
| `embed_decode` | DirtyCPU | Embedding forward pass |
| `get_embeddings` | Normal | Read embedding vectors |
| `batch_eval` | DirtyCPU | Batched forward pass (continuous batching) |
| `sampler_sample_at` | Normal | Sample at specific batch index |
| `decode_token` | DirtyCPU | Single-token forward pass |
| `decode_batch` | DirtyCPU | Multi-sequence decode + sample |

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

## Continuous Batching

The Server uses continuous batching to serve multiple concurrent users with a single forward pass per tick:

```mermaid
graph TD
    subgraph "Elixir Layer"
        A[Caller 1] --> D[LlamaCppEx.Server<br/>Tick-driven GenServer]
        B[Caller 2] --> D
        C[Caller 3] --> D
        D -->|Queue| Q[Request Queue<br/>FIFO :queue]
    end

    D -->|"One tick = one forward pass"| E[batch_eval NIF]

    subgraph "C++ Layer"
        E --> F[Build llama_batch<br/>decode tokens + prefill chunks]
        F --> G[Single llama_decode call]
    end

    G --> H[sampler_sample_at per slot]
    H --> I[Stream/reply to callers]
```

### Tick Loop

Each tick executes five phases:

1. **Finish** — Complete slots that hit EOG or max tokens, dequeue waiting requests
2. **Build batch** — Add decode tokens first (priority), then fill remaining budget with prefill chunks
3. **Forward pass** — Single `batch_eval` NIF call
4. **Sample** — `sampler_sample_at` for each generating/completing slot at their batch index
5. **Continue** — Schedule next tick if any active slots remain

### Chunked Prefill

Long prompts are split into chunks (default 512 tokens) and processed across multiple ticks, interleaved with decode tokens from generating slots:

```mermaid
sequenceDiagram
    participant S as Server
    participant NIF as batch_eval NIF

    Note over S: Tick 1: Slot 0 generating, Slot 1 prefilling (2048 tok prompt)
    S->>NIF: batch_eval([slot0_decode_tok, slot1_prefill_chunk_0..511])
    NIF-->>S: :ok
    Note over S: Sample slot 0, advance slot 1 prefill_pos

    Note over S: Tick 2: Slot 0 generating, Slot 1 still prefilling
    S->>NIF: batch_eval([slot0_decode_tok, slot1_prefill_chunk_512..1023])
    NIF-->>S: :ok

    Note over S: Tick 3-4: Continue chunking...

    Note over S: Tick 5: Slot 1 prefill complete (last chunk has logits=true)
    S->>NIF: batch_eval([slot0_decode_tok, slot1_prefill_chunk_1536..2047])
    NIF-->>S: :ok
    Note over S: Sample both slots — slot 1 now generating
```

### Why Batching Matters

- **Prefill** (prompt processing): Already GPU-efficient, compute-bound
- **Decode** (token generation): Memory-bandwidth-bound, GPU utilization 10-30%
- **Batching**: Converts N serial matrix-vector ops into one matrix-matrix multiply

## File Map

```
llama_cpp_ex/
├── mix.exs                          # Project config, deps, Hex package metadata
├── Makefile                         # CMake + NIF build system
├── vendor/llama.cpp/                # Git submodule (pinned to release)
├── c_src/llama_cpp_ex/
│   ├── llama_nif.h                  # RAII wrappers (LlamaModel, LlamaContext, LlamaSampler)
│   └── llama_nif.cpp                # All NIF implementations (~900 lines)
├── lib/
│   ├── llama_cpp_ex.ex              # High-level API: generate, stream, chat, embed
│   └── llama_cpp_ex/
│       ├── nif.ex                   # @on_load + NIF stubs
│       ├── model.ex                 # Model loading + introspection
│       ├── context.ex               # Inference context with KV cache
│       ├── sampler.ex               # Sampling chain configuration
│       ├── tokenizer.ex             # Text <-> token conversion
│       ├── chat.ex                  # Chat template formatting
│       ├── embedding.ex             # Text embeddings (L2 norm, batched)
│       └── server.ex                # Continuous batching GenServer
├── priv/                            # Build output (.so / .dylib)
├── docs/                            # Architecture docs + ADRs
└── test/
    └── llama_cpp_ex_test.exs        # 46 tests (2 unit + 44 model-dependent)
```
