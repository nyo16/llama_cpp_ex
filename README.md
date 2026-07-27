# LlamaCppEx

[![Precompile NIFs](https://github.com/nyo16/llama_cpp_ex/actions/workflows/precompile.yml/badge.svg)](https://github.com/nyo16/llama_cpp_ex/actions/workflows/precompile.yml)
[![CI](https://github.com/nyo16/llama_cpp_ex/actions/workflows/ci.yml/badge.svg)](https://github.com/nyo16/llama_cpp_ex/actions/workflows/ci.yml)

Elixir bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) — run LLMs locally with Metal, CUDA, Vulkan, or CPU acceleration.

Built with C++ NIFs using [fine](https://github.com/elixir-nx/fine) for ergonomic resource management and [elixir_make](https://hex.pm/packages/elixir_make) for the build system.

## Features

- Load and run GGUF models directly from Elixir
- **HuggingFace Hub integration** — search, list, and download GGUF models
- GPU acceleration: Metal (macOS), CUDA (NVIDIA), Vulkan, or CPU
- Streaming token generation via lazy `Stream`
- Jinja chat templates with `enable_thinking` support (Qwen3, Qwen3.5, etc.)
- RAII resource management — models, contexts, and samplers are garbage collected by the BEAM
- Configurable sampling: temperature, top-k, top-p, min-p, repetition penalty, frequency & presence penalty
- Embedding generation with L2 normalization
- Grammar-constrained generation (GBNF)
- Structured output via JSON Schema (auto-converted to GBNF grammar)
- Optional Ecto schema to JSON Schema conversion
- Continuous batching server for concurrent inference
- **Multi-model manager** — keep several models resident, route requests by id, with a placement-aware (per-GPU VRAM) memory budget
- **Device introspection** — `LlamaCppEx.devices/0` lists GPUs/accelerators with per-device VRAM
- **Multi-Token Prediction (MTP) speculative decoding** — ~2x token-generation speedup on Qwen 3.6 with live acceptance-rate stats
- **Prefix caching** — cross-slot KV reuse with session affinity and an optional RAM prompt cache (TTFT 115 → 35 ms at ~75% hit ratio under concurrency)
- **Pluggable batching strategies** — DecodeMaximal, PrefillPriority, Balanced
- **Pre-tokenized API** — tokenize outside the GenServer for lower contention
- **Request lifecycle controls** — per-request sampling params, `:max_queue` backpressure, and cancellation (dead or halted stream consumers free their slot immediately)
- Telemetry integration for observability

## Installation

Add `llama_cpp_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:llama_cpp_ex, "~> 0.7.5"}
  ]
end
```

### Prerequisites

Precompiled NIFs are published for `aarch64-apple-darwin` (Metal) and
`x86_64-linux-gnu` (CPU) at NIF versions 2.17 and 2.18 — that is **Erlang/OTP 26
or newer** (OTP 26, 27 and 28 report NIF 2.17; OTP 29 reports 2.18). On those
platforms `mix deps.get` and `mix compile` download a binary and none of the
build tooling below is needed.

Everything else builds from source: OTP 25 (NIF 2.16), other architectures,
musl, Windows, and every GPU backend except Metal. A source build needs

- a C++17 compiler (GCC, Clang, or MSVC),
- CMake 3.14+,
- Git — the Hex package ships `.gitmodules` rather than the llama.cpp tree, so
  the Makefile clones the pinned commit on demand.

### Backend Selection

What the published artifacts actually contain is Metal on Apple Silicon and
plain CPU on Linux. **There is no CUDA or Vulkan artifact**, and a downloaded
artifact never runs the Makefile, so nothing is auto-detected at install time.
GPU acceleration beyond Metal always means an explicit source build:

```bash
mix compile                        # Precompiled artifact when one matches this
                                   # OS/arch/NIF version, else a source build
LLAMA_BACKEND=metal mix compile    # Apple Silicon GPU
LLAMA_BACKEND=cuda mix compile     # NVIDIA GPU, needs the CUDA toolkit
LLAMA_BACKEND=vulkan mix compile   # Vulkan, needs the Vulkan SDK
LLAMA_BACKEND=cpu mix compile      # CPU only
```

Setting `LLAMA_BACKEND` to anything forces a source build and bypasses the
precompiled artifact — that is how a CUDA or Vulkan build is obtained. When a
source build runs with `LLAMA_BACKEND` unset it picks Metal on macOS, CUDA if
`nvcc` is on `PATH`, and CPU otherwise.

Power users can pass arbitrary CMake flags:

```bash
LLAMA_CMAKE_ARGS="-DGGML_CUDA_FORCE_CUBLAS=ON" mix compile
```

Two more build variables:

- `LLAMA_PORTABLE=1` drops `-march=native`. ggml turns it on by default, which
  tunes the binary to the exact CPU it was built on; the release workflow sets
  this so published artifacts run on every machine of that architecture. Leave
  it unset locally, where the native flags are free performance.
- `LLAMA_COMMIT=<sha>` overrides the pinned llama.cpp commit used when
  `vendor/llama.cpp` has to be cloned.

## Quick Start

```elixir
# Initialize the backend (once per application)
:ok = LlamaCppEx.init()

# Load a GGUF model (use n_gpu_layers: -1 to offload all layers to GPU)
{:ok, model} = LlamaCppEx.load_model("path/to/model.gguf", n_gpu_layers: -1)

# Generate text
{:ok, text} = LlamaCppEx.generate(model, "Once upon a time", max_tokens: 200, temp: 0.8)

# Stream tokens
model
|> LlamaCppEx.stream("Tell me a story", max_tokens: 500)
|> Enum.each(&IO.write/1)

# Chat with template
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "system", content: "You are a helpful assistant."},
  %{role: "user", content: "What is Elixir?"}
], max_tokens: 200)

# Chat with thinking disabled (Qwen3/3.5 and similar models)
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "What is 2+2?"}
], max_tokens: 64, enable_thinking: false)

# Stream a chat response
model
|> LlamaCppEx.stream_chat([
  %{role: "user", content: "Explain pattern matching in Elixir."}
], max_tokens: 500)
|> Enum.each(&IO.write/1)
```

## HuggingFace Hub

Download GGUF models directly from HuggingFace Hub. Requires the optional `:req` dependency:

```elixir
{:req, "~> 0.5 or ~> 0.6"}
```

```elixir
# Search for GGUF models
{:ok, models} = LlamaCppEx.Hub.search("qwen3 gguf", limit: 5)

# List GGUF files in a repository
{:ok, files} = LlamaCppEx.Hub.list_gguf_files("Qwen/Qwen3-0.6B-GGUF")

# Download (cached locally in ~/.cache/llama_cpp_ex/models/)
{:ok, path} = LlamaCppEx.Hub.download("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")

# Or download + load in one step
{:ok, model} = LlamaCppEx.load_model_from_hub(
  "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf",
  n_gpu_layers: -1
)

# Private or gated repo — pass a HuggingFace token explicitly
{:ok, model} = LlamaCppEx.load_model_from_hub(
  "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf",
  token: "hf_xxx",
  n_gpu_layers: -1
)
```

For private/gated models, set `HF_TOKEN` or pass `token: "hf_..."`. Set `LLAMA_OFFLINE=1` for offline-only cached access.

## Structured Output (JSON Schema)

Constrain model output to valid JSON matching a schema. Pass `:json_schema` to any generate or chat function — the schema is automatically converted to a GBNF grammar via llama.cpp's built-in converter.

```elixir
schema = %{
  "type" => "object",
  "properties" => %{
    "name" => %{"type" => "string"},
    "age" => %{"type" => "integer"},
    "hobbies" => %{"type" => "array", "items" => %{"type" => "string"}}
  },
  "required" => ["name", "age", "hobbies"],
  "additionalProperties" => false
}

# Works with generate
{:ok, json} = LlamaCppEx.generate(model, "Generate a person:",
  json_schema: schema, temp: 0.0)
# => "{\"name\": \"Alice\", \"age\": 30, \"hobbies\": [\"reading\", \"hiking\"]}"

# Works with chat
{:ok, json} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Generate a person named Bob who is 25."}
], json_schema: schema, temp: 0.0)

# Works with streaming
model
|> LlamaCppEx.stream("Generate a person:", json_schema: schema, temp: 0.0)
|> Enum.each(&IO.write/1)

# Works with chat completions
{:ok, completion} = LlamaCppEx.chat_completion(model, [
  %{role: "user", content: "Generate a person."}
], json_schema: schema, temp: 0.0)
```

> **Tip:** Set `"additionalProperties" => false` in your schema to produce a tighter grammar
> that avoids potential issues with the grammar sampler.

### Manual Grammar Conversion

You can also convert the schema to GBNF manually for more control:

```elixir
{:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(schema)
IO.puts(gbnf)
# root ::= "{" space name-kv "," space age-kv "," space hobbies-kv "}" space
# ...

# Use the grammar directly
{:ok, json} = LlamaCppEx.generate(model, "Generate a person:", grammar: gbnf, temp: 0.0)
```

### Ecto Schema Integration

Convert Ecto schema modules to JSON Schema automatically (requires `{:ecto, "~> 3.0"}` — optional dependency):

```elixir
defmodule MyApp.Person do
  use Ecto.Schema

  embedded_schema do
    field :name, :string
    field :age, :integer
    field :active, :boolean
    field :tags, {:array, :string}
  end
end

# Ecto schema -> JSON Schema -> constrained generation
schema = LlamaCppEx.Schema.to_json_schema(MyApp.Person)
# => %{"type" => "object", "properties" => %{"name" => %{"type" => "string"}, ...}, ...}

{:ok, json} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Generate a person."}
], json_schema: schema, temp: 0.0)
```

Supported Ecto types: `:string`, `:integer`, `:float`, `:decimal`, `:boolean`, `:map`, `{:array, inner}`, `:date`, `:utc_datetime`, `:naive_datetime`, and embedded schemas (`embeds_one`/`embeds_many`). Fields `:id`, `:inserted_at`, and `:updated_at` are excluded automatically.

## Lower-level API

For fine-grained control over the inference pipeline:

```elixir
# Tokenize
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Hello world")
{:ok, text} = LlamaCppEx.Tokenizer.decode(model, tokens)

# Create context and sampler separately
{:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 4096)
{:ok, sampler} = LlamaCppEx.Sampler.create(model, temp: 0.7, top_p: 0.9)

# Run generation with your own context
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "The answer is")
{:ok, text} = LlamaCppEx.Context.generate(ctx, sampler, tokens, max_tokens: 100)

# Model introspection
LlamaCppEx.Model.desc(model)          # "llama 7B Q4_K - Medium"
LlamaCppEx.Model.n_params(model)      # 6_738_415_616
LlamaCppEx.Model.chat_template(model) # "<|im_start|>..."
LlamaCppEx.Tokenizer.vocab_size(model) # 32000
```

## Server (Continuous Batching)

For concurrent inference, `LlamaCppEx.Server` manages a shared model/context with a slot pool and continuous batching:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_gpu_layers: -1,
  n_parallel: 4,
  n_ctx: 8192
)

# Synchronous
{:ok, text} = LlamaCppEx.Server.generate(server, "Once upon a time", max_tokens: 100)

# Streaming
LlamaCppEx.Server.stream(server, "Tell me a story", max_tokens: 200)
|> Enum.each(&IO.write/1)
```

Multiple callers are batched into a single forward pass per tick, improving throughput under load.

### Prefix Caching

The server caches KV state between requests (on by default) and shares it **across slots**: with unified KV (`kv_unified: true`, the default) a system prompt prefilled by any slot is adopted by every other slot via a metadata-only copy, so it is computed once, ever. Requests carrying a `:session` term stick to their slot under concurrency, keeping conversations on their cached prefix:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 4,
  cache_prompt: true,        # default: true; also overridable per request
  prompt_cache_ram_mb: 1024  # optional level-2 RAM cache for evicted prefixes (default: 0 = off)
)

{:ok, text} = LlamaCppEx.Server.generate(server, prompt, session: "conversation-42")
```

Benchmark (8 interleaved conversations × 4 turns on 4 slots, shared system prompt): **TTFT median 115 → 35.5 ms (3.2x)** at a ~75% prefix-cache hit ratio.

Notes:

- Hybrid GDN models (e.g. Qwen 3.5/3.6) only hit on exact-prefix continuations; dense-attention models additionally get partial-prefix and cross-slot hits.
- If a chat template rewrites history (e.g. stripping thinking blocks), cache hits silently degrade — the server emits `[:llama_cpp_ex, :server, :prefix_instability]` telemetry when it detects this.

### Chat Completions via the Server

`chat_completion/3` and `stream_chat_completion/3` accept a running server in place of a `%Model{}` — templating and tokenization happen in the caller, and the multi-turn prompt benefits from the prefix cache (**1.6x faster** than the stateless path over a 4-turn conversation):

```elixir
{:ok, completion} =
  LlamaCppEx.chat_completion(server, messages, max_tokens: 200, session: "conversation-42")
```

### Backpressure & Cancellation

- `max_queue: n` bounds the request queue; overflow returns `{:error, :queue_full}` immediately and streams emit a single `{:error, :queue_full}` element (default: `0`, unlimited).
- Halting a stream early (`Enum.take/2`, consumer exit) cancels generation and frees the slot right away instead of decoding to `max_tokens`; `LlamaCppEx.Server.cancel/2` is also available explicitly.
- Sampling options (`:temp`, `:seed`, `:grammar`, ...) can be set per request, overriding the server defaults.

### Batching Strategies

Choose how the token budget is split between generation and prompt processing:

```elixir
# Default: generation latency optimized
batch_strategy: LlamaCppEx.Server.Strategy.DecodeMaximal

# Throughput optimized (batch processing)
batch_strategy: LlamaCppEx.Server.Strategy.PrefillPriority

# Fair split (mixed workloads)
batch_strategy: LlamaCppEx.Server.Strategy.Balanced
```

### Pre-Tokenized API

Tokenize outside the GenServer to reduce contention under concurrent load:

```elixir
model = LlamaCppEx.Server.get_model(server)
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)
{:ok, text} = LlamaCppEx.Server.generate_tokens(server, tokens, max_tokens: 100)
```

### llama.cpp Optimizations

Pass llama.cpp optimization parameters directly:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 8,
  n_ctx: 32768,

  # KV cache quantization — 2x memory savings, identical output
  type_k: :q8_0,
  type_v: :q8_0,

  # Flash attention — faster prefill
  flash_attn: :enabled
)
```

These also work with the high-level API:

```elixir
{:ok, text} = LlamaCppEx.generate(model, "Hello",
  max_tokens: 256,
  type_k: :q8_0,
  type_v: :q8_0,
  flash_attn: :enabled
)
```

See [Performance Guide](docs/performance.md) for all available parameters including RoPE context extension, GPU offload control, attention type, and more.

## Multiple Models (ModelManager)

`LlamaCppEx.ModelManager` keeps several models resident at once and routes requests to them by id. It reuses the HuggingFace Hub downloader and the batching `Server`, and adds named load/unload, capability-based routing, and an advisory memory budget.

Add `LlamaCppEx.ModelSupervisor` to your application's supervision tree (it starts a `Registry`, a `DynamicSupervisor`, and the manager):

```elixir
children = [
  {LlamaCppEx.ModelSupervisor,
   memory_budget: :auto,
   models: [
     # Server-backed (batching + streaming), marked as the default route
     {"chat", {:hub, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"},
      n_gpu_layers: -1, default: true},
     # Embedding model — :embed capability auto-selects :direct mode
     {"embed", {:path, "/models/nomic-embed.gguf"}, capabilities: [:embed]}
   ]}
]
```

For scripts or IEx, start it directly and load at runtime:

```elixir
{:ok, _sup} = LlamaCppEx.ModelSupervisor.start_link([])

# Download from the Hub (cached in ~/.cache/llama_cpp_ex/models/) and keep resident
{:ok, "chat"} = LlamaCppEx.ModelManager.load(
  "chat", {:hub, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"}, n_gpu_layers: -1
)
# Or from a local path
{:ok, "embed"} = LlamaCppEx.ModelManager.load(
  "embed", {:path, "/models/nomic-embed.gguf"}, capabilities: [:embed]
)

# Route by id
{:ok, text} = LlamaCppEx.ModelManager.generate("chat", "Once upon a time", max_tokens: 100)
LlamaCppEx.ModelManager.stream("chat", "Tell me a story") |> Enum.each(&IO.write/1)
{:ok, reply} = LlamaCppEx.ModelManager.chat("chat", [%{role: "user", content: "Hi!"}])
{:ok, vector} = LlamaCppEx.ModelManager.embed("embed", "text to embed")

# Route to the default model
{:ok, text} = LlamaCppEx.ModelManager.generate(:default, "Hello")

# Inspect and manage
LlamaCppEx.ModelManager.list()        # sanitized views, no raw refs
LlamaCppEx.ModelManager.loaded?("chat")
LlamaCppEx.ModelManager.unload("chat")  # stops the backing server, frees memory
```

### Loading and concurrency

`ModelManager` is a **node-wide singleton** — run one `ModelSupervisor` per node. The client API targets the manager by module name, and the backing `Registry`/`DynamicSupervisor` use fixed names, so a second instance is refused at startup.

`load/3` blocks the *calling* process until the model is ready (returning `{:ok, id}` or `{:error, reason}`), but the slow work — the Hub download and the native model load — runs in a supervised `Task`, **not** on the manager process. So a long load never blocks other lifecycle calls: a concurrent `load/3`, an `unload/1`, a `set_default/1`, or reads like `list/0`/`info/1` all proceed while it runs. A model in flight shows `status: :loading`, and re-loading the same id returns `{:error, :already_loaded}`. The memory-budget check and the ETS commit are serialized on the manager, so resident models are always accounted for.

### Backing modes

- **`:server`** (default for generation/chat) — backs the model with a supervised `LlamaCppEx.Server`, so you get continuous batching, streaming, prefix caching, and telemetry.
- **`:direct`** (auto-selected when `:embed` is in `:capabilities`) — holds the model and runs stateless calls. Required for embeddings, since the server has no embedding path.

Override with `mode: :server | :direct`.

### GPU placement

All of llama.cpp's placement options pass straight through `load/3` (per model) to `Model.load/2`/`Server.start_link/1`:

| Option | Meaning |
|---|---|
| `:n_gpu_layers` | Layers to offload (`-1` = all, `0` = CPU only) |
| `:split_mode` | `:none` (single GPU), `:layer` (split layers across GPUs), `:row` (split tensor rows) |
| `:tensor_split` | A **list of per-device proportions** — one float per GPU, indexed by device order. Zeros exclude a device. |
| `:main_gpu` | Primary device: the single GPU under `:none`, or the device holding non-split tensors under `:layer` |

`:tensor_split` is the "array of GPUs": it's a weight per device (llama.cpp normalizes the values), **not** a list of indices. Device order follows `CUDA_VISIBLE_DEVICES`. See [docs/multi-gpu.md](docs/multi-gpu.md) for a full multi-GPU guide and verification steps.

```elixir
# Pin a model to one specific GPU
LlamaCppEx.ModelManager.load("a", {:path, m}, n_gpu_layers: -1, split_mode: :none, main_gpu: 5)

# Spread one big model across all 8 GPUs equally
LlamaCppEx.ModelManager.load("big", {:path, m},
  n_gpu_layers: -1, split_mode: :layer,
  tensor_split: [1, 1, 1, 1, 1, 1, 1, 1]
)

# Use only a subset — e.g. "big" on GPUs 0–3, "embed" on GPUs 4–7
LlamaCppEx.ModelManager.load("big", {:path, m1},
  n_gpu_layers: -1, split_mode: :layer,
  tensor_split: [1, 1, 1, 1, 0, 0, 0, 0]
)

LlamaCppEx.ModelManager.load("embed", {:path, m2},
  capabilities: [:embed], n_gpu_layers: -1, split_mode: :layer,
  tensor_split: [0, 0, 0, 0, 1, 1, 1, 1]
)
```

> On a multi-GPU box, `memory_budget: :auto` reads each card's free VRAM and tracks placement per device — `:tensor_split` and `:main_gpu` are accounted for (see Memory budget below).

### Memory budget

`:memory_budget` is **placement-aware** — it knows whether a model lands in RAM or on specific GPUs (from `:n_gpu_layers`/`:split_mode`/`:tensor_split`/`:main_gpu`) and checks each pool independently. It accepts:

- `:infinity` (default) — no limit.
- an **integer** — a single combined pool (RAM + all VRAM count against one number).
- `:auto` — RAM ≈ 80% of system memory, and **per-GPU VRAM from each card's free memory** (via `LlamaCppEx.devices/0`).
- a map `%{ram: …, vram: …}` — explicit per-device limits. `vram` is a list `[b0, b1, …]` indexed by GPU, or a map `%{gpu_index => bytes}`; `ram`/`vram` may be `:auto` or `:infinity`.

The manager estimates footprint from GGUF size (plus a coarse KV-cache estimate for `:server` mode) and **refuses** over-budget loads, naming the device that didn't fit:

```elixir
# combined (integer) budget
{:error, {:insufficient_memory, device: :total, required: r, available: a}} = ...

# per-device (:auto / map) budget — e.g. GPU 3 is full
{:error, {:insufficient_memory, device: {:gpu, 3}, required: r, available: a}} =
  LlamaCppEx.ModelManager.load("too-big", {:path, "70b.gguf"}, n_gpu_layers: -1, main_gpu: 3)
```

`device` is `:total` (combined), `:ram`, or `{:gpu, index}`. There is no automatic eviction — unload a model yourself to make room. `LlamaCppEx.devices/0` lists each GPU's `:memory_total`/`:memory_free` and its `:gpu_index` (the same index space as `:tensor_split`).

> **Coarse estimation:** footprint is advisory. Partial offload (`0 < n_gpu_layers < n_layers`) is treated as fully on GPU; compute buffers and fragmentation aren't modeled.

### Unloading and memory reclamation

Model cleanup is garbage-collection based. `unload/1` stops the backing server (dropping its context and model references) and forces a GC. Because reclamation is by GC, **any caller still holding a `%LlamaCppEx.Model{}` obtained via `fetch_model/1` keeps the model alive** past `unload/1` — prefer id-based routing and avoid holding raw refs.

## Speculative decoding (MTP)

Multi-Token Prediction speculative decoding (upstream PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)) drafts several tokens at once via a head shipped inside the same GGUF as the target model. Upstream llama-server reports ~2x speedup at ~75% draft acceptance on Qwen 3.6.

> **Performance note: Apple Silicon.** The upstream 2× claim is from NVIDIA datacenter GPUs, where a batched verify decode costs ~1.2× a single-token decode. On Apple Silicon (Metal), a 4-wide verify costs ~2.4× a single decode, which cancels MTP's iteration savings. We measured upstream's own `llama-server --spec-type draft-mtp` on M1 Max: **39.80 tok/s with MTP vs 39.14 tok/s plain** on Qwen 3.6 35B-A3B (1.02×) — i.e. effectively zero speedup from the reference implementation itself. This matches the pattern in upstream [#23011](https://github.com/ggml-org/llama.cpp/issues/23011); a Metal MTP optimization is tracked in [#23114](https://github.com/ggml-org/llama.cpp/pull/23114).
>
> **Tuning for Apple Silicon:** use `n_draft: 1`. With one draft per iteration the verify batch is only 2-wide (much cheaper on Metal) and acceptance jumps to ~79% on Qwen 3.6 35B-A3B. Our measurements on M1 Max with `n_draft: 1`:
> - Qwen 3.6 35B-A3B-MTP (hybrid MoE): plain 39.5 → MTP **44.0 tok/s (1.11×)**
> - Qwen 3.6 27B (dense): plain 10.7 → MTP **10.6 tok/s (~1.0×, neutral)**
>
> Larger `n_draft` hurts on Metal because verify cost grows faster than acceptance benefit. On NVIDIA, `n_draft: 3` is the right default — that's what the upstream 2× number assumes.

### Other speculative types (EAGLE-3, DFlash, n-gram)

Upstream llama.cpp implements more speculative types behind the same `common_speculative` API — `draft-eagle3`, `draft-dflash` (block-diffusion drafting via a separate drafter GGUF), and several n-gram self-speculation modes. **This binding currently exposes only MTP**: `MTP.init/2` pins `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` and builds both contexts from the same model, so there is no way to load a separate drafter model yet.

> **DFlash status (July 2026, llama.cpp b9932).** DFlash runs end-to-end on Metal via upstream `llama-cli`/`llama-server`, but we measured it *slower* than plain decoding on Apple Silicon at small target sizes: Qwen3.5-4B target + z-lab 0.6B drafter on M4 Max reached 42 tok/s with DFlash vs 85 tok/s plain (greedy sampling; 30% draft acceptance, mean accepted run 2.8 — and stochastic sampling at `temp 0.8` collapses acceptance to ~7%). The Metal economics are the same as the MTP note above (wide verify batches are expensive), and the community drafter-GGUF conversions are still churning: of three third-party Qwen 4B drafter repos tested, only one loads with current upstream (the others hit the `dflash-draft` arch mismatch [#25116](https://github.com/ggml-org/llama.cpp/issues/25116) or lack the `target_layers` metadata added by the conversion refactor [#25110](https://github.com/ggml-org/llama.cpp/pull/25110)). Worth revisiting when the drafter format settles; the natural entry point is a `spec_type` + drafter-model option on `speculative_init`.

### Models with MTP heads

- [`ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF) (recommended: `Q4_K_M`, ~21 GB)
- [`ggml-org/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-27B-MTP-GGUF)
- [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)

A regular (non-MTP) Qwen 3.6 quant will fail at `LlamaCppEx.MTP.init/2` — the GGUF must contain `mtp-*` tensors.

### Usage

#### Minimal: stream a single response

```elixir
:ok = LlamaCppEx.init()

{:ok, model} =
  LlamaCppEx.load_model(
    Path.expand("~/Downloads/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"),
    n_gpu_layers: 999
  )

# Build the speculative session once — it owns a target context and a
# separate MTP draft context on the *same* model file (no extra download).
{:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 8192)

mtp
|> LlamaCppEx.MTP.stream("Write a haiku about the sea:", max_tokens: 256)
|> Stream.each(&IO.write/1)
|> Stream.run()

# Final stats (also returned via the {:done, stats} stream event)
stats = LlamaCppEx.MTP.stats(mtp)
IO.puts("\nacceptance: #{Float.round(stats.acceptance_rate * 100, 1)}%  " <>
        "throughput: #{Float.round(stats.tokens_per_sec, 1)} tok/s")
```

#### Synchronous generate (collect to a string)

```elixir
{:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 4096)

{:ok, text} =
  LlamaCppEx.MTP.generate(mtp, "Explain monads to a Go programmer:",
    max_tokens: 200,
    temp: 0.7,
    top_p: 0.95,
    seed: 42
  )

IO.puts(text)
```

#### Reuse a session across multiple prompts

`MTP.init/2` allocates two `llama_context`s and the speculative state. It's the expensive bit. Reuse the same `%MTP{}` value across calls — KV caches are cleared at the start of each `stream/3` / `generate/3`:

```elixir
{:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 8192)

for q <- ["What is Elixir?", "What is OTP?", "What is BEAM?"] do
  IO.puts("\n> #{q}")
  mtp |> LlamaCppEx.MTP.stream(q, max_tokens: 150) |> Stream.each(&IO.write/1) |> Stream.run()
end

# Counters are cumulative across all calls on this session.
LlamaCppEx.MTP.stats(mtp) |> IO.inspect(label: "cumulative")
```

#### Watch stats live from a separate process

`MTP.stats/1` is lock-free, so a sibling process can poll it while a stream is in flight — handy for Phoenix LiveView dashboards:

```elixir
parent = self()

gen_task =
  Task.async(fn ->
    mtp
    |> LlamaCppEx.MTP.stream("Generate a 500-line Python implementation of A*:",
      max_tokens: 1024,
      temp: 0.7
    )
    |> Enum.into("")
    |> then(&send(parent, {:done, &1}))
  end)

# Sample every 200 ms while the generation runs.
Stream.repeatedly(fn ->
  Process.sleep(200)
  s = LlamaCppEx.MTP.stats(mtp)
  IO.puts(
    "iters=#{s.iters}  emitted=#{s.tokens_emitted}  " <>
      "accept=#{Float.round(s.acceptance_rate * 100, 1)}%  " <>
      "tok/s=#{Float.round(s.tokens_per_sec, 1)}"
  )
end)
|> Stream.take_while(fn _ -> not Task.yield(gen_task, 0) |> match?({:ok, _}) end)
|> Stream.run()

Task.await(gen_task, :infinity)
```

For in-band progress events (no separate process), use `stream_events/3` with `emit_stats_every`:

```elixir
mtp
|> LlamaCppEx.MTP.stream_events("Write a sonnet:",
  max_tokens: 400,
  emit_stats_every: 32
)
|> Enum.each(fn
  {:token, _id, text} -> IO.write(text)
  {:stats, s}        -> IO.puts("\n[stats] accept=#{Float.round(s.acceptance_rate * 100, 1)}%")
  {:done, _final}    -> IO.puts("\n[done]")
  {:eog, _}          -> IO.puts("\n[eog]")
end)
```

### Options

`LlamaCppEx.MTP.init/2`:

  * `:n_draft` — draft tokens proposed per iteration (default `3`). On NVIDIA, 2–4 is the sweet spot. On Apple Silicon, set this to `1` — see the Apple Silicon performance note above.
  * `:n_ctx`, `:n_threads`, `:flash_attn`, `:type_k`/`:type_v`, `:offload_kqv`, … — any `LlamaCppEx.Context` option; applied to both target and draft contexts.

`LlamaCppEx.MTP.stream/3`:

  * `:max_tokens` (default `256`), plus all sampling options (`:temp`, `:top_k`, `:top_p`, `:min_p`, `:seed`, `:penalty_*`, `:grammar`).
  * `:emit_stats_every` — when set, periodic `{:stats, _}` events become available via `stream_events/3`.

### Caveats

- Upstream currently requires `n_parallel = 1` for MTP; this binding mirrors that. Use `LlamaCppEx.Server` for concurrent non-MTP inference, or stick to a single MTP session at a time.
- Prompt prefill is somewhat slower with MTP than without (the MTP head also processes the prompt). The win shows up at decode time.

See [`examples/mtp_speculative.exs`](examples/mtp_speculative.exs) for a runnable demo with full timing breakdown.

## Benchmarks

Measured on Apple M4 Max (64 GB), Metal backend (`n_gpu_layers: -1`).

### Single-model generation speed

| Model | Quantization | Tokens/sec |
|-------|-------------|------------|
| Llama 3.2 3B Instruct | Q4_K_XL | 125.6 |
| Ministral 3 3B Reasoning | Q4_K_XL | 113.0 |
| Ministral 3 3B Instruct | Q4_K_XL | 104.3 |
| GPT-OSS 20B | Q4_K_XL | 79.4 |
| Qwen3.5-35B-A3B | Q6_K | 56.0 |
| Qwen3.5-27B | Q4_K_XL | 17.5 |

### Qwen3.6-35B-A3B (v0.7.8)

New `qwen35moe` architecture with Gated Delta Net (hybrid linear/full attention). Measured on Apple M1 Max (64 GB) with v0.7.8 bindings — not directly comparable to the M4 Max numbers above.

| Model | Quantization | Tokens/sec (M1 Max) |
|-------|-------------|---------------------|
| Qwen3.6-35B-A3B | Q4_K_XL | 43.8 |

128-token generation, `temp: 0.0`, 3-run average (43.3 / 44.1 / 44.0 t/s).

### Single-sequence generation (Qwen3-4B Q4_K_M)

| Prompt | 32 tokens | 128 tokens |
|--------|-----------|------------|
| short (6 tok) | 0.31s (3.19 ips) | 1.01s (0.98 ips) |
| medium (100 tok) | 0.36s (2.79 ips) | 1.06s (0.94 ips) |
| long (500 tok) | 0.65s (1.53 ips) | 1.29s (0.77 ips) |

### Continuous batching throughput (Qwen3-4B Q4_K_M)

```
max_tokens: 32, prompt: "short"
──────────────────────────────────────────────────────────────────────────────
Concurrency  Wall time    Total tok/s  Per-req tok/s  Speedup  Avg batch
1            318ms        100.6        100.6          1.00x    1.1
2            440ms        145.5         72.7          1.45x    2.2
4            824ms        155.3         38.8          1.54x    4.5
```

Run benchmarks yourself:

```bash
MIX_ENV=bench mix deps.get
LLAMA_MODEL_PATH=path/to/model.gguf MIX_ENV=bench mix run bench/single_generate.exs
LLAMA_MODEL_PATH=path/to/model.gguf MIX_ENV=bench mix run bench/server_concurrent.exs
```

## Running Qwen3.5-35B-A3B

[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GGUF) is a Mixture-of-Experts model with 35B total parameters but only 3B active per token. It supports 256K context and both thinking (CoT) and non-thinking modes.

### Hardware requirements

| Quantization | RAM / VRAM | File size |
|-------------|------------|-----------|
| Q4_K_M | ~20 GB | ~19 GB |
| Q8_0 | ~37 GB | ~36 GB |
| BF16 | ~70 GB | ~67 GB |

### Download

```bash
# Install the HuggingFace CLI if needed: pip install huggingface-hub
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir models/
```

### Thinking mode (general)

```elixir
:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model("models/Qwen3.5-35B-A3B-Q4_K_M.gguf", n_gpu_layers: -1)

# Qwen3.5 recommended: temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Explain the birthday paradox."}
], max_tokens: 2048, temp: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, penalty_present: 1.5)
```

### Thinking mode (math/code)

```elixir
# For math and code, lower temperature without presence penalty
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "Write a function to find the longest palindromic substring."}
], max_tokens: 4096, temp: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0)
```

### Non-thinking mode

```elixir
# Disable thinking via enable_thinking option (uses Jinja chat template kwargs)
{:ok, reply} = LlamaCppEx.chat(model, [
  %{role: "user", content: "What is the capital of France?"}
], max_tokens: 256, enable_thinking: false, temp: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, penalty_present: 1.5)
```

### Streaming with Server

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  n_gpu_layers: -1,
  n_parallel: 2,
  n_ctx: 16384,
  temp: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, penalty_present: 1.5
)

LlamaCppEx.Server.stream(server, "Explain monads in simple terms", max_tokens: 1024)
|> Enum.each(&IO.write/1)
```

### Qwen3.5 enable_thinking benchmarks

Measured on **MacBook Pro, Apple M4 Max (16-core, 64 GB)**, Metal backend, `n_gpu_layers: -1`, 512 output tokens, `temp: 0.6`.

| Metric | Qwen3.5-27B (Q4_K_XL) | Qwen3.5-35B-A3B (Q6_K) |
|---|---|---|
| | Think ON / Think OFF | Think ON / Think OFF |
| **Prompt tokens** | 65 / 66 | 65 / 66 |
| **Output tokens** | 512 / 512 | 512 / 512 |
| **TTFT** | 599 ms / 573 ms | 554 ms / 191 ms |
| **Prompt eval** | 108.5 / 115.2 t/s | 117.3 / 345.5 t/s |
| **Gen speed** | 17.5 / 17.3 t/s | 56.0 / 56.0 t/s |
| **Total time** | 29.77 / 30.10 s | 9.69 / 9.33 s |

The MoE model (35B-A3B) is ~3.2x faster at generation since only 3B parameters are active per token despite the 35B total. Thinking mode only affects the prompt template, not inference speed.

## Examples

The `examples/` directory contains runnable scripts demonstrating key features:

```bash
# Basic text generation
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/basic_generation.exs

# Streaming tokens to terminal
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/streaming.exs

# Interactive multi-turn chat
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/chat.exs

# JSON Schema constrained generation + Ecto integration
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/structured_output.exs

# Embedding generation and cosine similarity
LLAMA_EMBEDDING_MODEL_PATH=/path/to/embedding-model.gguf mix run examples/embeddings.exs

# Continuous batching server with concurrent requests
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/server.exs
```

## Architecture

```
Elixir API (lib/)
    │
LlamaCppEx.NIF (@on_load, stubs)
    │
C++ NIF layer (c_src/) — fine.hpp for RAII + type encoding
    │
llama.cpp static libs (vendor/llama.cpp, built via CMake)
    │
Hardware (CPU / Metal / CUDA / Vulkan)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

llama.cpp is licensed under the MIT License.
