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
- **Multi-Token Prediction (MTP) speculative decoding** — ~2x token-generation speedup on Qwen 3.6 with live acceptance-rate stats
- **Prefix caching** — same-slot KV cache reuse for multi-turn chat (1.23x faster)
- **Pluggable batching strategies** — DecodeMaximal, PrefillPriority, Balanced
- **Pre-tokenized API** — tokenize outside the GenServer for lower contention
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

- C++17 compiler (GCC, Clang, or MSVC)
- CMake 3.14+
- Git (for the llama.cpp submodule)

### Backend Selection

```bash
mix compile                        # Auto-detect (Metal on macOS, CUDA if nvcc found, else CPU)
LLAMA_BACKEND=metal mix compile    # Apple Silicon GPU
LLAMA_BACKEND=cuda mix compile     # NVIDIA GPU
LLAMA_BACKEND=vulkan mix compile   # Vulkan
LLAMA_BACKEND=cpu mix compile      # CPU only
```

Power users can pass arbitrary CMake flags:

```bash
LLAMA_CMAKE_ARGS="-DGGML_CUDA_FORCE_CUBLAS=ON" mix compile
```

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
{:req, "~> 0.5"}
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

The server caches KV state between requests on the same slot. Multi-turn chat benefits automatically — the system prompt and prior turns aren't recomputed:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 4,
  cache_prompt: true  # opt-in (default: false)
)
```

Benchmark: **1.23x faster** for multi-turn conversations (487ms vs 597ms per 4-turn exchange).

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

## Speculative decoding (MTP)

Multi-Token Prediction speculative decoding (upstream PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)) drafts several tokens at once via a head shipped inside the same GGUF as the target model. Upstream llama-server reports ~2x speedup at ~75% draft acceptance on Qwen 3.6.

> **Status (initial release):** this binding ships a single-sequence MTP loop with full statistics. On a hybrid model like Qwen 3.6 (GDN + attention layers) we still pay a per-iteration recurrent-state checkpoint, and our current draft acceptance peaks around 50–60% — so end-to-end throughput is roughly on par with non-MTP decoding rather than 2x. The pipeline is correct and stats are accurate; performance tuning (priming the MTP head, lighter checkpointing) is tracked for follow-up. See `LlamaCppEx.MTP` and `examples/mtp_benchmark.exs`.

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

  * `:n_draft` — draft tokens proposed per iteration (default `3`). 2–4 is the sweet spot.
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
