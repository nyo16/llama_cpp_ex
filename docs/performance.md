# Performance Guide

This guide covers llama.cpp optimization parameters, server tuning, prefix caching, batching strategies, and optimization patterns for llama_cpp_ex.

## Server Configuration

The `LlamaCppEx.Server` manages a pool of concurrent inference slots with continuous batching. Key configuration options:

| Option | Default | Description |
|---|---|---|
| `n_parallel` | 4 | Number of concurrent inference slots |
| `n_ctx` | 8192 | Total KV cache size shared across all slots |
| `n_batch` | n_ctx | Maximum tokens per forward pass |
| `chunk_size` | 512 | Maximum prefill tokens per slot per tick |
| `cache_prompt` | false | Enable same-slot KV cache reuse |
| `batch_strategy` | DecodeMaximal | Batch building strategy module |
| `type_k` | `:f16` | KV cache K quantization type |
| `type_v` | `:f16` | KV cache V quantization type |
| `flash_attn` | `:auto` | Flash Attention mode |
| `offload_kqv` | `true` | Offload KQV ops to GPU |
| `op_offload` | `true` | Offload host tensor ops to device |

All options are also available on `LlamaCppEx.Context.create/2` and pass through from `LlamaCppEx.generate/3`, `LlamaCppEx.chat/3`, etc.

### Context Size (`n_ctx`)

The KV cache is shared across all active slots. As a rule of thumb:

```
effective_per_slot = n_ctx / n_parallel
```

Each slot needs enough room for its prompt tokens plus generated tokens. If a slot's total tokens exceed the per-slot budget, `batch_eval` will fail and the request will receive an error.

For multi-turn chat with long conversation histories, increase `n_ctx` accordingly:

```elixir
LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 4,
  n_ctx: 32768  # 8K per slot
)
```

### Chunk Size

Controls how many prompt tokens are processed per slot per tick during prefill. Smaller values reduce generation stalls (other slots keep generating while a long prompt is being prefilled), but increase the number of ticks needed to finish prefill.

- **Default (512)**: Good balance for interactive use
- **Larger (1024–2048)**: Faster prefill, but may stall generation for other slots
- **Smaller (128–256)**: Smoother generation at the cost of slower prefill

## KV Cache Quantization

By default, the KV cache uses F16 (half-precision float). You can quantize it to reduce memory usage by 2-4x, allowing larger context windows or more concurrent slots with the same hardware.

### Available Types

| Type | Memory vs F16 | Quality | Use Case |
|---|---|---|---|
| `:f32` | 2x more | Highest | Debugging, reference |
| `:f16` | Baseline | Excellent | Default — recommended for most use |
| `:bf16` | Same as F16 | Excellent | BFloat16 hardware support |
| `:q8_0` | **2x less** | Near-lossless | **Recommended** — best memory/quality tradeoff |
| `:q5_1` | ~3x less | Good | Aggressive savings with acceptable quality |
| `:q5_0` | ~3x less | Good | Slightly less quality than q5_1 |
| `:q4_1` | **4x less** | Acceptable | Maximum context length |
| `:q4_0` | **4x less** | Lower | Only when memory is critical |

### Usage

```elixir
# Standalone context — Q8_0 (recommended for most users)
{:ok, ctx} = LlamaCppEx.Context.create(model,
  n_ctx: 32768,
  type_k: :q8_0,
  type_v: :q8_0
)

# With the Server
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 8,
  n_ctx: 32768,
  type_k: :q8_0,
  type_v: :q8_0
)

# High-level API
{:ok, text} = LlamaCppEx.generate(model, "Hello",
  max_tokens: 256,
  type_k: :q8_0,
  type_v: :q8_0
)

# Aggressive — Q4_0 for maximum context
{:ok, ctx} = LlamaCppEx.Context.create(model,
  n_ctx: 131072,
  type_k: :q4_0,
  type_v: :q4_0
)
```

### Quality Validation

We tested Q8_0 against F16 with 10 deterministic prompts (temp: 0.0) covering arithmetic, factual recall, sequence completion, and more. **All 10 produced bit-for-bit identical output.** See `test/kv_quantization_test.exs` for the full regression suite.

| Test Case | F16 vs Q8_0 |
|---|---|
| Arithmetic (2+2) | IDENTICAL |
| Counting (1-5) | IDENTICAL |
| Capital city (Paris) | IDENTICAL |
| Largest ocean (Pacific) | IDENTICAL |
| Opposite (hot→cold) | IDENTICAL |
| Sequence (2,4,6,8→10) | IDENTICAL |
| Color (sky→blue) | IDENTICAL |
| Continent (Japan→Asia) | IDENTICAL |
| Multiplication (10×5) | IDENTICAL |
| Chemistry (H2O) | IDENTICAL |

Run the regression tests yourself:

```bash
LLAMA_MODEL_PATH=model.gguf mix test test/kv_quantization_test.exs --include slow
```

### When to Use Each Type

- **Interactive chat**: `:q8_0` — saves memory with no perceptible quality loss
- **Long document processing**: `:q8_0` or `:q5_1` — fit more context
- **Many concurrent users**: `:q8_0` — double the slots with same memory
- **Research / precision-critical**: `:f16` (default) — maximum precision
- **Maximum context length**: `:q4_0` — 4x memory savings, test quality for your use case

## Flash Attention

Flash Attention computes attention more efficiently, using less memory and running faster, especially for long sequences. llama.cpp enables it automatically when supported.

### Usage

```elixir
# Auto (default) — llama.cpp decides based on hardware
{:ok, ctx} = LlamaCppEx.Context.create(model, flash_attn: :auto)

# Force enable — error if hardware doesn't support it
{:ok, ctx} = LlamaCppEx.Context.create(model, flash_attn: :enabled)

# Force disable — useful for debugging or comparing performance
{:ok, ctx} = LlamaCppEx.Context.create(model, flash_attn: :disabled)

# With Server
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  flash_attn: :enabled
)

# With high-level API
{:ok, text} = LlamaCppEx.generate(model, "Hello",
  max_tokens: 256,
  flash_attn: :enabled
)
```

### When to Use

- `:auto` (default) — let llama.cpp decide. Works well in most cases.
- `:enabled` — force on when you know your hardware supports it (Metal on Apple Silicon, CUDA compute capability 7.0+). Can improve prefill speed significantly.
- `:disabled` — for debugging if you suspect flash attention is causing issues, or for benchmarking the difference.

## GPU Offload Control

Two flags control how operations are distributed between CPU and GPU:

```elixir
{:ok, ctx} = LlamaCppEx.Context.create(model,
  offload_kqv: true,   # Offload KQV attention ops + KV cache to GPU (default: true)
  op_offload: true      # Offload host tensor operations to device (default: true)
)
```

- **`offload_kqv: false`** — keep KQV operations on CPU. Useful when GPU memory is tight and you'd rather use it for model weights.
- **`op_offload: false`** — disable general operation offloading. Rarely needed.

For most users, the defaults (`true` for both) are optimal.

## RoPE Context Extension

Extend the model's context window beyond its training length using Rotary Position Embedding (RoPE) scaling.

### Linear Scaling

Simple frequency scaling. Works well for moderate extensions (2-4x):

```elixir
# Extend 4K training context to 16K
{:ok, ctx} = LlamaCppEx.Context.create(model,
  n_ctx: 16384,
  rope_scaling_type: :linear,
  rope_freq_scale: 0.25  # 4x extension (1/4 = 0.25)
)
```

### YaRN Scaling

Better quality for larger extensions. Recommended for 4x+ extensions:

```elixir
# Extend to 32K with YaRN
{:ok, ctx} = LlamaCppEx.Context.create(model,
  n_ctx: 32768,
  rope_scaling_type: :yarn,
  rope_freq_base: 1_000_000.0
)

# Full YaRN parameter control
{:ok, ctx} = LlamaCppEx.Context.create(model,
  n_ctx: 65536,
  rope_scaling_type: :yarn,
  yarn_ext_factor: 1.0,
  yarn_attn_factor: 1.0,
  yarn_beta_fast: 32.0,
  yarn_beta_slow: 1.0,
  yarn_orig_ctx: 4096
)
```

### LongRoPE

For models trained with LongRoPE (some newer models):

```elixir
{:ok, ctx} = LlamaCppEx.Context.create(model,
  n_ctx: 131072,
  rope_scaling_type: :longrope
)
```

### Custom Frequency Base

Override the RoPE base frequency directly. Some models use high base frequencies (e.g., 500,000 or 1,000,000) for long context:

```elixir
{:ok, ctx} = LlamaCppEx.Context.create(model,
  n_ctx: 32768,
  rope_freq_base: 500_000.0  # Override model's default
)
```

### Note

Context extension always involves a quality tradeoff. The model was trained on a specific context length, and extending beyond it degrades output quality progressively. Test with your specific model and use case. Many modern models (Qwen3, Llama 3.1+) already support long contexts natively and don't need RoPE scaling.

## Attention Type

Control whether the model uses causal or non-causal attention. This primarily matters for embedding models:

```elixir
# For embedding models — non-causal attention gives better embeddings
{:ok, ctx} = LlamaCppEx.Context.create(model,
  embeddings: true,
  attention_type: :non_causal
)

# For text generation — causal (default, model decides)
{:ok, ctx} = LlamaCppEx.Context.create(model,
  attention_type: :causal
)
```

## Model Loading Options

Additional options when loading models:

```elixir
{:ok, model} = LlamaCppEx.load_model("model.gguf",
  n_gpu_layers: -1,       # Offload all layers to GPU
  use_mmap: true,          # Memory-map file (default, faster loading)
  use_mlock: true,         # Pin in RAM (prevent swapping)
  use_direct_io: false,    # Bypass page cache
  check_tensors: true      # Validate tensor data (debugging)
)
```

## Complete Optimization Example

Here's a production-ready server configuration combining multiple optimizations:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_gpu_layers: -1,

  # Concurrency
  n_parallel: 8,
  n_ctx: 32768,

  # KV cache quantization — 2x memory savings
  type_k: :q8_0,
  type_v: :q8_0,

  # Flash attention — faster prefill
  flash_attn: :enabled,

  # Prefix caching — skip redundant prefill for multi-turn chat
  cache_prompt: true,

  # Strategy — latency-optimized for interactive use
  batch_strategy: LlamaCppEx.Server.Strategy.DecodeMaximal,

  # Sampling
  temp: 0.7,
  top_p: 0.9
)
```

This gives you: 8 concurrent users, 4K tokens per user, quantized KV cache, flash attention, and prefix caching — all working together.

## Prefix Caching

When `cache_prompt: true`, the server retains the KV cache after a slot completes a request. On the next request, it detects the longest common prefix with the cached tokens and skips re-computing that portion.

### When It Helps

- **Multi-turn chat**: Each message appends to the conversation — the system prompt and prior turns are cached
- **Shared system prompts**: Multiple users with the same system prompt benefit when routed to the same slot
- **Few-shot prompting**: Shared examples only need to be computed once

### Benchmark Results

Qwen3-0.6B-Q8_0, Apple M1 Max, 4-turn multi-turn chat:

| Scenario | Average | Median | Improvement |
|---|---|---|---|
| WITH prefix cache | 487ms | 452ms | — |
| WITHOUT prefix cache | 597ms | 591ms | — |
| **Speedup** | **1.23x** | **1.31x** | 110ms saved |

### Prefix-Affinity Slot Selection

When acquiring an idle slot, the server prefers the slot whose cached token history has the longest common prefix with the incoming request. This maximizes cache hits without requiring manual slot assignment.

### Disabling

```elixir
LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  cache_prompt: false  # Always clear KV cache
)
```

### Telemetry

Monitor cache effectiveness via the `:llama_cpp_ex, :server, :request, :done` telemetry event:

```elixir
:telemetry.attach("cache-monitor", [:llama_cpp_ex, :server, :request, :done],
  fn _event, measurements, _meta, _config ->
    ratio = Float.round(measurements.prefix_cache_ratio * 100, 1)
    IO.puts("Cache hit: #{measurements.prefix_cache_tokens} tokens (#{ratio}%)")
  end, nil)
```

## Batching Strategies

The server supports pluggable batching strategies that control how the token budget is allocated between decode (generation) and prefill (prompt processing) each tick.

### Built-in Strategies

#### DecodeMaximal (default)

```elixir
batch_strategy: LlamaCppEx.Server.Strategy.DecodeMaximal
```

Decode tokens get priority. Best for **interactive use** where users are waiting for each generated token. Generation latency is minimized at the cost of slower prompt prefill for new requests.

#### PrefillPriority

```elixir
batch_strategy: LlamaCppEx.Server.Strategy.PrefillPriority
```

Prefill chunks get priority. Best for **batch processing** where overall throughput matters more than per-request latency. New requests get through prefill faster, but active generation may see slightly higher latency.

#### Balanced

```elixir
batch_strategy: LlamaCppEx.Server.Strategy.Balanced
```

Splits the budget equally between decode and prefill. Best for **mixed workloads** where both latency and throughput matter.

### Custom Strategies

Implement the `LlamaCppEx.Server.BatchStrategy` behaviour:

```elixir
defmodule MyAdaptiveStrategy do
  @behaviour LlamaCppEx.Server.BatchStrategy

  @impl true
  def build_batch(slots, budget, chunk_size, opts) do
    queue_depth = Keyword.get(opts, :queue_depth, 0)

    # Adapt based on queue pressure
    if queue_depth > 4 do
      # High load: prioritize prefill to clear the queue
      PrefillPriority.build_batch(slots, budget, chunk_size, opts)
    else
      # Low load: prioritize generation latency
      DecodeMaximal.build_batch(slots, budget, chunk_size, opts)
    end
  end
end
```

## Pre-Tokenized API

For high-throughput scenarios, tokenize prompts outside the GenServer to reduce mailbox contention:

```elixir
model = LlamaCppEx.Server.get_model(server)

# Tokenize in the caller process (parallel-safe)
{:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)

# Send pre-tokenized — skips tokenization in the GenServer
{:ok, text} = LlamaCppEx.Server.generate_tokens(server, tokens, max_tokens: 256)
```

This matters under concurrent load where multiple callers serialize on the GenServer mailbox. Each tokenization call saved is one fewer blocking operation in the critical path.

## Optimization Patterns

### Multi-Turn Chat

Combine prefix caching with the chat API for optimal multi-turn performance:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 4,
  n_ctx: 16384,      # Room for long conversations
  cache_prompt: true  # Reuse KV cache across turns
)

# Each turn extends the previous — prefix cache skips re-computing history
messages = [%{role: "system", content: "You are helpful."}]

for user_msg <- conversation do
  messages = messages ++ [%{role: "user", content: user_msg}]
  {:ok, prompt} = LlamaCppEx.Chat.apply_template(model, messages)
  {:ok, reply} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 256)
  messages = messages ++ [%{role: "assistant", content: reply}]
end
```

### Batch Processing

For processing many independent requests, use prefill-priority strategy:

```elixir
{:ok, server} = LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  n_parallel: 8,
  batch_strategy: LlamaCppEx.Server.Strategy.PrefillPriority,
  cache_prompt: false  # No benefit for independent prompts
)

results =
  prompts
  |> Task.async_stream(fn prompt ->
    LlamaCppEx.Server.generate(server, prompt, max_tokens: 100)
  end, max_concurrency: 8, timeout: 60_000)
  |> Enum.to_list()
```

## Running Benchmarks

The project includes Benchee benchmarks in `bench/`:

```bash
# Prefix cache comparison
MIX_ENV=bench LLAMA_MODEL_PATH=model.gguf mix run bench/prefix_cache.exs

# Strategy comparison
MIX_ENV=bench LLAMA_MODEL_PATH=model.gguf mix run bench/strategies.exs

# Tokenization overhead
MIX_ENV=bench LLAMA_MODEL_PATH=model.gguf mix run bench/tokenize_overhead.exs

# Existing benchmarks
MIX_ENV=bench LLAMA_MODEL_PATH=model.gguf mix run bench/single_generate.exs
MIX_ENV=bench LLAMA_MODEL_PATH=model.gguf mix run bench/server_concurrent.exs
```
