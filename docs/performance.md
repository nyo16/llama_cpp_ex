# Performance Guide

This guide covers server tuning, prefix caching, batching strategies, and optimization patterns for llama_cpp_ex.

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
