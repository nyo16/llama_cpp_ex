# ADR 008: Pluggable Batching Strategies

## Status

Accepted

## Context

ADR 006 introduced continuous batching with a hardcoded **decode-maximal** scheduling policy: decode tokens (one per generating slot) are always added to the batch first, with remaining budget filled by prefill chunks. This is optimal for interactive use where generation latency matters most, but suboptimal for batch processing workloads where throughput is the priority.

Different workloads have different needs:
- **Interactive chat**: Minimize generation latency → decode-maximal
- **Batch processing**: Maximize throughput → prefill-priority
- **Mixed workloads**: Fair allocation → balanced

## Decision

Extract batch building into a **behaviour module** (`LlamaCppEx.Server.BatchStrategy`) with a single callback:

```elixir
@callback build_batch(slots, budget, chunk_size, opts) :: {entries, updated_slots}
```

The server delegates to the configured strategy module each tick. Three built-in strategies are provided:

### DecodeMaximal (default)

Decode tokens first, prefill fills remaining budget. Prioritizes active generation — users waiting for tokens get the lowest latency. This is the existing behavior from ADR 006, extracted into its own module.

### PrefillPriority

Prefill chunks first, decode fills remaining budget. Gets new requests through the prefill phase faster, which maximizes overall throughput when processing many requests. Per-request generation latency may increase slightly.

### Balanced

Splits the budget equally between decode and prefill. Decode slots only need 1 token each, so unused decode budget flows to prefill. Fair under mixed workloads.

### Configuration

```elixir
LlamaCppEx.Server.start_link(
  model_path: "model.gguf",
  batch_strategy: LlamaCppEx.Server.Strategy.PrefillPriority
)
```

### Custom Strategies

Users can implement the behaviour for specialized scheduling:

```elixir
defmodule MyStrategy do
  @behaviour LlamaCppEx.Server.BatchStrategy

  @impl true
  def build_batch(slots, budget, chunk_size, opts) do
    # Custom logic
    {entries, updated_slots}
  end
end
```

The `opts` keyword list includes `:model_ref` (for token detokenization) and `:queue_depth` (for adaptive strategies).

## Performance

Benchmarked with Qwen3-0.6B-Q8_0 on Apple M1 Max (single request):

| Strategy | Average | Median |
|---|---|---|
| Balanced | 265ms | 259ms |
| DecodeMaximal | 274ms | 264ms |
| PrefillPriority | 287ms | 282ms |

For single requests, differences are minimal. The strategies diverge under concurrent load where decode/prefill budget allocation matters.

## Consequences

- Zero behavior change for existing users (default is DecodeMaximal, identical logic)
- Strategy modules are self-contained — each implements the full batch building logic
- The behaviour contract is simple enough for custom implementations
- Strategy receives `queue_depth` in opts, enabling adaptive strategies that respond to load
- Side effects (streaming, text accumulation, first-token timing) are co-located with batch building in the strategy — this keeps the server's tick loop clean but means strategies are tightly coupled to slot structure

## Alternatives Considered

### Configuration-driven scheduling (no behaviour)

Use atoms like `:decode_maximal` and switch internally. Rejected because it prevents custom strategies and moves scheduling logic back into the server module.

### Separate side effects from batch building

Have strategies only decide token allocation, with the server handling streaming/accumulation. Rejected because it would require two passes over the slots per tick and complicate the data flow. The current approach is simpler and the slot structure is internal anyway.
