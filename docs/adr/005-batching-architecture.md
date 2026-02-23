# ADR 005: Batching Architecture

## Status

Proposed (Phase 3)

## Context

Serving multiple concurrent users with a single model requires efficient GPU utilization. Without batching, each user's token generation runs as a separate forward pass, leaving the GPU underutilized during the decode phase.

llama.cpp's `llama_batch` API supports multi-sequence batching — processing tokens from multiple independent sequences in a single `llama_decode` call.

## Decision

We will implement a **GenServer-based batching server** that accumulates requests from multiple callers and flushes them as a single batched `llama_decode` call.

## Background: Why Batching Matters

### The Prefill vs Decode Asymmetry

| Phase | Operation | Bottleneck | GPU Utilization |
|---|---|---|---|
| **Prefill** | Process entire prompt | Compute-bound (matrix-matrix) | ~100% |
| **Decode** | Generate 1 token | Memory-bandwidth-bound (matrix-vector) | 10-30% |

During decode, each sequence contributes a single token per step. The GPU performs a matrix-vector multiply (accessing all model weights to produce one output), which is limited by memory bandwidth, not compute.

### Batching Converts the Bottleneck

Batching N sequences together converts N matrix-vector multiplies into one matrix-matrix multiply:

| Sequences | Operation | Throughput |
|---|---|---|
| 1 | Matrix × Vector | ~30 tok/s |
| 8 | Matrix × Matrix (8 cols) | ~200 tok/s total (~25 tok/s per user) |
| 16 | Matrix × Matrix (16 cols) | ~350 tok/s total (~22 tok/s per user) |

Per-user latency decreases slightly, but total throughput increases dramatically.

## Design

### NIF Layer

Two new NIFs for batched operation:

#### `prefill(ctx, tokens, seq_id, n_past)`
Processes a prompt for a single sequence in `n_batch`-sized chunks. Only computes logits on the very last token. Runs on DirtyCPU.

#### `decode_batch(ctx, entries)`
Accepts a list of `{seq_id, token_id, position}` tuples, builds a single `llama_batch`, calls `llama_decode` once, and samples a next token for each sequence. Returns `[{seq_id, next_token_id, token_text}]`. Runs on DirtyCPU.

### GenServer Batcher

```
                   ┌─────────────────────────┐
  Caller 1 ──────→│                           │
  Caller 2 ──────→│  LlamaCppEx.Server        │
  Caller 3 ──────→│                           │
                   │  State:                   │
                   │  - pending: [{from, ...}] │
                   │  - sequences: %{id => ..} │
                   │  - seq_pool: MapSet       │
                   │                           │
                   │  Flush trigger:            │
                   │  - batch_size reached      │
                   │  - batch_timeout (20ms)    │
                   └─────────┬─────────────────┘
                             │
                     decode_batch NIF
                             │
                   ┌─────────┴─────────────────┐
                   │  GenServer.reply/2         │
                   │  per caller                │
                   └───────────────────────────┘
```

#### Sequence Lifecycle

1. Caller sends `{:generate, prompt, opts}` → server acquires a `seq_id` from pool
2. Server prefills prompt tokens for the sequence
3. Decode loop: server batches all active sequences, calls `decode_batch`, replies to callers whose sequences finished or produced a token
4. On completion/error/timeout: server calls `llama_memory_seq_rm` to free the KV cache slot, returns `seq_id` to pool

#### Configuration

```elixir
{LlamaCppEx.Server,
  model_path: "model.gguf",
  n_ctx: 8192,            # Total KV cache (shared across all sequences)
  n_parallel: 8,          # Max concurrent sequences
  n_gpu_layers: -1,       # GPU layer offload
  batch_size: 512,        # Max tokens per decode call
  batch_timeout: 20}      # ms accumulation window
```

#### Flush Strategy

The server uses `:noreply` for `handle_call` to hold callers, then flushes on:

1. **Batch size**: When `n_pending >= batch_size`, flush immediately
2. **Batch timeout**: After `batch_timeout` ms with pending entries, flush whatever is accumulated

This balances latency (small batches flush quickly) vs throughput (large batches amortize the forward pass).

### Shared System Prompt

For chat applications where every request starts with the same system prompt:

1. Prefill system prompt tokens tagged with ALL sequence IDs
2. When a new sequence starts, `llama_memory_seq_cp` copies the shared prefix
3. Only user-specific tokens need prefilling per request

This avoids redundant computation of the system prompt for every request.

### KV Cache Management

Total KV cache capacity (`n_ctx`) is shared across all active sequences:

```
n_ctx = 8192
n_parallel = 8
max_per_sequence = n_ctx / n_parallel = 1024 tokens
```

The server tracks positions per sequence and enforces limits. When a sequence completes, `llama_memory_seq_rm` frees its cache slots for reuse.

## Alternatives Considered

### Thread Pool in C++

Run the batching loop entirely in C++ with a thread pool. Rejected because:
- Harder to debug and monitor from Elixir
- Loses BEAM's process supervision and fault tolerance
- More complex error handling across the language boundary

### One GenServer per Sequence

Each sequence gets its own GenServer + context. Rejected because:
- No batching benefit — each forward pass is still single-sequence
- N contexts × N KV caches = much higher memory usage
- Does not leverage GPU parallelism

### Nx.Serving

Wrap as an `Nx.Serving` for automatic batching. This is planned as an optional Phase 5 integration, but the core batcher is a GenServer for:
- No Nx dependency in the core library
- More control over sequence lifecycle and KV cache management
- Simpler mental model for users not using Nx

## Consequences

- The GenServer is a serialization point — all requests funnel through one process
- This is by design: `llama_decode` is not thread-safe, and batching requires coordinated access to the shared context
- The `batch_timeout` adds up to 20ms latency for the first request in a batch window
- Memory usage is bounded by `n_ctx` total, divided among `n_parallel` sequences
- Callers that are slower than generation speed won't cause backpressure issues (GenServer.reply is non-blocking)
