# ADR 006: Continuous Batching

## Status

Accepted (supersedes ADR 005)

## Context

ADR 005 proposed a batch-accumulate-flush model with `batch_timeout` for serving multiple concurrent users. In practice, a tick-driven continuous batching loop — inspired by llama.cpp's server, vLLM, and SGLang — is simpler and lower latency.

The previous `Server` implementation processed slots sequentially: N active slots meant N separate `llama_decode` calls per tick, each a full forward pass. This wasted GPU parallelism and stalled generation during long prompt prefills.

## Decision

Implement continuous batching with a single forward pass per tick, mixing decode tokens and prefill chunks in one batch.

### Key Design Elements

**Decode-maximal scheduling:** Decode tokens (one per generating slot) are always added to the batch first. They represent active generation that users are waiting on, so they get priority.

**Chunked prefill:** Long prompts are split into chunks (default 512 tokens) and processed across multiple ticks, interleaved with decode tokens from other slots. This prevents a large prompt from stalling all generation.

**Token budget:** Each tick's batch is capped at `n_batch` tokens. The Elixir scheduler enforces this — no sub-batching in C++.

**Two new NIFs:**

- `batch_eval(ctx, entries)` — Builds a `llama_batch` from a list of `{token_id, pos, seq_id, logits_flag}` tuples and calls `llama_decode`. Forward pass only, no sampling. Runs on DirtyCPU.
- `sampler_sample_at(sampler, ctx, idx)` — Calls `llama_sampler_sample` with an explicit batch index, enabling sampling at specific positions after a batched decode. Runs on Normal scheduler (fast — just reads logits).

**Per-slot samplers with batch-index-aware sampling:** After `batch_eval`, each slot samples at its specific batch index using `sampler_sample_at`. This preserves per-slot sampler state (grammar, penalties, etc.).

**Request queue:** When all slots are busy, requests enter a FIFO `:queue` instead of being rejected with `{:error, :no_slots}`. Requests are served as slots become available. An optional `:max_queue` limit provides backpressure.

**Telemetry events:** `:telemetry` events are emitted for request completion and per-tick batch metrics, enabling monitoring without coupling to a specific metrics backend.

### Tick Loop

```
Phase 1 — Finish completed slots
Phase 2 — Build batch (decode tokens first, then prefill chunks)
Phase 3 — Forward pass (single batch_eval call)
Phase 4 — Sample (sampler_sample_at per slot at their batch index)
Phase 5 — Continue (schedule next tick if any active slots)
```

### Slot States

```
:idle → :prefilling → :generating → :idle
```

- **:idle** — Slot is available for new requests
- **:prefilling** — Prompt tokens being chunked into batches across ticks
- **:generating** — Actively producing tokens, one per tick

## Consequences

- Single forward pass per tick improves GPU utilization dramatically compared to sequential per-slot decode calls
- Chunked prefill prevents generation stalls — existing generating slots continue producing tokens while a new prompt is being prefilled
- Per-slot samplers preserve grammar/penalty state across the batched decode
- `:telemetry` enables monitoring without coupling to a specific metrics backend
- Request queue provides graceful degradation under load instead of immediate rejection
- The tick-based approach has inherent latency of one message round-trip per tick (negligible compared to forward pass time)

## Alternatives Considered

### Sub-batch processing in C++

Process the entire tick loop (batch construction, decode, sampling) in a single C++ NIF call. Rejected because it moves scheduling logic into C++, making it harder to debug and losing per-slot sampler flexibility from Elixir.

### One-NIF-per-tick with sampling in C++

Have a single NIF that builds the batch, decodes, and samples all tokens. Rejected because it loses the ability to use different sampler configurations per slot (grammar, temperature, etc.) managed from Elixir.

### Batch-accumulate-flush (ADR 005)

The original design accumulated requests over a `batch_timeout` window before flushing. The continuous batching approach is simpler (no timeout tuning) and lower latency (ticks fire immediately when work is available).
