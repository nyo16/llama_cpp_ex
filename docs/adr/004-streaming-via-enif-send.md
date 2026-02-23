# ADR 004: Streaming via enif_send

## Status

Accepted

## Context

Token generation in LLMs is inherently iterative — each token is produced one at a time. Users expect to see tokens as they're generated (streaming), not wait for the entire response.

We evaluated several approaches for streaming tokens from C++ to Elixir:

1. **Yield-based** — NIF returns one token at a time, called repeatedly from Elixir
2. **Callback-based** — NIF calls an Elixir function per token
3. **Message-based** — NIF sends Erlang messages via `enif_send`
4. **Port-based** — Separate process writes tokens to stdio

## Decision

We chose **message-based streaming via `enif_send`** from a dirty CPU scheduler.

## Rationale

### Why not yield-based?

Each NIF call would need to:
- Re-acquire the context state
- Perform one decode step
- Return the token

This adds per-token overhead from NIF entry/exit and makes it difficult to batch the prompt processing (prefill) step.

### Why not callback-based?

Erlang NIFs cannot directly call Elixir/Erlang functions. `enif_send` is the only safe way to communicate from a NIF back to the BEAM.

### Why enif_send?

- **Non-blocking**: The NIF runs the tight decode loop on a dirty scheduler while sending tokens as they're produced
- **Natural fit**: Erlang message passing is the standard concurrency primitive
- **Backpressure-free**: The mailbox buffers tokens naturally — no complex flow control needed for LLM-speed generation
- **Clean Elixir API**: Maps directly to `Stream.resource/3` with `receive` in the next function

### Implementation

```
Dirty CPU Scheduler           BEAM Scheduler
┌─────────────────┐          ┌──────────────────┐
│ generate_tokens  │          │ Stream.resource/3 │
│                  │          │                   │
│ loop:            │          │ receive:          │
│   decode(batch)  │─token──→│   {:token, text}  │──→ User
│   sample(ctx)    │          │   :eog            │──→ halt
│   enif_send(msg) │          │   :done           │──→ halt
│                  │          │                   │
└─────────────────┘          └──────────────────┘
```

The `generate_tokens` NIF:
1. Runs on `ERL_NIF_DIRTY_JOB_CPU_BOUND`
2. Performs prefill (prompt processing) in chunks of `n_batch` size
3. Enters the decode loop, producing one token per iteration
4. Sends `{ref, {:token, id, text}}` per token via `enif_send`
5. Checks `enif_send` return value — stops if the caller process is dead
6. Sends `{ref, :eog}` on end-of-generation or `{ref, :done}` on max_tokens

The Elixir side:
1. `Stream.resource/3` start function: tokenizes prompt, creates context + sampler, `spawn_link`s the NIF caller
2. Next function: `receive` on the ref, yields text chunks
3. Cleanup function: kills the generator process, flushes remaining messages

### Message format

```elixir
{ref, {:token, token_id, "text"}}  # Generated token
{ref, :eog}                         # End of generation (EOS token)
{ref, :done}                        # Max tokens reached
{ref, {:error, reason}}             # Error during generation
```

The unique `ref` per stream prevents interference between concurrent streams.

## Consequences

- Dirty scheduler thread is occupied for the duration of generation (acceptable — token generation is inherently serial per sequence)
- Messages accumulate in the mailbox if the consumer is slow (not a practical concern at LLM generation speeds of ~20-100 tokens/sec)
- Early stream termination requires killing the generator process and flushing messages
- The `spawn_link` ensures the generator dies if the stream consumer crashes
- `enif_send` return value check ensures the generator stops if the consumer dies
