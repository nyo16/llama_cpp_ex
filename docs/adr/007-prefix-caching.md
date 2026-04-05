# ADR 007: Prefix Caching (Same-Slot KV Reuse)

## Status

Accepted

## Context

The continuous batching server (ADR 006) clears the KV cache for every slot on every request via `memory_seq_rm(ctx, seq_id, 0, -1)`. This means repeated requests with shared prefixes — common in multi-turn chat, few-shot prompting, and system prompt reuse — must re-compute the same KV entries each time.

llama.cpp's own server implements per-slot prompt caching: when a slot finishes and gets a new request, it checks the common prefix between the old and new token sequences and skips prefill for the matching portion. SGLang takes this further with a global radix attention tree, but that requires page-level KV cache management that llama.cpp doesn't expose.

## Decision

Implement **same-slot KV reuse** — after a slot completes a request, keep its KV cache alive and store the full token history (prompt + generated tokens). When the next request arrives:

1. Compute `common_prefix_length(new_tokens, cached_tokens)` — a simple zip-and-compare
2. If match > 0: call `memory_seq_rm(ctx, seq_id, n_match, -1)` to trim only the non-matching suffix, then set `prefill_pos = n_match` to skip the matched portion
3. If no match: clear fully with `memory_seq_rm(ctx, seq_id, 0, -1)`

Additionally, **prefix-affinity slot selection**: when acquiring an idle slot, prefer the one whose cached tokens have the longest prefix match with the incoming request's tokens. This maximizes cache hits when multiple slots are available.

### Configuration

- `cache_prompt: false` (default) — identical to pre-caching behavior
- `cache_prompt: true` — enable same-slot KV reuse (opt-in)

### Telemetry

New measurements in `:llama_cpp_ex, :server, :request, :done`:
- `prefix_cache_tokens` — number of tokens skipped via cache
- `prefix_cache_ratio` — ratio of cached to total prompt tokens

## Performance

Benchmarked with Qwen3-0.6B-Q8_0 on Apple M1 Max, simulating 4-turn multi-turn chat where each request extends the previous:

| Scenario | Average | Median |
|---|---|---|
| WITH prefix cache | 487ms | 452ms |
| WITHOUT prefix cache | 597ms | 591ms |
| **Improvement** | **1.23x** | **1.31x** |

The improvement grows with longer shared prefixes (system prompts, multi-turn history).

## Consequences

- Multi-turn chat TTFT improves significantly — the system prompt and prior turns don't need re-computation
- Memory usage is unchanged — KV cache was already allocated; we just avoid clearing it
- Prefix-affinity slot selection adds a linear scan of idle slots (negligible for typical `n_parallel` values)
- Generated tokens are tracked in a reverse list (`O(1)` prepend) and reversed once on slot reset

## Alternatives Considered

### Cross-slot prefix sharing via token trie (SGLang-style)

A global radix tree mapping token sequences to dedicated cache sequence IDs, using `memory_seq_cp` to copy KV cache between slots. Deferred because:
- Requires additional sequence IDs beyond `n_parallel` (API change)
- KV cache capacity management across slots + cache sequences is complex
- Same-slot reuse captures the most common case (multi-turn chat)
- Can be added as a future enhancement building on this foundation

### No caching (current behavior)

Always clear KV cache. Simple but wastes compute on repeated prefixes. The `cache_prompt: false` option preserves this behavior for users who need it.
