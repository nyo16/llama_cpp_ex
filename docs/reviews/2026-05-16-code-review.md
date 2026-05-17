# llama_cpp_ex code review — findings

**Scope:** MTP speculative decoding + `LlamaCppEx.Server` + full NIF surface.
**Method:** Static read of all 1620 lines of `c_src/llama_cpp_ex/llama_nif.cpp`,
the 939-line `lib/llama_cpp_ex/server.ex`, the 300-line `lib/llama_cpp_ex/mtp.ex`,
plus targeted reads of `lib/llama_cpp_ex/context.ex` and `mix.exs`. Confirmed
suspect findings with 6 dynamic experiments against `Qwen3.5-0.8B-UD-Q4_K_XL`
and `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL` from `~/Downloads/`.

---

## Critical (crash / data loss / correctness)

### C1. `Context.create(n_ctx: 0)` aborts the entire BEAM

`vendor/llama.cpp/src/llama-context.cpp:2188` asserts `GGML_ASSERT(n_outputs >= 1)`,
which is reached when `n_ctx == 0`. `lib/llama_cpp_ex/context.ex:74` does
`Keyword.get(opts, :n_ctx, 2048)` with no `> 0` validation. Anyone calling
`Context.create(model, n_ctx: 0)` takes the whole VM down with SIGABRT
(exit 134). Confirmed in `case_n_ctx_zero.exs`.

**Fix:** Reject `n_ctx <= 0` in Elixir before the NIF call. Same for `n_batch`,
`n_ubatch`, and the MTP `n_rs_seq`. Optional but cheap: add a Rust-style
"validate params, then call" gate inside `context_create` in C++ before
`llama_init_from_model`.

### C2. `MTP` greedy generation is non-deterministic across calls on the same `%MTP{}`

Two `MTP.generate(mtp, prompt, max_tokens: 50, temp: 0.0, top_k: 1, seed: 42)`
calls on a freshly-initialised `%MTP{}` produced **different** outputs (one
truncated 3 tokens earlier than the other). Confirmed by `exp7_mtp_unit.exs`.

`llama_nif.cpp:1007-1008` clears the KV cache on both contexts at the start of
every `generate_mtp_tokens` call, but the `common_speculative*` state — which
includes the MTP head's internal `pending_h` / `verify_h` hidden-state buffers
— is not re-primed between calls. Whatever residual state survives biases the
next call's drafts, and the divergence compounds.

Workarounds:
- Re-init the `%MTP{}` for every generation (defeats the documented "reuse"
  ergonomics).
- In the NIF, call something equivalent to `common_speculative_reset(sp.spec)`
  (or destroy+recreate `sp.spec`) at the top of `generate_mtp_tokens`.

This contradicts the `mtp.ex` moduledoc that explicitly tells users to "Reuse
the same `%MTP{}` value across calls to `stream/3` / `generate/3` to avoid
rebuilding the contexts; KV caches are cleared on each call."

### C3. `Server.stream` (and `Server.generate`) leak slots when the caller dies

`lib/llama_cpp_ex/server.ex` has **zero** `Process.monitor` calls. When a stream
consumer halts early (or dies), the slot keeps generating up to `max_tokens`,
sending tokens to a dead pid via `send/2`. New requests get queued behind the
zombie.

Confirmed by `exp2_caller_death.exs`:
- Consumer pulled 3 tokens, then exited at t≈200 ms.
- Slot stayed `active: 1` for **4708 ms** while max_tokens=512 finished.
- A concurrent `Server.generate("Hello", max_tokens: 8)` sat queued the entire time.

**Fix:** `handle_call({:stream, ...})` should `Process.monitor(pid)` and
`handle_info({:DOWN, ref, ...})` should reset the matching slot. Same for
`handle_call({:generate, ...})` (which is harder — the caller's GenServer.call
times out independently, but the slot doesn't know).

### C4. `MTP.stream_events` / `LlamaCppEx.stream` halt does not cancel the NIF

Both wrap the NIF call in `spawn_link`, then on `Stream.resource`'s end-fun do
`Process.unlink(gen_pid); Process.exit(gen_pid, :kill)`. The link signal kills
the *wrapper* BEAM process, but the NIF call is running on a dirty scheduler
thread — that thread is not interruptible by BEAM process death. The NIF
continues until it next checks `enif_send` to the *consumer* pid (not gen_pid),
which is still alive, so `enif_send` returns 1 (success) and the NIF keeps
generating tokens into the consumer's mailbox.

`mtp.ex:184` and `lib/llama_cpp_ex.ex:231/375/492` are all affected.

**Fix:** Pass `gen_pid` as the `caller_pid` to the NIF rather than the user's
consumer pid; then the NIF's existing `enif_send`-return check correctly
detects death. The consumer-pid forwarding can happen in the wrapper process
itself. Alternative: add an explicit cancellation token (atomic flag in the
resource that the NIF polls).

This finding makes `Stream.take(n)` on a streaming generator *waste CPU* —
the NIF runs to completion regardless of how many tokens the consumer
actually pulled.

---

## High (perf / scheduler / leak)

### H1. `sampler_sample` and `sampler_sample_at` are not on the dirty scheduler

`llama_nif.cpp:378` and `:758` both call `llama_sampler_sample()` from a
regular BEAM scheduler. Verified empirically in `exp1_sampler_stall.exs` with
4 workers hammering the call on Qwen3.5-0.8B (151k vocab):

| Phase | Heartbeat lag p99 | Heartbeat lag max |
|-------|-------------------|-------------------|
| Idle  | 2.7 ms            | 4.4 ms            |
| Under load | **31.7 ms** | **51.3 ms** |

p99 for a single `sampler_sample` is 178 µs and the max observed was **62 ms** —
a single call that holds an OS scheduler thread for 62 ms is an unambiguous
scheduler-etiquette violation.

`sampler_sample_at` is even more impactful: every continuous-batching tick
runs it once per active slot. At n_parallel=4 with ~60 ticks/sec that's
240 calls/sec on the BEAM scheduler.

**Fix:**
```c++
FINE_NIF(sampler_sample,    ERL_NIF_DIRTY_JOB_CPU_BOUND);
FINE_NIF(sampler_sample_at, ERL_NIF_DIRTY_JOB_CPU_BOUND);
```

### H2. `chat_apply_template_jinja` is not on the dirty scheduler

`llama_nif.cpp:835`. Renders a Jinja template through `common_chat_templates_apply`,
which on long histories and templates with `{% for %}` loops can blow past
1 ms. Particularly relevant because the Server's `LlamaCppEx.chat_completion/3`
path calls this from inside `handle_call`, blocking the GenServer for the
duration.

**Fix:** `FINE_NIF(chat_apply_template_jinja, ERL_NIF_DIRTY_JOB_CPU_BOUND);`

### H3. `tokenize` is not on the dirty scheduler

`llama_nif.cpp:155`. `llama_tokenize` is O(input bytes). With a 100 KB chat
history (50k+ tokens, confirmed in `case_tokenize_100kb.exs`) this is multi-ms.
The Server's `handle_call({:generate, prompt, ...})` runs `Tokenizer.encode`
synchronously inside the GenServer.

**Fix:** `FINE_NIF(tokenize, ERL_NIF_DIRTY_JOB_CPU_BOUND);`
For extra safety, also mark `detokenize` and `speculative_init` (the latter
does a probe decode internally during `common_context_can_seq_rm`).

### H4. `Server` documents `:max_queue` but never enforces it

`lib/llama_cpp_ex/server.ex:106` documents:
> `:max_queue` - Max queued requests. `0` for unlimited. Defaults to `0`.

`grep` confirms `max_queue` appears only in the docstring — `init/1` (line 258)
never reads it. Under a thundering-herd scenario (e.g. an HTTP endpoint that
fans out into `Server.generate`), the internal `:queue.t()` grows unbounded
until the BEAM OOMs.

**Fix:** Read `:max_queue` in `init/1`, store on state, and in
`enqueue_request` return `{:error, :queue_full}` rather than appending when
the limit is reached.

### H5. `decode_batch` resets the sampler before every token

`llama_nif.cpp:657` calls `llama_sampler_reset(sampler->sampler)` before every
`llama_sampler_sample` in the per-entry loop. This means **repetition penalty,
grammar samplers, and any other history-dependent samplers do nothing on this
code path** — each per-position sampler call starts from an empty history.

This NIF is not the Server's hot path (the Server uses `batch_eval` +
`sampler_sample_at` per slot) but it's part of the public NIF surface and
likely used by user code that does manual batched inference.

**Fix:** Decide on semantics. Either:
- Per-sequence samplers passed in (proper continuous-batching design), or
- Drop the `_reset` (per-position state accumulates across the batch — wrong
  for unrelated sequences), or
- Document that `decode_batch` is best-effort for stateless sampling.

### H6. `accumulated_text` string concat per slot per token

`lib/llama_cpp_ex/server.ex` stores per-slot `accumulated_text` as a string,
appended to on every generated token. For `max_tokens: 4096` that's 4096
concatenations of an O(N)-sized binary, which is O(N²) work.

Not in the empirical results because tests used small `max_tokens`. Worth
fixing as iolist: append to a list, `IO.iodata_to_binary/1` at finalize time.

### H7. `Server.tick` schedules immediately, no idle gap

`maybe_schedule_tick` re-arms the tick whenever any slot is non-idle. There's
no explicit "yield to the scheduler" between ticks — the GenServer runs flat
out under load. Under saturated continuous-batching this is correct (you want
maximum throughput), but for *long-running individual generations with no
queue*, this means the GenServer hot-loops between `:tick` messages back to
itself, starving its own mailbox for control messages (`:get_stats`, etc.).
Probably not visible in practice because each tick takes 10–50 ms of decode
time and that's plenty of yielding. Note for later.

---

## Medium (ergonomics / footguns)

### M1. `LlamaCppEx.NIF.context_can_seq_rm/1` silently clears KV memory

`llama_nif.cpp:419-433` documents the side effect in a code comment ("calling
this clears the context's KV memory as a side effect — only call once at init
time"). The Elixir wrapper `LlamaCppEx.NIF.context_can_seq_rm/1` is part of
the public NIF module — anyone reading it would assume "query for what kinds
of seq_rm are supported" and get a destroyed cache for their trouble.

Confirmed by `case_side_effect_demo.exs`: after `decode("Hello world")`,
`memory_seq_pos_max` returned 1; after `context_can_seq_rm`, it returned -1.

**Fix:** Don't expose it as a public NIF call. Either inline the probe inside
`speculative_init` and the `Server`'s init (the only two real callers) and
make the NIF private, or rename to `__probe_seq_rm_kind_unsafe__/1` and add
a moduledoc warning. The library already uses it correctly internally; nobody
outside should need it.

### M2. `Context.create(n_ctx: 1_000_000)` accepted silently

No validation against `Model.n_ctx_train`. Returns `{:ok, _}` even though the
context will OOM or assert on the next decode. Confirmed in `case_n_ctx_huge.exs`.

**Fix:** Warn (or reject) when `n_ctx > Model.n_ctx_train(model) * 4` (or
similar generous bound), or at minimum mention this in the `Context.create`
moduledoc.

### M3. `Tokenizer.decode([-1])` and OOV tokens raise `unknown exception thrown within NIF`

`llama_nif.cpp:179` (`detokenize`) throws `std::runtime_error("detokenization
failed")` on negative indices and out-of-vocab tokens. Fine returns this as a
RuntimeError with the generic message instead of a clean `{:error, _}`.
Confirmed in `case_decode_negative_token.exs` and `case_decode_oov_token.exs`.

**Fix:** Return `fine::Error(...)` from the C++ side; the wrapper already
expects `{:ok, _} | {:error, _}` for other paths.

### M4. `generate(max_tokens: 0)` and `generate(max_tokens: -1)` return `{:ok, ""}`

Silent rather than `{:error, :invalid_max_tokens}`. Confirmed.
Minor — easy to fix in `LlamaCppEx.generate` and `Server.generate`.

### M5. Sampler params not validated

`top_k: 0` (disabled), `top_k: 1, temp: 0.0` (greedy), `temp: -1.0`, `temp: 100.0`,
`top_p: 0.0` — all accepted by `Sampler.create` without complaint. Some are
legitimate edge cases (top_k:0 disables the top-k filter; temp:0 with top_k:1
is greedy and works), some are nonsense (temp:-1.0). Documenting the
contract in the `Sampler.create` moduledoc would help.

### M6. `Context.create` opts list is 25 keywords deep

Documented well in `lib/llama_cpp_ex/context.ex:11-69`, but the implementation
is 90 lines of `Keyword.get` boilerplate. A nested-keyword grouping (e.g.
`rope: [...], yarn: [...], kv: [type_k:, type_v:]`) would scale better as the
upstream API grows. Style/maintenance, not a bug.

### M7. `mix.exs @version` lags the latest tag

`@version "0.8.5"` while tags exist for `v0.8.6`, `v0.8.7`, `v0.8.8`. Each
release commit on master bumps `checksum.exs` (via release automation) but
doesn't touch `mix.exs`. This works for the precompiled-binary distribution
path but would conflict if anyone runs `mix hex.publish` manually from master
(it would try to publish 0.8.5 which is already on Hex).

**Fix:** Either bump `@version` in the same commit as the tag, or make the
release automation bump both. The user's memory notes that release-guide.md
is out of date — this is the missing step.

---

## Low (style / docs)

### L1. `embed_decode` silently wipes KV memory

`llama_nif.cpp:491` calls `llama_memory_clear(...)` at the start of every
`embed_decode`. Correct for embedding mode but means `embed_decode` is not
safe to interleave with any other decode on the same context. Worth a one-liner
in `Embedding.embed/2`'s moduledoc.

### L2. `LlamaCppEx.init/0` returns bare `:ok`

Every other call returns `{:ok, _} | {:error, _}`. `init/0` returns `:ok`.
Trivial inconsistency.

### L3. The MTP module's "keep sampler alive in closure" comment is probably unnecessary

`mtp.ex:196-199` keeps the sampler struct in the Stream state to prevent GC.
The `fine` resource holding pattern (`fine::ResourcePtr<LlamaSampler>` in
`LlamaSpeculative`) should retain its own refcount; the Elixir-side reference
isn't load-bearing once the NIF has its hands on the resource. Defensive code,
not wrong — but worth checking if it's actually needed.

### L4. `LlamaCppEx.MTP` `init/2` constrains MTP to `n_parallel = 1`

The moduledoc mentions this. The struct doesn't enforce it (Context.create
isn't passed `n_seq_max`). A test that tries `n_parallel=4` would presumably
fail at decode time — could be friendlier with a clearer error.

---

## Test-coverage gaps

Confirmed missing in the test suite (per the Explore-agent inventory plus
this review's experiments):

- **No caller-death tests for `Server.stream`** — C3 would have been caught
  by an integration test that kills a stream consumer and checks slot
  recovery.
- **No NIF-cancellation tests** — C4 would have been caught by an early-take
  test on `LlamaCppEx.stream`.
- **No MTP reuse determinism tests** — C2 would have been caught by an
  exp7-style "two greedy generations on the same `%MTP{}` must match" test.
- **No edge-case invariants in test/llama_cpp_ex_test.exs** — empty prompt,
  `max_tokens=0/-1`, `n_ctx=0`, OOV tokens, `tokenize` on a 100 KB blob.
- **No `max_queue` enforcement test** — would have flagged H4 immediately.
- **No multi-Server BEAM tests** — start 2+ Servers concurrently, sanity-check
  no shared state.
- **No NIF dirty-scheduler-classification check** — could be an automated
  test that scans `llama_nif.cpp` for `FINE_NIF(name, 0)` and matches each
  against an allowlist of known-fast operations. Caught H1/H2/H3 cheaply.

---

## Existing test suite drift

Running `mix test` against the three local models (`Qwen3.5-0.8B-UD-Q4_K_XL`,
`Qwen3-Embedding-0.6B-f16`, `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL`) gave **148 / 150
passing**. Both failures are in the MTP block of `test/llama_cpp_ex_test.exs`
and are *not* regressions — they're either out-of-sync with a recent code
change or set on different hardware.

### TS1. `test/llama_cpp_ex_test.exs:962` is contradicted by the current code

```elixir
# the default target ctx should report 0 (no rollback).
assert LlamaCppEx.Context.n_rs_seq(mtp.mtp_ctx) >= mtp.n_draft
```

`lib/llama_cpp_ex/mtp.ex:97-100` was changed to create the draft context with
`n_rs_seq: 0`:

```elixir
# Match upstream server: MTP draft context is created with n_rs_seq=0.
# The MTP impl handles state rollback internally via cached hidden
# states (pending_h / verify_h), not via recurrent-state snapshots.
draft_opts = Keyword.merge(base_ctx_opts, ctx_type: :mtp, n_rs_seq: 0)
```

So the assertion `>= mtp.n_draft` (e.g. ≥ 3) can never hold for the draft
context; the test should be either:

```elixir
assert LlamaCppEx.Context.n_rs_seq(mtp.mtp_ctx) == 0
```

…or deleted, since the property it's checking no longer exists.

Implication beyond this one test: when the `n_rs_seq` semantics changed,
nobody re-ran the suite, so the contradiction sat undetected. Worth a
quick `grep` through the rest of the MTP tests for anything similar
(particularly anything that exercises the rollback path).

### TS2. `test/llama_cpp_ex_test.exs:983` acceptance threshold too aggressive for Apple Silicon

```elixir
assert stats.acceptance_rate > 0.3,
       "acceptance_rate=#{stats.acceptance_rate} (expected > 0.3); ..."
```

On `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL` running on M1 Max, observed
`acceptance_rate = 0.218` (38 / 174 drafts accepted, 97 tokens emitted,
19.6 tok/s). Matches the Apple Silicon MTP profile already documented in
upstream issues #23011 (Metal batched-decode scaling) and #23114 (Metal
MTP drafting optimization, still open).

This test does flag a real wiring break if it drops to single-digit
acceptance (which is what upstream #23011 sees: 95.6% on CUDA → 1.93 tok/s
on M-series at the same nominal acceptance). At 21.8% the wiring is fine,
just hardware-bound.

Options:
- Lower the threshold to ~0.15 on Apple, parameterize by `:os.type/0`, or
- Skip this assertion on macOS arm64 with `@tag :skip_metal` and rely on a
  CUDA CI runner for the strict check, or
- Replace with a sanity-only check (e.g. `acceptance_rate > 0.05` to catch
  total wiring breaks while not relying on hardware-specific upper bounds).

---

## Recommendations: triage priority

If you want to ship fixes in waves:

**Wave 1** (one PR, no behavior change): C1, M3, H1, H2, H3 — pure additions
of `> 0` validation and dirty-flag annotations. Low risk, immediate impact.

**Wave 2** (one PR, behavior change but well-bounded): C3 — add
`Process.monitor` for `stream_pid` and `from` in `Server`. New
`handle_info({:DOWN, ...})` reclaims the slot.

**Wave 3** (one PR, NIF-side correctness work): C4 — restructure the
streaming NIFs to send through `gen_pid` so death detection works. Also
M1 (privatize `context_can_seq_rm`) — single-file API cleanup.

**Wave 4** (investigation required): C2 — needs an upstream dive into
`common_speculative_*` to find where the residual state lives, plus a
`reset` call or a re-init. Worth filing upstream as an issue too.

**Wave 5** (the kitchen sink): H4 (max_queue), H5 (decode_batch sampler
reset), M2 (n_ctx_train sanity), M4/M5 (param validation), M7 (mix.exs
@version drift), H6 (iodata accumulator), and the test-coverage gaps.

---

## NIF dirty-classification table (full)

For reference — the entire `llama_nif.cpp` audit:

| NIF | Line | Dirty? | Verdict |
|-----|------|--------|---------|
| backend_init | 29 | no | OK (trivial) |
| backend_free | 35 | no | OK (trivial) |
| model_load | 67 | IO | OK |
| model_n_ctx_train | 72 | no | OK |
| model_n_embd | 77 | no | OK |
| model_desc | 84 | no | OK |
| model_size | 89 | no | OK |
| model_n_params | 94 | no | OK |
| model_chat_template | 103 | no | OK |
| vocab_n_tokens / vocab_bos / vocab_eos / vocab_is_eog | 110–125 | no | OK |
| tokenize | 155 | no | **H3 hazard** |
| detokenize | 179 | no | borderline |
| token_to_piece | 197 | no | OK |
| context_create | 290 | CPU | OK |
| context_n_ctx / n_seq_max / n_rs_seq | 295/474/300 | no | OK |
| sampler_init | 360 | no | borderline (grammar compile) |
| sampler_accept / sampler_reset | 366/372 | no | OK |
| **sampler_sample** | 378 | **no** | **H1 hazard** |
| decode | 399 | CPU | OK |
| memory_clear / seq_rm / seq_cp / seq_keep / seq_pos_max | 407/417/448/458/467 | no | OK |
| **context_can_seq_rm** | 433 | no | M1 (side-effect footgun) |
| embed_decode | 514 | CPU | OK |
| get_embeddings | 564 | no | OK |
| prefill | 608 | CPU | OK |
| decode_batch | 681 | CPU | OK (but H5 sampler-reset bug) |
| decode_token | 710 | CPU | OK |
| batch_eval | 746 | CPU | OK |
| **sampler_sample_at** | 758 | **no** | **H1 hazard** |
| chat_apply_template | 802 | no | OK |
| **chat_apply_template_jinja** | 835 | **no** | **H2 hazard** |
| speculative_init | 887 | no | borderline |
| speculative_stats / print_stats | 952/958 | no | OK |
| generate_mtp_tokens | 1412 | CPU | OK |
| generate_tokens | 1535 | CPU | OK |
| generate | 1601 | CPU | OK |
| json_schema_to_grammar_nif | 1615 | no | borderline (bounded by schema) |

Three confirmed hazards (sampler_sample, sampler_sample_at, chat_apply_template_jinja)
plus three borderline candidates (tokenize, sampler_init w/ grammar,
speculative_init).
