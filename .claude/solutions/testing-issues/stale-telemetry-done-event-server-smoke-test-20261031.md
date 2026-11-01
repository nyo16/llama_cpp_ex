---
module: "LlamaCppEx.ServerSmokeTest"
date: "2026-10-31"
problem_type: test_failure
component: testing
symptoms:
  - "test halting a stream early cancels generation and frees the slot: ** (MatchError) no match of right hand side value: {%{prompt_tokens: 39, generated_tokens: 4, ...}, %{stop_reason: :max_tokens, ...}}"
  - "fails only on CI (CPU, seed 489492) on a docs-only PR; passes locally with the same seed"
root_cause: "A :telemetry handler is global, and the test's collector forwarded the first [:llama_cpp_ex, :server, :request, :done] event from any server. The previous test's server was still alive: ExUnit does not wait for a test's linked processes to stop, and Server.finish_slot/3 replies to the caller before emitting :done, so that server's stale event arrived after the next test attached its handler."
severity: medium
tags: [telemetry, flaky-test, exunit, global-handler, linked-process, server, race]
---

# Flaky server smoke test: stale `:done` telemetry from the previous test's server

## Symptoms

The CI Inference smoke on #115, a docs-only PR, failed:

```
1) test halting a stream early cancels generation and frees the slot (LlamaCppEx.ServerSmokeTest)
   ** (MatchError) no match of right hand side value:
   {%{prompt_tokens: 39, generated_tokens: 4, ...}, %{stop_reason: :max_tokens, server: #PID<0.1528.0>, ...}}
   code: {%{generated_tokens: n}, %{stop_reason: :cancelled}} = next_telemetry()
```

The test streams "Write an endless story:" (about 5 prompt tokens,
`max_tokens: 400`), so a 39-token, `max_tokens` (4) event cannot be its own.

## Investigation

1. **"Cancel lost the race with generation"**: no. Stop reason `max_tokens`
   after 4 tokens doesn't match this request's `max_tokens: 400`.
2. **Find the 39-token request**: the preceding test, "chat_completion/3
   propagates a server-side generation error". Its last call is
   `chat_completion(server, messages, max_tokens: 4)`, a chat-templated
   prompt of 39 tokens on SmolLM2.
3. **CI log order**:
   - 54.311: handler attached by the new test's process, `#PID<0.1529.0>`;
   - 54.314: `[debug] slot 0 done (max_tokens): 39 prompt tokens`;
   - 54.514: the real `cancelled` event.
4. Locally (Metal) the same seed passed with and without the fix. On faster
   hardware the old server usually emits before the next test attaches, so it
   only reproduces on timing.

## Root Cause

- `finish_slot/3` calls `GenServer.reply` first, then `Logger.debug` and
  `:telemetry.execute(... :done ...)`. The caller can return, its test can
  finish, and the next test can start in between.
- The server is `start_link`ed to the test process and traps exits.
  ExUnit does not wait for linked processes to terminate after a test, so the
  old server keeps running briefly.
- `:telemetry.attach/4` handlers are global. The collector sent every event to
  the test process with no filter:

```elixir
fn _event, measurements, metadata, _ ->
  send(parent, {:telemetry, measurements, metadata})
end
```

## Solution

Pin the server under test. Every server telemetry event carries
`server: self()` in its metadata.

```elixir
defp attach_collector(server, event) do
  parent = self()
  handler_id = {__MODULE__, self(), event}

  :telemetry.attach(handler_id, event,
    fn
      _event, measurements, %{server: ^server} = metadata, _ ->
        send(parent, {:telemetry, measurements, metadata})
      _event, _measurements, _metadata, _ ->
        :ok
    end, nil)

  on_exit(fn -> :telemetry.detach(handler_id) end)
end
```

The assertions are unchanged; only foreign events are dropped.

### Files Changed

- `test/server_smoke_test.exs` — `attach_collector/2` filters on `server`; 4
  callers updated
- `test/llama_cpp_ex_test.exs` — the telemetry test's three handlers pin
  `server`, detach in `on_exit` (previously only on success), and assert
  `metadata.server == server`

Shipped in #117 (v0.8.64). CI green afterwards.

## Prevention

- Every test-side `:telemetry.attach` must filter on an identity the test owns
  (`server`, `ref`, `seq_id`) and detach in `on_exit`, not at the end of the
  test body.
- Don't assume a linked process is gone when the next test starts. Stop it
  explicitly or filter its output.
- A flake on an unrelated PR: read the event payload before re-running. Here
  the prompt-token count identified the sender.

## Related

- `.claude/solutions/phoenix-issues/metal-exit-abort-rsets-assert-nif-teardown-20261101.md`
  — same "process outlives its test" family (a live model at halt)
- Iron Law #29 (iron-law-judge): test telemetry handlers pin an owned identity
  and detach in `on_exit`. The last unfiltered handler (hybrid-cache test in
  `test/llama_cpp_ex_test.exs`) was fixed when this law was added.
