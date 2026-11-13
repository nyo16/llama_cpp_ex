---
module: "LlamaCppEx.RPC / rpc_add_server NIF (c_src/llama_cpp_ex/llama_nif.cpp)"
date: "2026-11-02"
problem_type: runtime_error
component: rpc
symptoms:
  - "ggml-rpc.cpp:653: Failed to connect to 127.0.0.1:1, then SIGABRT (exit 134) from mix test on an LLAMA_RPC=1 build"
  - "RPC.add_server/1, RPC.ping/1 or Model.load(path, rpc_servers: [...]) with a dead endpoint kills the VM instead of returning {:error, :unreachable}"
  - "default (non-RPC) builds and macOS CI never show it: they answer :rpc_unsupported without touching ggml-rpc"
root_cause: "llama.cpp #18626 (d0132a680, 'rpc : implement event and async backend APIs') moved the connect and HELLO into rpc_dispatcher::start, which GGML_ABORTs on a parse failure, connect failure, HELLO failure or version mismatch. ggml_backend_rpc_add_server used to return nullptr for all of them. negotiate_hello also RPC_STATUS_ASSERTs when a non-RPC peer answers."
severity: high
tags: [rpc, ggml-rpc, abort, nif, upstream-regression, dgx-spark]
related_solutions: []
---

# RPC registration aborts the VM since llama.cpp #18626

## Symptoms

First `mix test` on a CUDA + `LLAMA_RPC=1` build on a DGX Spark aborted
within seconds at `rpc_test.exs`'s `RPC.add_server("127.0.0.1:1")`. The
published 0.8.66 does the same when built with `LLAMA_RPC=1`
(`mix deps.compile llama_cpp_ex --force` with `LLAMA_RPC=1 LLAMA_BACKEND=cuda`,
then `RPC.add_server("127.0.0.1:1")` → exit 134).

## Investigation

`git log -S "Failed to connect to %s" -- ggml/src/ggml-rpc/ggml-rpc.cpp`
pointed at d0132a680; its diff replaces every `return nullptr` in the old
`get_socket` with `GGML_ABORT` in `rpc_dispatcher::start`. CI never builds
RPC, so the bump that brought it in (v0.8.5x) passed.

## Fix

- NIF: `rpc_probe_hello/2` connects (IPv4, first-':' parse, `std::stoi`,
  exactly as upstream) with a 5 s timeout and sends a TCP-only HELLO (zeroed
  conn caps, so the worker never starts an RDMA upgrade for the probe), then
  checks major/minor like `negotiate_hello`. Anything short of that is
  `{:error, :unreachable}` before `ggml_backend_rpc_add_server` runs. The wire
  layout is pinned by `static_assert(RPC_PROTO_MAJOR_VERSION == 8)`.
- Upstream: branch `rpc-add-server-no-abort` on `nyo16/llama.cpp` makes
  `start()` return false, adds `try_get_dispatcher()` for registration and
  keeps aborting only on established connections.
- Tests: `test/rpc_test.exs` — a listener that closes, one that answers HTTP,
  and unparseable endpoints; each aborted the VM with the probe disabled.

## Lessons

- An RPC build is a different artifact; run `mix test` on one after every
  llama.cpp bump that touches `ggml-rpc`, not only `--include rpc_live`.
- `rsync -a` preserves source mtimes: after editing a file on a node, a sync
  that restores an older-mtime copy leaves make's `.o` newer than its source.
  `touch` the source before rebuilding.

## Update (llama.cpp 9daa0aa22, v0.8.69)

Upstream #30415 (`6122b566b`) makes `rpc_dispatcher::start` return false
instead of aborting, so `ggml_backend_rpc_add_server` returns `nullptr` for an
unreachable, unparseable or mismatched endpoint again. The probe stays: upstream
connects with no timeout and answers an already-registered endpoint from
`reg_map` without contacting it. The protocol is 10.0.0 now; the HELLO layout
did not change, so only the `static_assert` moved.
