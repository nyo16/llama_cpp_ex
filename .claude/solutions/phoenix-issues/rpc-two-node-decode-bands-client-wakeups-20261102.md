---
module: "ggml-rpc client (vendor/llama.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp), two-node layer split"
date: "2026-11-02"
problem_type: performance
component: rpc
symptoms:
  - "Two-node (RPC layer split) decode sits in one of two bands per client process: gpt-oss-120b ~44 or ~53 tok/s via llama-bench, ~38 or ~47 via the binding"
  - "The same worker serves consecutive clients at different speeds; restarting the worker does not settle it"
  - "Kernel times (nsys) are identical in both bands on both nodes; the extra ~4 ms per token is CPU-side"
root_cause: "Every RPC command is two thread handoffs: the caller blocks on a std::future while the dispatcher thread, woken from its queue, does the network round trip and wakes the caller. The dispatcher only busy-spins while a -sm tensor communicator holds it (busy_spin_acquire), so for a layer split each handoff is a futex wake of an idle core. On GB10's Cortex-X925 cores those wakes are slow enough to cost ~4 ms per token; on the A725 cores they are not (why is not established)."
severity: medium
tags: [rpc, ggml-rpc, dgx-spark, gb10, two-node, decode, cpu-affinity, latency]
related_solutions: [rpc-add-server-aborts-vm-dispatcher-20261102]
---

# Two-node decode bands: RPC client thread wake-ups

## Symptoms

`llama-bench --rpc ... -dev CUDA0/RPC0 -n 128` on two DGX Sparks gave 44 or 53
tok/s for gpt-oss-120b on different runs (single node 59). The docs had
blamed the worker for the spread.

## Investigation

- `nsys` on the worker in a fast and a slow session: same kernel times (118
  µs per MXFP4 expert matvec, 2.7 ms LM head), same per-token
  `cudaStreamSynchronize` total. Same on the client. So not GPU, not clocks.
- `taskset` matrix (worker cores x client cores, upstream worker restarted
  per run): the client on the A725 cores (0-4,10-14) was always fast; on the
  X925 cores (5-9,15-19) slow; unpinned mostly slow. `--poll 0`, `-t 1`,
  BEAM `+sbwt none` changed nothing.
- An env-gated hack spinning the dispatcher queue and the callers' future
  waits gave 57.6 on any cores: the wake-ups were the cost.

## Fix

- Upstream patch, `nyo16/llama.cpp` branch `rpc-client-spin-wait` (merged
  into `spark-all`): spin the dispatcher whenever a command arrived in the last
  `RPC_BUSY_SPIN_IDLE_TIME` (100 ms), and spin callers on their future for up
  to that long before blocking; `busy_spin_acquire/release` removed. 57.3-57.6
  tok/s on X925, A725 or unpinned; prefill unchanged.
- Without the patch (vendored llama.cpp): run the client on the A725 cores,
  `scripts/spark/remote.sh --little-cores` (+25% two-node decode through the
  binding, 37-38 -> 47 tok/s).

## Lessons

- A latency microbenchmark (1-byte RDMA ping-pong favours the X925) did not
  predict the end-to-end result; measure the decode loop itself.
- A bimodal per-process number with identical kernel times is a CPU-side
  handoff, not the GPU. Profile both nodes before blaming either.
- `pkill -f <pattern>` over ssh matches the ssh shell's own command line and
  kills it (exit 255); use `pkill -x` or a pid.
