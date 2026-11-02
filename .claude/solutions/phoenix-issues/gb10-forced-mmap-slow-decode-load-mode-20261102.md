---
module: "LlamaCppEx.Model (lib/llama_cpp_ex/model.ex), model_load NIF"
date: "2026-11-02"
problem_type: performance
component: model_loading
symptoms:
  - "gpt-oss-120b decodes at ~48.5 tok/s through the binding on DGX Spark (GB10) while llama-bench and llama-server on the same build do ~57-59"
  - "the gap is per process: about one binding load in five decodes fast (57), and repeated runs inside one process never change state"
  - "nsys shows the same kernels (mul_mat_vec_q MXFP4/Q8_0) running ~22% longer in the slow processes"
  - "n_ctx, n_threads, sampler chain, BEAM busy-wait, ASLR, page-cache eviction: no effect"
root_cause: "Model.load/2 defaulted :use_mmap to true, which the NIF mapped to an explicit LLAMA_LOAD_MODE_MMAP. Upstream's default is LLAMA_LOAD_MODE_AUTO (#26081), which reads the file instead of mapping it when a device reports caps.mmap_support == false — ggml-cuda reports that for an integrated GPU, and GB10 is one. Forcing mmap overrode that and made the slow placement the common case."
severity: high
tags: [cuda, gb10, dgx-spark, igpu, mmap, load-mode, performance, benchmark-noise]
related_solutions: []
---

# Forced mmap on GB10 costs ~15% decode (load mode)

## Symptoms

Binding vs upstream on one Spark, gpt-oss-120b MXFP4, 128 greedy tokens:
`LlamaCppEx.stream/3` 48.5 tok/s, `llama-bench` tg128 58.9, `llama-server`
56.9. Qwen3-8B showed no gap, which hid it at first.

## Investigation

1. Binding overhead ruled out first: on Qwen3-8B the binding matched
   `llama-bench` and `llama-server` within 1-2%, single stream and 8-way.
2. Per-process bimodality: the same script gave 48.5 in most processes and
   57 in a few; inside one process every context agreed.
3. `nsys profile -t cuda --cuda-graph-trace=node` on both: identical kernel
   list and launch shapes, but the memory-bound matvecs ~22% slower in the
   slow processes. So GPU-side, not CPU, sampling or messaging.
4. Ruled out: `n_threads`, sampler chain, `n_ctx`/`swa_full`, BEAM `+sbwt none`,
   `setarch -R`, THP (weights are not in RSS), page-cache eviction via
   `posix_fadvise(DONTNEED)`.
5. `use_mmap: false` and `use_direct_io: true`: 6/6 processes fast. Upstream's
   `llama_model_default_params().load_mode` is `AUTO`; `src/llama-model.cpp`
   turns mmap off under AUTO when a device lacks `mmap_support`, and
   `ggml-cuda.cu` sets that false for `GGML_BACKEND_DEVICE_TYPE_IGPU`.

## Fix

`Model.load/2` resolves the three flags itself (`resolve_load_mode/3`) and
passes the enum to the NIF; with none of them given the result is
`LLAMA_LOAD_MODE_AUTO`. Explicit flags resolve exactly as before. Metal, CPU
and discrete CUDA report `mmap_support`, so AUTO maps there as the old
default did. After: 5/5 processes at 57.0-57.2 tok/s; forced mmap still
bimodal (48.5 / 57).

## Lessons

- Never override an upstream default with a binding default that looks
  equivalent — `use_mmap: true` was "the default" only until upstream grew a
  smarter one.
- On GB10, a single decode measurement is one sample of a bimodal process.
  The ~17% dip also appeared occasionally in `llama-bench` itself and on the
  RPC worker per client session, independent of mmap; re-run before
  believing a number.
- Test: `test/option_forwarding_test.exs` "load-mode flags resolve to
  llama.cpp's enum, auto when none is given".
