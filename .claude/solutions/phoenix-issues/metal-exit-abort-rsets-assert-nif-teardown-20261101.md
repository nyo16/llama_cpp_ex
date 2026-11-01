---
module: "LlamaCppEx.NIF (c_src/llama_cpp_ex/llama_nif.cpp)"
date: "2026-11-01"
problem_type: runtime_error
component: nif_teardown
symptoms:
  - "ggml-metal-device.m:1041: GGML_ASSERT([rsets->data count] == 0) failed, then SIGABRT (exit 134) after the program finished"
  - "a model held in persistent_term, ETS or a process alive at System.halt/1 aborts the VM instead of exiting with its own status"
  - "0.8.62 exits 0 on the same script, 0.8.63 and 0.8.64 abort (precompiled Metal NIF)"
  - "a green mix test run on Metal exits 134 unless GGML_METAL_NO_RESIDENCY=1 is set"
root_cause: "The BEAM never runs NIF resource destructors at halt, so live models/contexts still own Metal buffers when exit(3) runs C++ static destructors; llama.cpp's Metal device is a function-local static whose destructor asserts no residency set is left. Residency sets are only compiled with a macOS 15+ SDK, and the move to macos-15 runners (v0.8.63) put them into the precompiled NIF."
severity: high
tags: [metal, nif, atexit, static-destruction, residency-sets, vm-halt, macos-sdk, fine]
related_solutions:
  - ".claude/solutions/deployment-issues/macos-14-runner-brownout-precompile-20261031.md"
---

# VM aborts at exit on Metal while a model is still referenced (#116)

## Symptoms

```
ggml/src/ggml-metal/ggml-metal-device.m:1041: GGML_ASSERT([rsets->data count] == 0) failed
3  ggml_metal_device_free
4  std::vector<unique_ptr<ggml_metal_device, ggml_metal_device_deleter>>::~vector()
5  __cxa_finalize_ranges
6  exit
7  beam.smp  erts_exit_vv
```

Repro: load a model, keep it referenced from any live process, let the
script end. Exit 134 on 0.8.63/0.8.64, 0 on 0.8.62. If the script process is
the only holder, nothing aborts: it dies before halt and the resource is GC'd.

## Investigation

1. **"llama.cpp changed"**: no. v0.8.62→v0.8.64 all pin `418f905a4`.
2. **Compare the artifacts**, which settled it. `otool -l` on the cached
   NIFs in `~/Library/Caches/elixir_make`:
   - 0.8.62: `sdk 14.5`, 0 residency-set selectors;
   - 0.8.63/0.8.64: `sdk 15.5`, `newResidencySetWithDescriptor:error:`,
     `addAllocation:`, `endResidency` and `requestResidency` present.

   `ggml-metal-device.m:20` defines `GGML_METAL_HAS_RESIDENCY_SETS` only when
   `__MAC_OS_X_VERSION_MAX_ALLOWED >= 150000`. Without it, `rsets->data` stays
   empty and the assert can never fire. That is why 0.8.62 worked, and why
   local builds (newer SDK) always needed `GGML_METAL_NO_RESIDENCY=1`.
3. **Ruled out `_exit` from an atexit hook**: it would lose the VM's exit
   status, which no atexit API exposes.
4. **Ruled out freeing without a guard**: the resources have no locks, so a
   dirty NIF mid-decode during halt would be a use-after-free.
5. **First fix had an ordering hole** (found by review, confirmed 3/3 SIGBUS,
   exit -10). The hook is registered once per allocation kind, so ggml statics
   first built *later* are destroyed *before* it. Case: a first model loaded
   with `devices: ["CPU"]`, then a Metal model and context. The Metal
   buffer-type statics (`ggml_backend_metal_buffer_type_shared/private`, whose
   `bufts`/`ctxs` vectors hold the names) are built after the hook, and
   `~llama_context` logs `ggml_backend_buft_name(buft)` → freed memory.

## Root Cause

Two facts combine:

- `erlang:halt` → `exit(3)` runs `atexit` handlers and C++ static destructors
  in reverse order of registration. The BEAM does not first destroy NIF
  resources still referenced from live processes, ETS or `persistent_term`.
- `ggml_metal_device_get` keeps devices in a function-local
  `static std::vector<ggml_metal_device_ptr>`. Its deleter calls
  `ggml_metal_rsets_free`, which asserts that every buffer's residency set was
  removed, i.e. that every model/context was freed first.

Changing the build SDK (runner image) silently switched that code path on for
every precompiled-NIF user.

## Solution

`c_src/llama_cpp_ex/llama_nif.{h,cpp}`, "Teardown at VM exit (#116)":

```cpp
// Resources register themselves; destructors free under the same lock.
explicit LlamaModel(llama_model* m) : ... { live::add(this); }
~LlamaModel() {
    std::lock_guard<std::recursive_mutex> lock(live::mutex());
    live::remove(this);
    release();               // frees + nulls; also called by the hook
}

// Every NIF: FINE_NIF plus an in-flight guard (seq_cst Dekker handshake).
#define LLAMA_NIF(name, flags) ... NifCall call; if (!call.entered) raise vm_exiting; ...

void free_live_resources_at_exit() {
    if (g_exiting.exchange(true)) return;
    // wait up to 5 s for g_in_flight == 0, else free nothing (no UAF)
    // under the lock: speculatives -> contexts with ctx_other -> contexts -> models
}

void arm_exit_hook(std::once_flag& once) {   // Apple only
    std::call_once(once, [] {
        // build Metal buffer-type statics BEFORE registering (CPU-first case)
        for each device whose reg name is "MTL":
            ggml_backend_dev_buffer_type(dev); ggml_backend_dev_host_buffer_type(dev);
        std::atexit(free_live_resources_at_exit);
    });
}
// armed after first model_load, context_create, speculative_init
```

Also:
- Generation loops, `cancel_abort_cb` and a model-load `progress_callback`
  return early on `g_exiting`, so a halt does not wait out a long call.
- Setting `progress_callback` replaces llama.cpp's default one, so it re-prints
  the progress dots via `log_callback`.

### Files Changed

- `c_src/llama_cpp_ex/llama_nif.h` — `live::` registry, `release()` on
  LlamaModel/LlamaContext/LlamaSpeculative, `has_ctx_other`
- `c_src/llama_cpp_ex/llama_nif.cpp` — registry, `NifCall`, `LLAMA_NIF` (all 67
  NIFs), exit hook, arming, `g_exiting` checks
- `test/vm_exit_test.exs` — child VM halts with live resources: model + context
  (`halt(7)` → 7), CPU-only then GPU model (`halt(5)` → 5), Server
  mid-generation (→ 0). It fails without the fix (134 / SIGBUS).
- `GGML_METAL_NO_RESIDENCY=1` removed from test_helper, test moduledocs,
  docs/release-guide.md, docs/performance.md, scripts/decision_compare.exs

Shipped in v0.8.65 (#118, #119). Verified with residency sets on: suites exit 0
(CI set ×8, release-guide set with MTP, default), and the e2e on the published
package passes 26/26 with four models referenced at halt.

## Prevention

- When the precompile runner image or SDK changes, diff the artifacts
  (`otool -l` → `sdk`/`minos`, `strings` for feature selectors) against the
  previous release. Feature gates keyed on SDK macros switch silently.
- Never keep a run-instruction env-var workaround for an abort without an
  issue tracking it. `GGML_METAL_NO_RESIDENCY=1` hid this from every local
  run, CI (Linux) and the v0.8.64 e2e smoke.
- Smoke the published package without workarounds and check the exit
  status, not just "N passed".
- Native code freed at process exit must be registered *after* every static
  it touches is constructed. Force lazily built statics (Metal buffer types)
  before `std::atexit`. Don't re-register per allocation: stateless
  `generate` makes a context per call, so the atexit list would grow without
  bound.
- New NIFs must use `LLAMA_NIF`, not `FINE_NIF`, or the exit hook can free
  under them.

## Related

- `.claude/solutions/deployment-issues/macos-14-runner-brownout-precompile-20261031.md`
  — the runner move that raised the SDK to 15.5
- Issue https://github.com/nyo16/llama_cpp_ex/issues/116, PR #118
- Iron Law #28 (iron-law-judge): `LLAMA_NIF` only; GPU-owning resources join
  `live::`; loops stop on `g_exiting`
