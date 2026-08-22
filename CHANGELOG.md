# Changelog

## Unreleased

DGX Spark (GB10) support: a silent ARM code-generation bug fixed, the ggml RPC
backend wired up so a model can span two machines, and a measured runbook for
both configurations in [docs/dgx-spark.md](docs/dgx-spark.md).

llama.cpp bumped to [`b10582`](https://github.com/ggml-org/llama.cpp/releases/tag/b10582)
(`e85caa81e`), by way of b10435, which brought Qwen 3.8 in under the existing
`qwen35` architecture, and MTP support for its target/sidecar split (see Added).

Verified on macOS (Metal, no RPC) at `e85caa81e`: **428 passed, 149 excluded**
for the default run, and with real models from the tags that need them —
**548 passed** `--include smoke --include embeddings` (Qwen3.5-0.8B-UD-Q4_K_XL,
Qwen3-Embedding-0.6B-f16), **439 passed** `--include mtp`
(Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL), **434 passed** `--include mtp_sidecar`
(Qwen3.8-27B-Q4_K_M plus its `mtp-*-Q4_0` head), **438 passed**
`--include slow`. On a DGX Spark (CUDA 13.0, `sm_121a`), measured at b10435:
**428 passed**.

### Fixed

- **The ggml CPU backend was silently compiled at base ARMv8-A on GB10.** GCC
  13.3 predates Cortex-X925/A725 and rejects `-mcpu=cortex-x925`, so ggml's
  `-mcpu=native` probe degraded to the base architecture behind a soft CMake
  warning and a zero exit status. Measured on the emitted `libggml-cpu.a`:
  **0 `sdot`, 0 `smmla`, no SVE**, versus 1134 / 370 / present once the
  architecture is named. Those are the Q4/Q8 quantized matmul kernels. New
  `LLAMA_CPU_ARM_ARCH` names it explicitly.
- **The build could ship a NIF that did not match its own flags.** Toggling a
  build variable changes `CXXFLAGS`, `LDFLAGS` or `CMAKE_FLAGS` while touching no
  source file, so make kept a stale object, a stale link, or a stale
  `CMakeCache.txt`. Observed twice as
  `Function not found 'Elixir.LlamaCppEx.NIF':model_load/11` and once as an RPC
  build reporting `:rpc_unsupported` at runtime — all silent, all looking like
  Elixir bugs. The cmake configure, the compile and the link now depend on a
  configuration stamp whose name hashes all three flag sets.
- **Every `MIX_ENV` shares one `llama_cpp_ex_nif.so`, and they were overwriting
  each other.** Mix symlinks `priv/` into each environment's build tree, so dev,
  test and bench write one artifact while keeping separate objects and separate
  llama.cpp trees. Building test with `LLAMA_RPC=1` and bench without it left
  whichever ran last in place, and the artifact's own timestamp then looked
  current to the other environment — a test suite reported
  `{:error, :rpc_unsupported}` against a live RPC worker the same tree had just
  talked to. The same hole let a downloaded precompiled artifact silently
  replace a source build. The link is now gated on a marker beside the artifact
  recording what it *is* — the configuration hash plus a digest of the linked
  bytes — so any replacement, from any source, forces a relink.
- **`make` outside `mix` wrote to `/obj` and `/priv`.** `elixir_make` always sets
  `MIX_APP_PATH`; a human running a documented command like
  `LLAMA_RPC=1 make rpc-server` does not, and an empty value was silently
  destructive rather than an error — `BUILD` became `/obj`, so cmake configured
  into the filesystem root (`Unable to (re)create the private pkgRedirects
  directory: /obj/rpc_server_build/CMakeFiles/pkgRedirects`) and `make clean`
  would have tried to `rm -rf /obj`. It now falls back to
  `_build/standalone/lib/llama_cpp_ex`, covering the defined-but-empty case too.
- **`scripts/spark/rpc-worker.sh start` compiled inside its own readiness
  window.** The worker unit runs `mix run`, which compiles on demand, and the
  wait loop cannot distinguish "still compiling" from "never going to listen" —
  it only sees an active unit and a silent port. After a llama.cpp bump that
  compile is a multi-minute rebuild, so the 120s budget expired, the script
  reported a failure, and the worker came up fine on its own minutes later. The
  build now happens before the unit is created, which also makes a compile error
  arrive as a compile error.

### Added

- **`LlamaCppEx.MTP.init/2` accepts a separate `:draft_model`.** The MTP head no
  longer has to live inside the target GGUF. Qwen 3.8 is why: `ggml-org/Qwen3.8-27B-GGUF`
  ships `Qwen3.8-27B-Q4_K_M.gguf` with *zero* nextn layers and the head alone in
  `mtp-Qwen3.8-27B-Q4_0.gguf`, so the old single-file path refused the pair
  outright with "this GGUF contains no MTP head". This is the binding's
  equivalent of upstream's `-hf <target> -hfd <draft> --spec-type draft-mtp`.
  The NIF already took two independent contexts; only the Elixir side was tying
  them to one model.

  Mismatched pairings are refused before any context is built, including a
  target/draft hidden-width mismatch — upstream compares those with a
  `GGML_ASSERT`, which is an unconditional `ggml_abort` and would take the VM
  down instead of returning an error.
- **`stats/1` reports `timing_us.ckpt`.** Recurrent-state save/restore, which
  only hybrid models pay, was previously folded into `:other` — a bucket whose
  documented cause is Metal GPU-sync waits. On Qwen 3.8 (48 SSM layers to 16
  attention ones, ~150 MiB of state snapshotted every iteration) it is the term
  that decides whether speculation helps at all: 6.9 s of a 16.3 s M1 Max run at
  `n_draft: 3`. Attributing it correctly dropped `:other` for that run from
  8.7 s to 0.17 s.
- **`LlamaCppEx.Model.n_embd_out/1` and `n_layer_nextn/1`** — the two numbers
  that decide whether a GGUF can serve as an MTP target or head. `n_layer_nextn`
  wraps a NIF that already existed but was only reachable through
  `LlamaCppEx.NIF`.

- **`LlamaCppEx.RPC`** — register a remote machine's devices into the local
  device registry so a model's layers can live on another host.
  `add_server/1`, `add_servers/1`, `devices/0`, `ping/1`, `supported?/0`.
  Registration reports `{:error, :unreachable}` rather than vanishing, which
  matters because upstream collapses an unreachable endpoint and a protocol
  mismatch into a null registration that `ggml_backend_register` silently
  ignores. `supported?/0` reports whether the backend was compiled in at all, so
  callers and tests never have to infer a build problem from `:rpc_unsupported`.
- **`LlamaCppEx.RPC.Server`** — the worker side, a supervised GenServer owning
  the native server thread. `restart: :temporary`, because a restart could not
  succeed: the listening socket lives on a detached thread in the same OS
  process, so it survives the GenServer and the next bind would fail
  `EADDRINUSE`. It traps exits so its "the native server is still running"
  warning reaches you on a supervised shutdown, which is the case that matters.
- **`:rpc_servers`** on `Model.load/2` and `Server.start_link/1` — endpoints to
  register before the load, in the order tensor placement needs.
- **`:devices`** on `Model.load/2` and `Server.start_link/1` — device names used
  **verbatim** as the placement list. Not cosmetic: llama.cpp's automatic list
  puts RPC devices first, which is *not* the order `LlamaCppEx.devices/0`
  reports, so `:tensor_split` and `:main_gpu` otherwise index a list the caller
  never saw. A backwards split still produces correct tokens and merely
  benchmarks badly.
- **`split_mode: :tensor`** now encodes to llama.cpp's `LLAMA_SPLIT_MODE_TENSOR`
  instead of raising `FunctionClauseError`, and an unknown split mode raises
  `ArgumentError` naming the accepted set.
- **Build variables** `LLAMA_CPU_ARM_ARCH`, `LLAMA_CUDA_ARCH`, `LLAMA_RPC`,
  `LLAMA_RPC_RDMA`, and a `make rpc-server` target for upstream's standalone
  worker. `LLAMA_CPU_ARM_ARCH` without `LLAMA_CUDA_ARCH` on a CUDA build is a
  hard `$(error)`: reaching the CPU flag needs `GGML_NATIVE=OFF`, which silently
  turns one CUDA architecture into a seven-architecture fat binary.
- **`scripts/spark/`** — `bootstrap.sh`, `sync.sh`, `remote.sh`,
  `verify-build-flags.sh`, `rpc-worker.sh`, `fetch_models.exs`, `rpc_check.exs`,
  and the cpuidle matrices. No `sudo` anywhere.
- **Benchmarks** `bench/spark_baseline.exs`, `spark_two_node.exs`,
  `spark_tuning.exs`, `spark_mtp.exs`, `spark_cpuidle.exs`, with results in
  `bench/results/v0.8.43-dgx-spark-{baseline,two-node}.md`.

### Changed

- `Bench.Helpers.start_server/1` forwards any option the server declares instead
  of a hand-maintained list of four, and raises on an option the server would
  reject. A benchmark that silently drops the option it is measuring is worse
  than no benchmark.
- `docs/multi-gpu.md` gains a remote-devices section, including the device
  ordering trap and a worked `:devices` example.
- **llama.cpp bumped to `a94d563ed801`**, 61 commits past b10362. Upstream
  removed `common_speculative_need_embd` in `f785fc9ea`: a draft implementation
  that wants the target's hidden states now arranges its own extraction, so the
  MTP prefill no longer requests logits at every prompt position — only at the
  final token, which is the one we sample from. This is upstream's own migration;
  `examples/speculative-simple` passes `false` for the whole prompt.
  Verified by re-measuring rather than by reading: on the dense Qwen3.6-27B pair,
  draft acceptance is **identical to the decimal** at every depth — 86.9 / 76.4 /
  68.2 / 57.1 % for `n_draft` 1–4, the same figures recorded at b10362 in
  `bench/results/v0.8.43-dgx-spark-baseline.md`. Identical acceptance is the
  claim worth making: it means the drafts themselves are unchanged, which a
  throughput number alone cannot show. Decode moved 18.65 → 18.54 t/s at the
  `n_draft: 3` peak, inside that run's recorded 18.4–18.7 range.
  `llama_model_default_params()` also moved `load_mode` from
  `LLAMA_LOAD_MODE_MMAP` to the new `LLAMA_LOAD_MODE_AUTO`, which drops mmap when
  a device reports it cannot support it. The NIF always sets `load_mode`
  explicitly from `:use_mmap`/`:use_mlock`/`:use_direct_io`, so behaviour is
  unchanged and `:auto` is not exposed yet.
- **llama.cpp bumped to `e85caa81e`** (b10582), 147 commits past b10435, and
  `LLAMA_COMMIT` moved with the submodule. Nothing in the binding changed:
  `include/llama.h`, `ggml/include/ggml-backend.h`, `common/chat.h`,
  `common/json-schema-to-grammar.h` and `common/speculative.h` are byte-identical
  across the range, so `llama_nif.cpp` needed no edit — the diff that matters for
  a bump is the header diff, and this one is empty. `ggml-rpc.h` moved only
  `RPC_PROTO_MINOR_VERSION` 0 → 1 (5.0.0 → 5.1.0), for `use_count` propagation
  that lets backends fuse ops behind RPC (`af5172627`); same major, no signature
  change, and nothing in this repo pins the protocol version.
  Two upstream commits land on the MTP path and both are no-ops for us:
  `f466cfa38` skips a null `dp.result` when `dp.drafting` is false, and the NIF
  sets `drafting = true` on every seq immediately before each
  `common_speculative_draft`; `2c6b141ef` fixes `draft-mtp` under
  `--embeddings` inside `common_base_params_to_speculative`, which the NIF does
  not call — it builds both contexts itself.
  All three defects in [docs/release-guide.md](docs/release-guide.md) still
  stand, re-checked as a source diff: `ggml_backend_rpc_start_server` still
  returns `void`, `ggml_backend_cuda_comm_init` was untouched (#26502's
  tensor-split work was reverted in `f20395dae`), and the `ggml-cpu` CMake diff
  is OpenMP target variables, KleidiAI SME2 GEMV sources and IntelLLVM
  fast-math gating — nothing near the `-mcpu=native` probe.

## v0.8.43

llama.cpp bump to b10362, on top of b10280 from v0.8.42, plus the CUDA work that
had been sitting unreleased: CUDA is now a first-class target — the NIF links
correctly against it, and the release publishes prebuilt CUDA artifacts for
CUDA 12 and CUDA 13.

Unlike the v0.8.42 range, this one does **not** break the upstream C API. Every
change in it is additive, and no NIF source change was required. One of those
additions is a near miss worth naming rather than glossing over, because it has
the exact shape of the `load_mtp` trap that broke MTP in v0.8.41 — a new
`llama_context_params` field whose default is restrictive. See **Changed** for
why the conclusion differs this time, and why it was checked by running MTP
rather than only by reading the diff.

Verified at this base (b10362) on an M1 Max, source builds with both
`LLAMA_BACKEND=metal` and `LLAMA_BACKEND=cpu`, each running the generation,
embedding and MTP suites against real GGUFs: **520 passed, 11 excluded** on both
(the exclusions are `:slow` and the known-broken `:mtp_cancel`). `mix credo
--strict`, `mix dialyzer` and `mix format --check-formatted` are clean, and a
source build from the Hex tarball resolves the pinned llama.cpp commit and links.

The CUDA work in this release was verified separately on 2x NVIDIA DGX Spark
(GB10, `sm_121a`, aarch64, CUDA 13.0.2) against the previous base (llama.cpp
b10280) — a source build with `LLAMA_BACKEND=cuda` loads, reports
`backend: "CUDA"` from `LlamaCppEx.devices()`, offloads 31/31 layers to the GPU,
and passes the smoke suite: **528 tests, 0 failures**.

### Fixed

- **`max_tokens` was not an upper bound under MTP.** The verify loop emits up to
  `1 + n_draft` tokens per iteration but checked the caller's budget only on
  iteration entry, so the last iteration could run past it. `max_tokens: 16`
  returned 16, 17 or 18 tokens for the same prompt under greedy decoding —
  measured across six successive `generate/3` calls on one session.

  The overshoot is not constant because it is a function of how many drafts the
  target accepts in the final iteration, and acceptance varies between runs on a
  reused session (11, 10, 12, 11, 10, 12 accepted of 15 drafted, same prompt and
  seed). The token *sequence* was deterministic throughout; only the stopping
  point moved. That is what made it visible: `stream/3` and `generate/3` are the
  same code path — `generate/3` is `stream_events/3` joined — yet they returned
  different-length prefixes of the same continuation, which reads as a streaming
  bug and is not one.

  The loop now re-checks the budget per token. Breaking mid-iteration leaves
  positions decoded but not emitted, which the existing partial-accept rollback
  already discards, so nothing else moved. `max_tokens: 16` now returns exactly
  16 tokens on every run, and a test pins the bound at 1, 4 and 16 rather than
  only comparing the two entry points against each other.

  Pre-existing, not from this bump — the loop last changed in #79 (v0.8.39). It
  survived because the `:mtp` suite has only ever run against one model, and the
  boundary happened to land consistently there; a 0.8B MTP model with a
  different acceptance profile exposes it on the first run.
- **MTP hybrid rollback corrupted the KV cache after a partial accept.** This is
  a different bug from the `load_mtp` one fixed in v0.8.42, and the two are
  complementary: that one stopped the MTP layers being read off disk at all,
  this one silently misplaces context once they are working.

  The verification batch is `[sampled, drafts...]` at positions starting at
  `n_past`, so `sampled` occupies batch element 0. When only some drafts are
  accepted the target's KV is rolled back to `n_past` and re-decoded — but the
  re-decode started at the first *accepted* token rather than at `sampled`.
  Every token therefore landed one position early, and the last accepted token
  was written into the context even though it becomes the next iteration's
  `sampled` and is decoded again there, duplicating it. Reading the slice from
  `prompt[size - n_accepted_total - 1]` restores `sampled, accepted[0..k-2]`
  across `[n_past, n_past + k)`, which is exactly the span `n_past` then
  advances over.

  Only reachable on a partial accept, which is why a working MTP setup can still
  post plausible acceptance rates while quietly drifting.

  Cross-checked against upstream's own reference path rather than reasoned about
  alone. `common_sampler_sample_and_accept_n` (`common/sampling.cpp`) returns the
  tokens sampled at batch indices `0..k`, and `tools/server/server-context.cpp`
  then does `slot.prompt.tokens.insert({ids.begin(), ids.end() - 1})` followed by
  `slot.sampled = ids.back()` — appending every accepted token *except the last*,
  and carrying the last into the next iteration. That is the invariant this fix
  restores. (Upstream trims with `seq_rm` where this binding rolls back and
  re-decodes, so the mechanisms differ; the resulting context must not.)
- **The CUDA NIF could not be loaded** — `ggml-cuda.a` leaves the CUDA runtime,
  cuBLAS/cuBLASLt and the CUDA driver API unresolved, but the Linux link line
  only ever added `-lstdc++ -lm -lpthread`. The resulting `.so` linked and then
  died at load with `undefined symbol: cuMemCreate`, a driver-API symbol
  ggml-cuda's VMM pool calls.
- **The toolkit was only ever looked for on `PATH`** — and that is the one place
  it frequently is not. DGX OS installs `nvcc` via `/etc/profile.d/nv_paths.sh`,
  which `ssh host mix compile`, systemd units and most CI shells never source;
  environment modules behave the same way. The two consequences were both
  silent: `LLAMA_BACKEND=auto` produced a CPU-only build on a machine with a
  complete toolkit, and `LLAMA_BACKEND=cuda` produced a link line with no `-L`
  at all. Discovery now tries `CUDA_HOME`, `CUDA_PATH`, `nvcc` on `PATH`,
  `/usr/local/cuda`, `/opt/cuda`, then the newest `/usr/local/cuda-*`, is shared
  between backend auto-detection and the link line so they cannot disagree, and
  passes `-DCMAKE_CUDA_COMPILER` so cmake's own `find_package(CUDAToolkit)`
  does not repeat the mistake. Selecting CUDA with no toolkit present is now an
  error naming `CUDA_HOME`, not `cannot find -lcudart`.
- **`undefined symbol: ncclAllReduce`** — ggml's `GGML_CUDA_NCCL` defaults to ON
  and links libnccl through cmake whenever the build host happens to have NCCL,
  which every DGX and most multi-GPU boxes do. The Makefile assembles its link
  line by hand from ggml's static archives, so cmake's `target_link_libraries`
  is invisible to it and the symbols went unresolved. The flag is now always
  stated explicitly rather than inherited, defaulting to OFF; `LLAMA_CUDA_NCCL=1`
  turns it on and adds the matching `-lnccl`.
- **Library path assumed `lib64`** — Debian's packaged `nvidia-cuda-toolkit`
  only has `lib`, and the stubs directory is now probed rather than assumed.

### Added

- **`LlamaCppEx.MTP.init/2` now says when a checkpoint has no MTP head**, via a
  new `LlamaCppEx.NIF.model_n_layer_nextn/1` wrapping upstream's
  `llama_model_n_layer_nextn`. Previously this surfaced as
  `{:error, "failed to create context"}`, with the real reason — llama.cpp's
  `context type MTP requested but model doesn't contain MTP layers` — buried in
  engine output the caller may not be showing.

  This is not the `load_mtp` case and no flag recovers it: most GGUF conversions
  of an MTP-capable model simply drop the head. Unsloth's
  `Qwen3.6-35B-A3B-UD-Q4_K_XL` reports zero nextn layers; their separate
  `Qwen3.6-35B-A3B-MTP-GGUF` build of the same model carries them. The message
  now says so and points at the `-MTP` build.

- **Precompiled CUDA artifacts** — `x86_64-linux-gnu-cu12` and
  `x86_64-linux-gnu-cu13` join the existing `aarch64-apple-darwin` (Metal) and
  `x86_64-linux-gnu` (CPU) targets, at NIF 2.17 and 2.18. `mix compile` on an
  x86_64 Linux box with a driver and a CUDA runtime now downloads a GPU build
  instead of silently installing a CPU one.

  They are separate artifacts because the NIF links `libcudart`/`libcublas`/
  `libcublasLt` dynamically and those sonames are major-versioned; one "Linux
  CUDA" binary cannot serve both. Selection requires **both** a CUDA runtime and
  a driver (`libcuda.so.1`) — a CUDA build links `-lcuda` and cannot be
  `dlopen`ed at all without one, so a toolkit-only machine keeps the CPU
  artifact rather than getting a NIF that fails to load. `LLAMA_CUDA_VARIANT`
  (`cu12`, `cu13`, `none`) overrides the probe.
- **`LLAMA_CUDA_NCCL`** build variable, off by default. See above.
- **A CUDA link gate in CI** — nothing in CI had ever selected the CUDA backend,
  which is how a NIF that could not resolve `cuMemCreate` reached a release. The
  new `cuda-link` job builds against both CUDA 12 and CUDA 13 on every pull
  request with the toolkit deliberately off `PATH`, asserts the `.so` declares
  `libcudart`, `libcublas` and `libcuda.so.1`, and then resolves every symbol
  with `ldd -r` — pointing the loader at the toolkit's driver stub, which
  carries the `libcuda.so.1` soname, so a GPU-less runner can still perform a
  real resolution. `enif_*` is excluded, being supplied by the BEAM at load.

  Each leg holds a runner for about twenty minutes, almost all of it nvcc, so
  the job runs on every pull request but on master only when a CUDA build input
  moved — `Makefile`, `c_src/`, `mix.exs`, `mix.lock`, the `vendor/llama.cpp`
  submodule, or the workflow itself. A squash merge of an up-to-date branch
  lands the exact tree the pull request tested; the case worth re-testing is
  master having moved underneath it, which is what that path list detects. A
  force-push, a missing base commit, or anything else unexpected resolves to
  running the job. The workflow also gained a `concurrency` group so repeated
  pushes to a branch cancel their predecessors instead of queueing behind them,
  keyed on `run_id` for master so every landed commit keeps its own verdict.
- **Precompiler unit tests** — `test/precompiler_test.exs` pins the artifact
  selection rules, including the case that motivates the driver check.

### Changed

- **llama.cpp submodule** — Updated from 61881b1f7 to 4801e3c56 (82 commits, tag
  b10280 to b10362). No NIF source change was required: every API change in the
  range is additive, and the binding builds all of its params from
  `llama_*_default_params()` and uses only the public `llama_sampler_init_*`
  constructors.

  - **`llama_context_params` gained `n_outputs_max_per_seq`** (#25532, multi-output
    backend sampling), and its default in `llama_context_default_params()` is `1`,
    not `0`. `llama_decode` enforces it and returns `-1` with
    `backend sampling supports at most %u outputs per sequence` when a batch
    exceeds it.

    This is the same shape as the `load_mtp` field that broke MTP in v0.8.41 — a
    new restrictive default picked up silently from the defaults struct — and the
    binding does request logits at every position of a sequence during MTP
    prefill, which is exactly the pattern the limit forbids. It is nonetheless
    inert here, for a reason worth writing down rather than rediscovering: the
    check is gated on `!sampling.samplers.empty()`
    (`src/llama-context.cpp:1664`), i.e. on *backend* samplers registered through
    `llama_context_params.samplers`. This binding never sets that field — it
    samples host-side via `llama_sampler_chain` and `llama_sampler_sample` — so
    the map is empty and the limit is never applied. Because reading the diff is
    how v0.8.41 got this wrong, it was also checked by running MTP end-to-end
    against a real MTP GGUF rather than by inspection alone.

    If backend sampling is ever adopted here, `n_outputs_max_per_seq` becomes
    load-bearing and must be set from `common_speculative_get_output_limits`
    (new in this range) the way upstream's server and `speculative-simple` now do.
  - **`llama_sampler_i` gained `backend_reset` and `copy_state`**, and
    `backend_init` gained an `n_outputs_max_per_seq` parameter. This only affects
    code that implements the sampler vtable itself; the NIF implements no custom
    sampler.
  - **New upstream entry points**, none currently used: `llama_sampler_copy`,
    `common_sampler_copy`, `common_speculative_get_output_limits`,
    `ggml_build_forward_order`.
  - **Grammar semantics**: a repetition bound of `>= 2000` now degrades to
    unbounded instead of raising (#26613). Reachable from `LlamaCppEx.Schema` and
    `LlamaCppEx.Grammar` — a schema whose `maxItems` exceeded the threshold used
    to fail grammar compilation with `number of repetitions exceeds sane
    defaults` and now compiles. A `minItems` over the threshold still raises.
  - **MTP/speculative upstream fixes**: memory allocation for MTP layers (#26605)
    and MTP support for Nemotron (#26725). `common/speculative.h` is otherwise
    additive and `common/chat.h` and `common/json-schema-to-grammar.h` are
    untouched in this range.
  - **Metal**: `NORM`/`RMS_NORM` fixed for row lengths that leave a partial
    simdgroup (#26708), a `threadgroup` matrix instantiation removed from
    `kernel_lightning_indexer` (#26646), and `ROLL` now requires a contiguous src
    (#25928). All three are correctness fixes on the backend that ships in the
    `aarch64-apple-darwin` artifact.
  - **CPU/aarch64**: HWCAP fallbacks and fp16 variant detection (#25554), which
    matters to the `LLAMA_PORTABLE=1` artifacts, plus a missing Q5_0 dispatch
    (#26792) and an Android CPU-affinity fix (#26838).
  - **CUDA**: `rms_norm + mul + rope` fusion (#26767) and a thread/block count
    fix in the quantized cpy kernel launches (#26731).
  - **ggml** version moved 0.18.1 to 0.19.0. No cmake option this Makefile passes
    was renamed or removed, so the build configuration is unchanged.
  - New architectures: Granite-Switch (#25107), Muse Glimmer (#26841), plus an
    EXAONE 4.5 SWA fix (#26848).

### Tests

- **The `:mtp` and `:embeddings` suites now run against real models on Apple
  Silicon**, not just the CPU-only generation model. `unsloth/Qwen3.5-0.8B-MTP-GGUF`
  makes the MTP suite cheap enough to run per-release — the previously
  recommended MTP GGUFs start at ~21 GB, which is why that suite had only ever
  been exercised against a single model, which is how the `max_tokens` bound
  above stayed broken.
- **`max_tokens` is pinned as an exact bound** at 1, 4 and 16 tokens, instead of
  being covered only indirectly by a `stream/3`-versus-`generate/3` comparison
  that could not distinguish a streaming bug from a budget bug.

## v0.8.42

llama.cpp bump to b10280, on top of b10217 from v0.8.41. Unlike the last two
ranges this one **does** break the upstream C API, and validating it against real
models surfaced a separate bug that had made MTP speculative decoding inoperable
since v0.8.41. The public Elixir API gains one option and no removals.

### Fixed

- **MTP speculative decoding was completely broken and is now working.** b10217
  (in v0.8.41) added `bool load_mtp` to `llama_model_params` to stop
  non-speculative callers paying for the MTP head's tensors (#26296, extended by
  #26412 here). v0.8.41's notes read that field as inert because the NIF builds
  its params from `llama_model_default_params()` and so picks up new fields
  automatically — which is true at compile time and wrong at runtime: the default
  is `false`, so the MTP layers stopped being read off disk. Nothing on the way in
  objects. Both contexts build and `common_speculative_init` returns ok; the first
  draft then fails with `verify decode failed: code=-1`, far from the cause.

  `LlamaCppEx.Model.load/2` therefore gains a `:load_mtp` option (default
  `false`, matching upstream), the flag is recorded on `%Model{}`, and
  `LlamaCppEx.MTP.init/2` refuses a model loaded without it, naming the remedy —
  the layers cannot be attached after the fact, so failing at `init/2` is the only
  place the error is actionable.

  This went unnoticed because the `:mtp` suite had never been run against a real
  model: it needs an MTP-enabled GGUF, and none was available when the tests were
  written. It is now verified end-to-end against `Qwen3.5-9B` (`qwen35` with
  `nextn_predict_layers`), at ~67% draft acceptance with `n_draft: 3`.

### Changed

- **llama.cpp submodule** — Updated from ddd4ec142 to 61881b1f7 (63 commits, tag
  b10280). Two binding-relevant signatures changed, both breaking:
  - `llama_sampler_init_penalties` gained a leading `int32_t n_vocab` parameter
    (#26520, which moved `n_vocab` out of `llama_sampler_data` and into the
    penalty sampler so the CUDA backend sampler added in #25262 can size its
    buffers). The NIF's one call site now passes
    `llama_vocab_n_tokens(model->vocab())`. Note that the argument *count*
    changed, so a missed call site is a compile error rather than a silent
    reordering — but the two leading `int32_t`s are interchangeable to the
    compiler, which is why the penalties path is now covered behaviourally
    (a repetition-inducing prompt under greedy decoding must be diverted by
    `penalty_repeat`, and `penalty_repeat: 1.0` must be a byte-identical no-op)
    rather than only by constructing a sampler.
  - `llama_sampler_init_dry` dropped its `int32_t n_ctx_train` parameter in the
    same range. The binding does not expose the DRY sampler, so nothing to do.
  - **Sampler semantics**: `-1` no longer means "context size" for
    `penalty_last_n` or `dry_penalty_last_n` — history-based samplers lost their
    full-context windows (#26524) and a negative value now clamps to `0`, i.e.
    disabled. The NIF hardcodes `penalty_last_n = 64`, so behaviour here is
    unchanged, but anyone who read `-1` as "whole context" upstream should note it.
  - `include/llama.h` aside, `common/chat.h`, `common/speculative.h` and
    `common/json-schema-to-grammar.h` are untouched in this range.
    `common/common.h` changed only in ways the binding does not reach: the
    `common_params_vocoder` struct was replaced by `tts_lang` / `tts_speaker_file`
    fields on `common_params` (#26254, which also breaks the `llama-tts` binary),
    and `common_get_env` / `common_set_env` were added.
  - **Speculative / MTP**: `common_speculative_init` refactored its enabled-config
    handling (#26510); MTP support added for Qwen3-Next (#25589), DeepSeek V3.2
    (#26457), GLM-4.7-Flash (#24868) and DeepSeek V4 alongside DSpark (#25784);
    MiMo V2 MTP tensors are loaded only when used (#26412); dflash `wo_a` reshape
    fixed on load (#26577).
  - **llama core / models**: tensors may be reshaped during load (#26531); the
    indexer cache is allocated only in "full" indexer layers (#26474); MiniMax M3
    moves MSA into a new memory implementation (#26338) and its graph no longer
    leaves input tensors unused (#26519); a DeepSeek V4 Flash 0731 chat template
    (#26398) and a Qwen3 specialized parser (#26252) were added.
  - **vocab / convert**: default special token ids (#26506) and plamo2 byte tokens
    (#26511) are now validated; `gguf-py`'s reader validates `n_dims` and guards
    against `uint64` overflow (#25401).
  - **ggml**: version bumped to 0.18.1 (ggml/1578) and synced; split-graph inputs
    now use dynamic allocation (#22789).
  - **Metal**: DeepSeek V4 Lightning Indexer (#25893) and hyper-connections
    (#26459) implemented; `SILU_BACK` (#25982) and F16 support for binary ops
    (#26465) added; `GGML_METAL_USE_BF16` removed from the build scripts (#26604).
  - **CUDA**: a backend sampler for penalties (#25262); a data race fixed when
    reusing SMEM in `block_reduce` (#26385).
  - **SYCL**: non-contiguous concat kernel parallelized (#25852); oneDNN SDPA
    extended to non-FP16 KV caches (#25874); iGPU classification fixed (#26105).
  - **Vulkan / OpenCL / WebGPU**: `GATED_LINEAR_ATTN` implemented (#25601);
    `topk_moe` fusion extended to `sqrt(softplus)` (#26124); large `q6_K` lm_head
    routed to the flat GEMV and workgroup sizes limited for GLU (#26427, #26383);
    f16 repeat support added to the WebGPU backend (#26307).
  - **vendor**: BoringSSL updated twice (#26353, #26523), cpp-httplib to 0.52.0
    (#26485), and patches applied for `subprocess.h` (#26606). None of these link
    into the NIF, which builds with `LLAMA_BUILD_SERVER=OFF` and
    `LLAMA_OPENSSL=OFF`.
- **`Makefile`** — `LLAMA_COMMIT` follows the submodule to 61881b1f7. It is what a
  Hex *source* build clones when `vendor/llama.cpp` is absent, so leaving it behind
  gives source builds the old llama.cpp and git checkouts the new one.

### Tests

- **Three smoke tests asserted prefix reuse that is impossible on hybrid GDN
  models.** `server_smoke_test.exs` (cache-scope isolation, `PromptCache.restore/4`
  partial trim) and `llama_cpp_ex_test.exs` (the hybrid `cache_prompt` regression
  test) all required `prefix_cache_tokens > 0`. Recurrent state cannot be rolled
  back to an arbitrary position, so `llama_memory_seq_rm` refuses a partial range
  and the Server *correctly* declines reuse (`server.ex:812`) — meaning the
  hybrid-model regression test contradicted the very fallback it documents. They
  passed only because CI's generation model is a dense one.

  Each now branches on `LlamaCppEx.TestModels.seq_rm_kind/1` (a memoised probe of
  `common_context_can_seq_rm`) and asserts the documented behaviour for the model
  in hand: reuse on `:part`, the full-reset fallback and `{:error, :seq_rm_refused}`
  on `:full`. No branch is a skip, and neither `case` has a catch-all, so an
  unexpected verdict raises instead of passing quietly. Verified both ways —
  502 passed against a `:part` model and 502 against a `:full` one.

  This was **not** a regression from the bump: `llama-memory-recurrent.cpp` and
  both hybrid memory implementations are byte-identical between b10217 and b10280.

### Known issues

- **Cancelling an MTP stream and immediately reusing the session aborts the VM.**
  `request_cancel` sets a flag and the MTP loop then stops without emitting a
  terminal event (`llama_nif.cpp:2009`), so `Generator.stop/1` has no completion
  signal to await and returns while the dirty scheduler may still be inside
  `llama_decode`. For `LlamaCppEx.stream/3` this is harmless — each call owns a
  context that dies with it — but an `%MTP{}` session holds two long-lived contexts
  that every call shares, so the next `generate/3` can put a second writer on a KV
  cache the cancelled loop has not released, tripping
  `GGML_ASSERT(offset + size <= ggml_nbytes(tensor))` or segfaulting.

  Pre-existing and unrelated to this bump; it became reachable only because MTP
  works again. Fixing it means acknowledging cancellation in the NIF and in the
  cancellation protocol that non-MTP streaming shares, so it is deliberately not
  bundled here. `LlamaCppEx.MTP`'s docs carry the warning and the reproduction
  lives in `LlamaCppEx.MTPCancelTest`, gated behind its own `:mtp_cancel` tag —
  its own module, without `:mtp`, because `--include` beats `--exclude` and a
  second gate tag would abort `--include mtp` runs.

### Validation

Built from source against the bumped submodule (Metal, `LLAMA_BACKEND=auto`) and
run against real GGUF models from local storage:

- **374 passed** in the default suite (no model required).
- **502 passed** with `--include smoke --include embeddings --include slow`,
  twice: once with `Llama-3.2-3B-Instruct` (`:part` seq_rm) and once with
  `Qwen3.5-0.8B` (`:full`, hybrid GDN).
- **384 passed** with `--include mtp` against `Qwen3.5-9B` (MTP head).
- Generation, chat templating, streaming, JSON-schema-constrained sampling,
  grammar-with-penalties, and the penalties path itself were additionally checked
  across nine architectures — `llama`, `qwen3`, `qwen35`, `qwen35moe`, `gemma4`,
  `gpt-oss`, `bert` (Ministral), plus `Qwen3-Embedding-0.6B` for the embedding
  paths (L2 norm exactly 1.0; paraphrase similarity 0.73 against 0.20 for
  unrelated text).

## v0.8.41

Maintenance release: llama.cpp bump to b10217, on top of b10178 from v0.8.40.
This range does **not** break the upstream C API — no NIF change was required and
the public Elixir API is unchanged. Full suite against the rebuilt NIF with real
GGUF models (`:smoke` and `:embeddings` included, dense Llama-3.2-1B as the
generation model): 489 passed, 0 failures.

### Changed

- **llama.cpp submodule** — Updated from 992c32532 to ddd4ec142 (39 commits, tag
  b10217). Two binding-relevant headers changed, both additive:
  - `include/llama.h` — additive only. `llama_model_params` gains a trailing
    `bool load_mtp` ("whether to load MTP layers"), so MTP tensors are only
    loaded when actually used (#26296). The NIF builds its params from
    `llama_model_default_params()` (`llama_nif.cpp:329`) rather than
    aggregate-initialising the struct, so the new field picks up its upstream
    default and the addition is source-compatible — no change needed. A new
    `llama_vocab_get_suppress_tokens` accessor exposes the
    `tokenizer.ggml.suppress_tokens` GGUF key now that suppress-token handling
    moved into `common/sampling` (#26276); the binding does not read it.
  - `common/common.h` — the inline `common_params_sampling::has_logit_bias()`
    helper was removed alongside the suppress-tokens refactor (#26276), and a
    test-only `common_get_model_or_exit` declaration was added (#26317). The
    binding calls neither — it does not use `common_params_sampling` — so both
    are inert here.
  - `include/llama.h` aside, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`,
    `ggml/include/ggml-metal.h`, `common/chat.h`, `common/sampling.h`,
    `common/speculative.h` and `common/json-schema-to-grammar.h` are all
    untouched in this range.
  - **llama core / models**: load MTP tensors only if they are really used
    (#26296); enforce the same K and V cache types for DeepSeek V4 and enable
    flash attention when the V cache is quantized (#25871); support rotated KV
    cache quantization (#26180); sync pending async copies before clearing
    `embd_seq` in `llama-context` (#25676); move suppress-tokens handling to
    `common/sampling` (#26276); enable tool calls inside thinking blocks for DS4
    (#26269); address MTP review feedback for mimo2 (#26228).
  - **ggml**: bump ggml version to 0.18.0 (ggml/1576) and sync.
  - **Metal**: fix memory unwire when a model is freed without any GPU
    operations having run (#26082); remove a custom CPU op from the M3 graph and
    express it with stock ops (#26297).
  - **CUDA**: add Q2_0 support (#25707); extract Q2_0 elements via `__byte_perm`
    (#25603); allow transpose-free gemmv computation (#26171); disable MMQ on
    devices with less than 48 KiB shared memory (#26141).
  - **Vulkan**: add a POOL_1D op (#25431); support quantized concat (#25684);
    update the Vulkan SDK to 1.4.357.0 (#26303); add a driver-version check for
    Windows Intel GPUs to mitigate crashing (#25192).
  - **SYCL**: add oneMKL GEMM flash attention for XMX-accelerated prompt
    processing (#25025); support q2 `mul_mat` (#26231); fuse RMS_NORM + MUL
    (#26015); support the missed types in `cpy` (#26005); support dev2dev memcpy
    via `DEV2DEV_MEMCPY_FORWARD` (#26234); contiguous fast path and 32-bit index
    math for unary elementwise ops (#25946).
  - **WebGPU**: improve `flash_attn_vec` for quantized KV at long contexts
    (#25956).
  - **ZenDNN**: group matmul direct API for `mul_mat_id` (#25918).
  - **mtmd**: add `n_embd_head` (#26342); add a lanczos resize method (#26341).
  - **server / tools**: correct accepted tokens when a draft-token replay is
    needed (#26320); support input embeddings to generate the next token
    (#26313). Neither path is used by this binding's own server.
  - **kleidiai**: fix a CI issue and a `stringop-overflow` warning (#26277).

## v0.8.40

Audit-remediation release. Closes 72 of the 106 findings from the
2026-07-27 project health audit (`.claude/audit/`), across VM safety at the NIF
boundary, public-API error contracts, the release pipeline, HuggingFace Hub
hardening, OTP hygiene, test integrity and the CI verification gate. The 34
remaining findings are all Low severity and explicitly deferred.

**Read the Breaking section before upgrading.** Two contracts changed, both
deliberately: `chat_completion/3` now reports failures as failures, and the
facade's streams emit errors instead of truncating silently.

### Breaking

- **`LlamaCppEx.chat_completion/3` returns `{:error, reason}` on a failed
  generation and `{:error, :timeout}` on a timeout.** It previously returned
  `{:ok, %ChatCompletion{finish_reason: "stop"}}` for a NIF error and
  `finish_reason: "length"` for a timeout, so a caller matching only `{:ok, _}`
  silently consumed truncated or empty output, and the error branch of the
  function's own `@spec` was unreachable. Callers that pattern-match `{:ok, _}`
  will now crash where they previously proceeded on bad data — that is the point.
  Match both branches.
- **`LlamaCppEx.stream/3`, `stream_chat/3` and `stream_chat_completion/3` emit a
  final `{:error, reason}` element** (and `{:error, :timeout}` on a chunk
  timeout) instead of halting silently. `LlamaCppEx.Server.stream/3` already
  behaved this way; the four facade streams did not, so the library shipped two
  opposite error conventions for the same operation. `Enum.join/1` on a stream
  that may fail is no longer safe — match on the element shape.
- **Unknown options now raise.** Every public entry point validates its option
  keys and raises `ArgumentError`, with a "did you mean" hint for near misses.
  Because `Keyword.take/2` is the routing mechanism in this library, an unknown
  key was structurally indistinguishable from another module's key: `generate(model,
  prompt, temperature: 0.1)` silently ran at the default temperature and
  `n_paralell: 8` silently ran 4 slots. Typos that were previously ignored will
  now fail loudly.
- **`LlamaCppEx.Server.get_model/1` raises `ArgumentError`** instead of exiting
  with `{:noproc, …}` on a dead or still-loading server. Its `@spec` claimed a
  total function. New `Server.fetch_model/1` returns
  `{:ok, model} | {:error, :noproc | :not_ready}` for callers that want to handle it.
- **`LlamaCppEx.Server` `:max_queue` defaults to `64`, not `0`.** At `0` the
  reject branch was dead code, the documented `:queue_full` error could never
  fire, and the queue was unbounded while each entry held a full token list.
  Pass `max_queue: 0` to restore the old unbounded behaviour.
- **`LlamaCppEx.ModelManager.list/0` and `default/0` return
  `{:error, :not_started}`** when the manager is not running, instead of `[]` and
  `nil`. A missing `LlamaCppEx.ModelSupervisor` in the supervision tree used to
  be indistinguishable from an empty registry.
- **`%LlamaCppEx.Sampler{}` gained a `:model` field** and
  `Sampler.create/2` can return `{:error, :invalid_grammar}`. `@enforce_keys` is
  still `[:ref]`, so pattern matches keep working; only hand-construction of the
  struct is affected.
- **The library now starts a supervision tree** (`LlamaCppEx.Application`, one
  `Registry` child). Host applications need no change.
- **Test env vars renamed** to one family: `LLAMA_SMOKE_GEN_MODEL`,
  `LLAMA_SMOKE_EMB_MODEL`, `LLAMA_SMOKE_MTP_MODEL`. The old
  `LLAMA_MODEL_PATH` / `LLAMA_EMBEDDING_MODEL_PATH` / `LLAMA_MTP_MODEL_PATH` are gone.
- **`LlamaCppEx.Hub` caches under `<repo_id>/<revision>/<filename>`.** `revision`
  was absent from the cache key, so pinning `revision: "<sha>"` returned the
  cached `main` copy once `main` had been fetched — the pin bought nothing, and
  said nothing about it. **Migration:** existing caches are laid out without the
  revision component and will be treated as misses, so the next `download/3` for
  each file re-downloads once. Delete the old tree or leave it; nothing reads it.
- **A download whose file has no published SHA-256 now fails.** `:verify_checksum`
  takes `true` (default, fail closed), `:best_effort` (warn and proceed when
  HuggingFace publishes no digest), or `false` (skip the check and the metadata
  request). Verification used to fall back to a warning whenever
  `siblings[].lfs.sha256` was absent, which made it downgradable by the *metadata*
  response: a MITM with a trusted cert — or the TLS-terminating corporate proxy
  this module explicitly supports — strips one JSON key and the bytes are cached
  unverified. Any other value for `:verify_checksum` is now rejected rather than
  treated as truthy.
- **`LlamaCppEx.Server.stream/3` and `stream_tokens/3` emit `{:error, :timeout}`**
  on a per-token timeout instead of truncating silently, and cancel the request
  server-side. They were the last two streams in the library that ended a failed
  generation indistinguishably from a successful one, contradicting their own
  `@doc`.
- **The undocumented-but-documented `:template` option is gone.** `chat/3` and
  `chat_completion/3` advertised "Custom chat template string" and accepted the
  key; nothing ever read it — `Chat.apply_template/3` goes through
  `chat_apply_template_jinja`, which uses the model's embedded template. Passing
  it now raises an unknown-option `ArgumentError` instead of being ignored.
- **`LlamaCppEx.Server` request options are validated on the streaming path too,
  and the server-routed facade rejects start-time options.**
  `chat_completion/3`/`stream_chat_completion/3` against a server used to accept
  `:n_ctx` and every `Context.tuning_option_keys()` entry — 21 keys a running
  server cannot honour — and then raised an `ArgumentError` naming
  `LlamaCppEx.Server.complete_tokens/3`, a function the caller never called. They
  are now rejected up front, naming the function the caller did call.
- **`use_mlock: true, use_mmap: false` no longer memory-maps the model.** Since
  v0.8.39 the three load booleans have collapsed into llama.cpp's single
  `load_mode` enum, where `LLAMA_LOAD_MODE_MLOCK` meant "mmap **and** mlock" — so
  `:use_mlock` silently forced a memory map and `:use_mmap` was unreachable
  whenever it was set. Upstream b10173 split that into `MLOCK` (mlock without
  mmap) and `MMAP_MLOCK` (both), and `Model.load/2` now honours the two options
  independently: both true selects `mmap_mlock`, `:use_mlock` alone selects
  `mlock` and reads the weights into anonymous memory. `:use_direct_io` still
  takes precedence over both, and the default path is unaffected because
  `:use_mmap` defaults to `true`.

### Security

- **Use-after-free in `%Sampler{}` (Critical).** `sampler_init` captured the raw
  `const llama_vocab*` from `model->vocab()` with no ownership link, so dropping
  the model term in Elixir freed the vocabulary under a live sampler and the next
  `Sampler.reset/1` or `accept/2` dereferenced freed heap. `LlamaSampler` now holds
  a `fine::ResourcePtr<LlamaModel>`, matching what `LlamaContext` and
  `LlamaSpeculative` already did.
- **Every value crossing the NIF boundary is now range-checked.**
  `GGML_ASSERT` is never `NDEBUG`-gated, so it calls `ggml_abort()` → `abort()`: a
  bad integer from Elixir took down the entire BEAM with no exception, no
  supervisor and no crash report. Verified by removing each guard in turn,
  rebuilding, and watching the VM abort with `NDEBUG` active — `embed_decode/3`
  with an out-of-range `seq_id` exits the OS process with `SIGABRT`.

  - `seq_id` is validated against `llama_n_seq_max/1` in every NIF that can reach
    a `llama_memory_*` or `llama_state_seq_*` call outside a batch:
    `memory_seq_rm/4`, `memory_seq_cp/5`, `memory_seq_keep/2`,
    `memory_seq_pos_max/2`, `state_seq_get_size/2`, `state_seq_get_data/2`,
    `state_seq_set_data/3`, `embed_decode/3`, `embed_batch_decode/2` and
    `batch_eval_sample/4`'s `purgeable_seq_ids`.
  - Sequence ids carried *inside* a `llama_batch` are bounded by upstream's
    `llama_batch_allocr::init`, which returns `false` rather than asserting, so
    `llama_decode` returns non-zero and the NIF returns `{:error, _}`. That is why
    `prefill/3`, `decode_batch/3`, `decode_token/4`, `batch_eval/2` and
    `batch_eval_sample/4`'s entries need no local guard — a distinction now
    recorded in code and enumerated in `test/nif_guards_test.exs`, whose
    `@seq_id_surface` table is checked against `LlamaCppEx.NIF`'s own source so a
    new `seq_id`-taking NIF fails the suite until it is classified.
  - The logits index in `sampler_sample/2` and `sampler_sample_at/3` is validated
    against the shape of the last successful decode, reproducing all three cases
    upstream's `output_resolve_row` rejects. `llama_get_logits_ith` only degrades
    to `nullptr` when llama.cpp itself was built with `NDEBUG`, and the following
    `GGML_ASSERT(logits != nullptr)` aborts regardless, so the bound has to be
    kept here.
  - `memory_seq_cp/5` refuses a *partial* cross-sequence copy on a split KV cache,
    which reaches `GGML_ASSERT(is_full)` in `llama-kv-cache.cpp` and aborts the
    process image. `Server` guarded this with its `cross_slot_sharing` flag, but
    that is an Elixir-side convention and the NIF is callable directly.
  - Grammar and JSON-schema text are bounded at 1 MiB and 64 nesting levels.
    Both parsers are recursive descent over the input, so text nesting depth is C
    stack depth and unbounded nesting is a `SIGSEGV` no `try/catch` recovers from.
  - `state_seq_set_data/3` rejects a blob too short to hold upstream's header and
    treats a `0` return as an error instead of proceeding as if the restore worked.
- **A malformed grammar is no longer silently dropped.** `sampler_init` did
  `if (grammar) chain_add(...)`, so a GBNF parse failure disabled the constraint
  and a caller who asked for JSON got unconstrained output with no indication.
  Now `{:error, :invalid_grammar}`. This was a validation bypass, not a cosmetic
  issue.
- **Path traversal to arbitrary file write in `Hub.cache_path/3`.** A `repo` or
  `filename` containing `..` or an absolute component escaped the cache directory.
  Both are now validated before joining; `cache_path/3` raises on unsafe input and
  `download/3` errors before any HTTP request.
- **Downloaded models are integrity-checked.** `Hub.download/3` fetches the
  published SHA-256 from the HuggingFace revision API and verifies the bytes that
  landed on disk, failing the download on mismatch. Opt out with
  `verify_checksum: false`.
- **The download temp file is randomized and opened `O_EXCL`**, so a symlink
  planted at the target fails the open instead of being written through, and two
  concurrent downloads cannot corrupt each other. Cache directories are `0o700`
  and cached files `0o600`.
- **New `:cache_scope` request option scopes KV prefix reuse to a trust
  boundary.** Prefix reuse — from the slot's own cache, from another slot, or from
  the RAM prompt cache — is a KV read, so two tenants sharing a system prompt
  could otherwise inherit each other's cache. Defaults to `nil`, a single shared
  pool, which is only safe when every caller is in one trust domain.
- **Release-pipeline hardening.** Every GitHub Action in the workflow holding
  `HEX_API_KEY` is SHA-pinned; `ci.yml` gained an explicit least-privilege
  `permissions:` block; the `sed` script built from `GITHUB_REF` is validated
  against a semver regex first (GNU sed `e`-flag injection); `mix
  elixir_make.checksum --ignore-unavailable` no longer allows a `checksum.exs`
  that vouches for fewer targets than were published.
- **`ErlNifEnv` and `llama_batch` leaks** in the three streaming NIFs are fixed
  with RAII guards, so a C++ exception unwinding out of a generation loop no
  longer leaks.
- **Session affinity is keyed by `{cache_scope, session}`, not by `:session`
  alone.** `:session` was a global keyspace: affinity routed on the session id
  while only prefix *reuse* checked `:cache_scope`, so a guessed session id let one
  scope claim the slot another scope was using and evict its prefix cache. A
  denial of service rather than a KV leak — the scope check still cleared the KV on
  mismatch — closed while the feature is new.
- **A failed write during a download is reported, and Req's retry can no longer
  append a second response body.** `IO.binwrite/2` does not return
  `{:error, reason}`; it calls `:erlang.error(reason)`, so a full disk raised
  `:enospc` out of Req's streaming callback and past `download/3` into the caller's
  crash report — with the request options, `:token` included, in the stacktrace.
  The write now goes through `:file.write/2` and a failure becomes
  `{:error, reason}` with nothing left in the cache. Separately, Req's retry
  re-runs the request with the same `into:` closure over the still-open device
  (`req/lib/req/steps.ex:2315`), so a transient 503 *appended* a second body rather
  than restarting the file — a corrupt GGUF with a plausible size and a valid ETag.
  Retry is off for this request; a failed download is the caller's to repeat.
- **A broken GPU backend no longer reports as "no GPUs".**
  `ModelManager`'s device enumeration rescued all of `ErlangError`, which covers
  both "the NIF is not loaded" and "the backend call failed", so a genuine failure
  silently turned every VRAM budget into a RAM budget. Only `:not_loaded` degrades
  now; anything else propagates with its original stacktrace.

### Fixed

- **Source builds are possible again (Critical).** `package()` `files:` omitted
  `.gitmodules` and the Makefile required `vendor/llama.cpp/CMakeLists.txt` with no
  fallback, so *every* source build died at `No rule to make target` — including
  every documented `LLAMA_BACKEND` user, since setting it forces a source build.
  The Makefile now clones llama.cpp pinned to the submodule SHA. Regression-tested
  by extracting a Hex tarball into a clean directory with no `vendor/` and no
  surrounding git repository and running `LLAMA_BACKEND=cpu mix compile`; it fails
  on the old Makefile and succeeds on the new one.
- **Published Linux artifacts were compiled with `-march=native`.**
  `GGML_NATIVE_DEFAULT` is `ON` unless cross-compiling, and nothing neutralized
  it, so the published `x86_64-linux-gnu` `.so` was tuned to whatever CPU the
  release runner had — a SIGILL risk on user machines and a non-reproducible
  build. The precompile workflow now sets `LLAMA_PORTABLE=1`, which adds
  `-DGGML_NATIVE=OFF`; local builds keep native tuning. Reproduced on an M4, where
  the untuned build emits `-mcpu=native+…+sme` and `sme` does not exist on an M1.
- **The `.built` sentinel made submodule bumps and backend switches silent
  no-ops.** The llama.cpp build directory is now keyed by backend and portability
  and the stamp by the llama.cpp commit, so either change forces a rebuild. This
  repo bumps llama.cpp in 61 of its 223 commits.
- **Hex publishing used `master`, not the tag.** The checksum job's five-attempt
  rebase loop is itself evidence that concurrent `master` commits are expected, so
  published source could differ from the binaries `checksum.exs` vouches for.
- **GitHub Releases went public with 1 of 4 assets.** The release is now created
  as a draft and published only after every artifact has uploaded.
- **`Server.init/1` hard-matched everything expensive.** `LlamaCppEx.init()`, a
  multi-hundred-MB `Model.load`, `Context.create` and `n_parallel` sampler chains
  were all `{:ok, x} = …` inside `init/1`, so a load failure surfaced as an opaque
  `MatchError` and `start_link/1` blocked the supervision tree's boot for the whole
  load. Cheap validation stays in `init/1` (so a bad `:model_path`, a missing
  `:model_path` and a typo'd `:batch_strategy` still fail `start_link/1`
  synchronously, with a named reason); everything expensive moved to
  `handle_continue/2` and reports `{:stop, {:load_failed, reason}}`.
- **`Server` had no `handle_info/2` catch-all.** One stray message — a late
  `:ssl_closed`, a reply to a timed-out call — was a `FunctionClauseError` that
  killed the model, dropped the `%Model{}`/`%Context{}` refs and failed every
  in-flight request; and because backing servers are `restart: :temporary`, it
  never came back.
- **`:batch_strategy` is validated at startup.** A typo previously surfaced as an
  `UndefinedFunctionError` inside `handle_info(:tick)` — after the model load.
- **`Server.stream/3` and `generate/3` no longer hard-match tokenization**, and
  `LlamaCppEx.generate/3`, `chat/3`, `stream/3` and both `chat_completion` clauses
  thread every fallible step through `with`. The `Stream.resource` cases used to
  raise lazily on the consumer's first `Enum.take`, far from the call that
  configured them.
- **Exception-driven feature detection no longer swallows real failures.**
  `ModelManager`'s `:ets` rescues could not distinguish "table absent" from "key
  absent"; device enumeration reported a genuine backend failure as "this machine
  has no GPUs", silently turning every VRAM budget into a RAM budget. The bare
  `rescue`/`catch` in `Budget.system_memory_bytes/0` is narrowed to the one
  exception `System.cmd/3` can actually raise.
- **A crashed backing server's permanent `:error` entry is now documented.**
  `restart: :temporary` is the right call — it stops the `DynamicSupervisor`
  resurrecting a server `ModelManager` has disowned — but the entry then returned
  `{:error, {:not_ready, :error}}` forever with no documented way out. The log line
  now names the recovery: `unload/1` then `load/3`.
- **`req` is no longer pinned away from 0.6.x** — `"~> 0.5 or ~> 0.6"`.
- **A caller-supplied `:grammar` can no longer crash a `LlamaCppEx.Server`.**
  `Sampler.create/2` gained `{:error, :invalid_grammar}` but one of its four call
  sites — `init_slot/4`, inside the GenServer that owns the model — still
  hard-matched `{:ok, sampler}`. Because backing servers run `restart: :temporary`,
  a single bad grammar killed the server permanently. A new validate-only
  `grammar_validate` NIF checks the grammar at the admission boundary, so every
  public entry point returns `{:error, :invalid_grammar}` synchronously and the
  request never reaches a slot; `init_slot/4` keeps a `case` as depth-2 defence
  that fails the request rather than the server.
- **`MTP.stream_events/3` no longer returns an infinite stream on setup failure.**
  The failure was signalled by *adding* a `:setup_error` key instead of flipping
  the discriminant the clauses dispatch on, so the halt clause never matched and
  `generate/3`'s `Enum.to_list/1` hung with a growing heap. Reachable from
  `grammar: "not gbnf"` or any tokenization failure.
- **A timed-out server-routed `stream_chat_completion/3` cancels its request.**
  The after-function decided by `phase != :done`, and the timeout path set
  `phase: :done`, so the slot stayed occupied and the model decoded to
  `max_tokens` for a consumer that had gone — the leak `stop_generator/3` prevents
  on the model-routed path, reintroduced on the server path.
- **`Generator.stop/1` no longer leaves a stray `{:EXIT, pid, :normal}`.**
  `Process.unlink/1` does not remove an exit signal already in the mailbox, and
  the runner exits normally the instant the NIF returns. A trapping consumer got
  noise; one using the `{:stop, reason, state}` shape — which `LlamaCppEx.Server`
  itself does — shut down silently. Reproduced 200/200 before the fix, 0/200 after.
- **`Server.fetch_model/1` and the streaming admission calls turn every exit into
  a value.** `fetch_model/1` caught `:noproc`, `:normal` and `:timeout`, so
  `handle_continue/2`'s `{:stop, {:load_failed, reason}, state}` — plus
  `{:shutdown, _}` and `:killed` — exited the caller from inside both
  `Stream.resource` start-functions, past a `@spec` that promised a total
  function. Both admission calls are also bounded by the caller's `:timeout` now
  rather than `GenServer.call/2`'s implicit 5000 ms.
- **`memory_seq_rm` returning `false` is handled instead of hard-matched.**
  `llama_memory_recurrent::seq_rm` refuses a partial rollback deeper than
  `n_rs_seq` and returns `false`; the Elixir guard covered only `seq_rm_kind ==
  :full`, so on an `:rs` context `true = ...` was a `MatchError` inside the
  server's tick. Both call sites now fall back to a full clear and log.
- **`ModelManager.load/3` forwards only what each destination reads.** The
  `Keyword.drop/2` denylist let `:vocab_only` through to `Server.start_link/1`,
  which rejects it — while `load/3` documents accepting "any `Model.load/2` or
  `Server.start_link/1` options". Two allowlists now route it, with one gate on
  their union so a typo like `n_paralell` still fails loudly instead of silently
  running the default `:n_parallel`.
- **A NIF failure inside `PromptCache.save/4` is logged rather than being
  indistinguishable from "the prefix was too short".** Four of the five `with`
  clauses are policy decisions and stay silent; the `{:error, reason}` clause is
  split out. The symmetric operation already disagreed — `apply_ram_restore/4`
  logs on restore failure.

### Performance

- **`tokenize`, `detokenize`, `sampler_init` and `sampler_reset` moved to dirty
  schedulers.** `tokenize` measured 862 µs at 1k tokens, 8.56 ms at 10k and 137 ms
  at 160k — it crosses the 1 ms normal-scheduler budget at ~1160 prompt tokens and
  it runs in the *caller's* process. `sampler_init` compiles GBNF eagerly (190 µs
  on a 245-rule schema) and `sampler_reset` silently re-parses the entire grammar,
  both per request inside `Server.handle_call/3`.
- **`Slots.donor_prefix_match/2` no longer rebuilds each candidate slot's token
  history.** `prompt_tokens ++ Enum.reverse(generated_token_ids)` measured 85.4 µs
  and ~460 KB of garbage per candidate at a 32k prompt — ~600 µs per request at
  `n_parallel: 8`, in the process that runs every forward pass. The two lists are
  now walked in sequence, and the generated tail is only reversed when the whole
  prompt matched.
- **The longest-common-prefix scan is no longer computed twice per request.**
  `Slots.pick_cached_slot/2` returns `{seq_id, lcp}` so the caller reuses the match
  it just computed (~69 µs at 32k), and `length(tokens)` is computed once instead
  of six times.
- **A busy slot no longer holds three copies of its prompt.** The previous
  request's `cached_tokens` stayed reachable alongside the new prompt's list *and*
  its tuple for the whole request — ~2.6 MB of garbage at `n_parallel: 8` with 8k
  prompts.
- **Model swaps no longer trigger a global GC.** The `Server` model handle moved
  from `:persistent_term` to a `Registry`. `:persistent_term.put/1` and `erase/1`
  each force every process in the VM to scan its heap, and `ModelManager`'s
  load/unload path did exactly that per swap. The `Registry` entry is also removed
  automatically when the server dies, closing the leak `:persistent_term` had on
  `Process.exit(server, :kill)`, where `terminate/2` never runs.

### Added

- **`LlamaCppEx.Options`** — single owner for cross-module option policy: the
  scalar `:timeout` defaults (`blocking_timeout/0` = 60_000 for calls that block
  until generation completes, `stream_timeout/0` = 30_000 for per-chunk stream
  waits) and unknown-key validation. There were six hand-rolled `:timeout`
  defaults across three modules, and `stream_chat_completion/3` chose between
  60s and 30s purely on the type of its first argument.
- **`LlamaCppEx.Server.PromptCache`** — the level-2 RAM prompt cache extracted
  from `LlamaCppEx.Server` with its state in a `%PromptCache{entries, bytes,
  budget_bytes}` struct. The two functions that were `@doc false`-public solely so
  tests could reach them are now this module's real API (`covers?/4`,
  `evict_to_budget/1`), and `best_candidate/4` — previously private and therefore
  untestable — is public.
- **`LlamaCppEx.Server.Request`** — replaces a positional 7-tuple that was
  destructured at five sites and pattern-matched by index inside two
  `:queue.filter/2` callbacks. `init_slot/9` takes it instead of seven positional
  arguments, two of which were always `nil`.
- **`LlamaCppEx.Generator`** — the streaming-NIF lifecycle protocol (cancel flag →
  `spawn_link` → `request_cancel` → `unlink` → `exit(:kill)` → drain), which was
  spelled out at three call sites. Killing the runner alone does not interrupt a
  running NIF, so a missed cancel flag means the dirty scheduler keeps decoding to
  `max_tokens` for a departed consumer.
- **`Server.request_option_keys/0`** and `Server.fetch_model/1`.
- **CI now runs real inference.** CI compiled the 2100-line NIF but never ran a
  single forward pass, while 61 of 223 commits bumped `vendor/llama.cpp`. A new
  `inference` job downloads two revision-pinned GGUF models (SmolLM2-135M-Instruct
  and bge-small-en-v1.5, ~181 MB, cached) and runs the `:smoke` and `:embeddings`
  suites: greedy generation, grammar-constrained generation, the batching server,
  and embeddings.
- **CI also runs `mix credo --strict` and `mix test --cover`**, and the
  `--warnings-as-errors` compile is no longer skipped on a cache hit — the cache
  key now includes `lib/**`, so the warnings gate was previously dead for exactly
  the changes most likely to introduce warnings. The dialyzer PLT cache now
  actually hits (`:dialyzer` `plt_local_path`/`plt_core_path` point at the
  `priv/plts` that CI caches; dialyxir was writing under `_build`).

### Changed

- **llama.cpp submodule** — Updated from ff067f76d to 992c32532 (45 commits, tag
  b10178). Three binding-relevant headers changed, one of them behaviour-changing:
  - `include/llama.h` — **behaviour change**: `enum llama_load_mode` gains
    `LLAMA_LOAD_MODE_MMAP_MLOCK` and renumbers `_DIRECT_IO` from 3 to 4.
    `LLAMA_LOAD_MODE_MLOCK` previously meant "mmap **and** mlock" and now means
    mlock *without* mmap, with the new `_MMAP_MLOCK` covering the combination
    (#26135). The NIF selects the mode by name so the renumbering is transparent,
    but the redefinition is not: `model_load/10` now maps `:use_mlock` +
    `:use_mmap` to `_MMAP_MLOCK` and `:use_mlock` alone to `_MLOCK`, which is what
    the Breaking entry above describes. Without that change `use_mlock: true`
    would have silently stopped memory-mapping.
  - `common/common.h` — the free functions `common_context_seq_rm`,
    `common_context_seq_add` and `common_context_seq_cp` became `static` and were
    replaced by a `common_memory` struct carrying target/draft contexts and
    exposing them as methods (#26221). The binding never called them — it drives
    `llama_memory_*` directly and only reads `common_context_can_seq_rm`, whose
    signature and `common_context_seq_rm_type` enum are both unchanged — so the
    refactor is inert here. A new `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` enum
    value and its `need_n_rs_seq()` branch (#25173) are likewise inert: the
    binding sets `types` explicitly for its MTP path.
  - `common/chat.h` — additive only: a `COMMON_CHAT_FORMAT_PEG_MINIMAX_M3` format
    for the new MiniMax-M3 parser (#26210).
  - `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`,
    `common/json-schema-to-grammar.h`, `common/sampling.h` and
    `common/speculative.h` are untouched in this range.
  - **llama core / models**: MiniMax-M3 (MiniMax Sparse Attention) support
    (#24908) and its indexer tensors kept at F32 for speed and accuracy (#26144);
    Nanbeige4.2 (#25994); a Laguna-S-2.1 `LLM_TYPE` (#26233); NextN/MTP
    speculative decoding for GLM_DSA / GLM-5.2 (#25980).
  - **common / speculative**: DSpark speculative decoding (#25173); eagle3-v3
    support for gpt-oss (#25794); `common_print_available_devices()` (#26170);
    explicit `-md` precedence over draft sidecar resolution (#26165); a
    `subproc.h` wrapper, disabled on android/ios (#26102); NextN (MTP) blocks now
    counted in `n_gpu_layers` so front layers stay on GPU (#26177).
  - **ggml / Metal**: an FWHT kernel for the Metal backend (#25924); view-src
    output is now set (#25729); op offloading logic adjusted to prefer the
    weight's backend (#25832).
  - **mtmd**: GLM-5.2-Vision (#26126); MiniMax-M3 vision (#25113); Nemotron 3
    Nano Omni / parakeet (#22520); MiMo-V2.5 RVQ audio input (#26190).

### Tests

- **67 of 268 test cases never compiled.** Five compile-time
  `if @model_path && File.exists?(@model_path)` gates meant a quarter of the suite
  rotted invisibly — the `server.ex` decomposition shipped with its 40 tests
  uncompiled. Those gates are now `:smoke` / `:embeddings` / `:slow` / `:mtp` tags,
  each naming the one model it needs, and every case compiles.
- Dead `@tag :skip` + `flunk(...)` placeholders are deleted: `:skip` means the body
  never runs, so the `flunk` message reached nobody.
- Whole-body `if capability do` wrappers are gone. The `chat` describe set
  `has_template: false` and then skipped every assertion *inside a passing test*.

## v0.8.39

Maintenance release: llama.cpp bump to b10133, on top of b10075 from v0.8.38.
Unlike recent bumps this range **breaks the upstream C API**, so one NIF change
was required — see the `model_load/10` entry below. The public Elixir API is
unchanged. Full suite against the rebuilt NIF with real GGUF models (smoke, slow
and MTP speculative-decoding tests all included): 263 passed, 0 failures, over 12
consecutive runs.

### Changed

- **llama.cpp submodule** — Updated from 76f46ad29 to ff067f76d (58 commits, tag b10133). Two binding-relevant headers changed, one of them breaking:
  - `include/llama.h` — **breaking**: `llama_model_params` loses the `use_mmap`, `use_direct_io` and `use_mlock` booleans; they are replaced by a single `enum llama_load_mode load_mode` field with values `LLAMA_LOAD_MODE_NONE` / `_MMAP` / `_MLOCK` / `_DIRECT_IO` (`none`/`mmap`/`mlock`/`dio`), plus new `llama_load_mode_name` / `llama_load_mode_from_str` helpers (#20834). Note `LLAMA_LOAD_MODE_MLOCK` means "mmap **and** mlock" — the two are no longer independent.
  - `common/chat.h` — `common_chat_params::thinking_end_tag` (`std::string`) became `thinking_end_tags` (`std::vector<std::string>`) so the reasoning-budget sampler can accept multiple end sequences (#25544). The binding never read that field, so no change was needed.
  - `common/common.h` changes (`common_params::load_mode`, `reasoning_budget_end` widened to `std::vector<llama_tokens>`, new MCP server config fields) do not affect the binding, which does not use `common_params`. `common/json-schema-to-grammar.h`, `common/speculative.h`, `common/sampling.h` and every `ggml/include/` header are untouched in this range.
  - **llama core / models**: add GLM 5.2 Indexer support (#25407); add support for Laguna XS.2 & M.1 (#25165); assorted llama bug fixes (#26051); fix DeepSeek4 APE tensor op in llama-arch (#25945); fix the crafted DeepSeek4 template (#25414); fix the reasoning-preserve variable for DS4 (#25999); cohere2 MoE template parser enforces the JSON schema for text responses when a response schema is provided (#26018); synchronize save-load-state generation in the tests (#26056).
  - **common**: support multiple end sequences in the reasoning budget sampler (#25544); fix a use-after-free when loading a LoRA adapter fails (#25611); skip the empty implicit default preset (#25643); infer the speculative type from draft-repo sidecars (#25989) and resolve a draft repo to its requested sidecar (#25955).
  - **ggml**: declare `gguf_writer_base`'s destructor virtual (#25867); enable PowerPC backend variants on AIX (#25983); add the `GGML_BACKEND_DL_IMPL` invocation for the OpenVINO backend (#25795).
  - **Metal**: add f16 type support to leaky relu (#25981).
  - **CUDA**: `GET_ROWS` for quantized types (#25962); vectorize same-type `get_rows` with an int4 copy (#25929); improve NVFP4 W4A4 activation quantization (#25730); add `sqrt_softplus` in topk-moe for dsv4 (#25896); fix external compilation of q1_0 MMQ (#25778).
  - **Vulkan**: refactor `vk_queue` to use per-instance mutexes and unique handles (#23570).
  - **HIP**: remove rocWMMA FlashAttention (#26046).
  - **WebGPU**: add a CONV_2D_DW (depthwise conv2d) kernel (#25847); fix WASM compilation with OpenMP (#25943).
  - **OpenCL**: cache compiled `cl_program` binaries on disk (#26050); do not treat NULL-mask flash attention as causal (#25771).
  - **hexagon**: further pipeline improvements to the core bits (L2, DMA, MM, FA) (#26049); partial im2col support (#26007); activation ops update (#25974); check tensor type when reusing descriptors (#25968); fix a Windows crash when `op_poll` is enabled (#26029).
  - **kleidiai**: warn once when a weight type has no KleidiAI kernel (#25701).
  - **mtmd**: use RAII for setting and resetting non-causal attention (#25723); use `align_corners` for qwen3vl vision position embedding interpolation (#25781).
  - **convert**: fix the non-MoE NomicBert GGUF conversion error (#25996); handle the HunyuanVL XD-RoPE config (#25514).
  - **tools/server/ui** (not linked into the binding): MCP stdio support (#26062) and MCP display-name conflict fix (#26011); `"reasoning_effort": "none"` in the OAI API (#26045); a `format` arg on the datetime tool (#26117); missing `adaptive_target`/`adaptive_decay` task parameters in `generation_settings` (#25830); return 400 instead of 500 on validation errors with `X-Conversation-Id` (#25760); properly handle a null `llama_context` (#25868); reduced per-token render cost while streaming (#26053); assorted web UI fixes.
  - **vendor / ci**: update cpp-httplib to 0.51.0 (#26067) and `subprocess.h` (#26061); fix the SYCL package shared-library lookup (#25987).
- **NIF `model_load/10`** — Now maps the existing `:use_mmap` / `:use_mlock` / `:use_direct_io` options onto the new `llama_load_mode` enum instead of setting the three removed booleans. The documented precedence is preserved (direct I/O takes precedence over mmap, and mlock implies mmap): `dio` > `mlock` > `mmap` > `none`. The Elixir API and its defaults are unchanged, so no caller updates are needed; all four resolved modes were verified against a real model load.

### Fixed

- **Documented the Metal teardown abort in the smoke-test instructions (exit 134)** — On Metal, a fully green suite could still abort while the VM shut down, printing `263 passed, 0 failures` and then exiting 134 with `ggml-metal-device.m:622: GGML_ASSERT([rsets->data count] == 0) failed`. llama.cpp's Metal device is owned by a function-local `static std::vector`, so it is destroyed by `__cxa_finalize_ranges` *after* the BEAM calls `exit(3)`, and its destructor asserts that the global `MTLResidencySet` collection is empty. The BEAM makes no promise that NIF resource destructors have run by then, so a model or context still holding Metal buffers trips the assert; it reproduced on 3 of 12 all-green full-suite runs. `test/test_helper.exs` now explains the mechanism and the documented smoke-test commands set `GGML_METAL_NO_RESIDENCY=1`, which stops the collection from being allocated at all so `ggml_metal_rsets_free` returns early — removing the assert rather than racing it (12 of 12 clean runs). A residency set is only an OS memory-residency hint, so buffer allocation and compute are unchanged. The variable has to come from the shell: `System.put_env/2` does not reach the C `getenv` ggml reads. Nothing in the library changed — production keeps the upstream default, and CI is unaffected either way because it runs on Linux with `LLAMA_BACKEND=cpu` and loads no models. The same exit-ordering race applies to any node that halts while a model or context is still referenced; `backend_free/0` does not help, as it only reaches `ggml_quantize_free()`.

## v0.8.38

Maintenance release: llama.cpp bump to b10075. Full suite against the rebuilt
NIF with the generation smoke tests included (real GGUF model): 191 passed, 0
failures (4 skipped; embedding smoke tests opt-in and left skipped).

### Changed

- **llama.cpp submodule** — Updated from 571d0d540 to 76f46ad29 (7 commits, tag b10075). No NIF changes were required: every binding-relevant header is untouched — `include/llama.h`, `common/chat.h`, `common/common.h`, `common/json-schema-to-grammar.h`, and `ggml/include/ggml.h` are all unchanged. The only header edits in this range are two internal hexagon backend headers (`ggml/src/ggml-hexagon/htp/htp-ops.h`, `unary-ops.h`), which are not part of the public API and not linked into the macOS/Linux binding.
  - **llama core**: DeepSeek V4 KV-cache state serialization now writes only the used rows rather than the full cache (#25325).
  - **hexagon**: add CLAMP op (#25934).
  - **OpenCL**: support broadcast for Adreno `MUL_MAT` and honor `view_offs` for Adreno Q8_0 `MUL_MAT` in llama-server multi-stream (#25910).
  - **tools/server/ui** (not linked into the binding): sidebar conversation bulk actions plus settings logic/UI improvements (#25815); enable the agentic flow when only the JS sandbox is active (#25865); fix Settings/Display tool-call content toggle (#25783); fix collapsed user bubble with markdown rendering (#25869).

## v0.8.37

Maintenance release: llama.cpp bump to b10068. Full suite against the rebuilt
NIF with real GGUF models (smoke tests included): 252 passed, 0 failures.

### Changed

- **llama.cpp submodule** — Updated from b2dd28a3b to 571d0d540 (16 commits, tag b10068). No NIF changes were required: `include/llama.h`, `common/chat.h`, `common/common.h`, and `common/json-schema-to-grammar.h` are all untouched, and the remaining header changes are additive only — `ggml/include/ggml.h` gains three DeepSeek V4 hyper-connection ops (`GGML_OP_DSV4_HC_COMB`/`_PRE`/`_POST`) with matching `ggml_dsv4_hc_comb`/`_pre`/`_post` API functions, `common/download.h` gains `download_eagle3`/`download_dflash` options plus eagle3/dflash plan fields, and `ggml-rpc.h` only bumps its protocol patch version (not linked into the binding).
  - **llama core / models**: DeepSeek V4 fused hyper-connection ops — a richer stream-based replacement for the plain residual connection (#25585); rotate injected K/V cache for DFlash (#25823); llama-quant excludes the i32 `ffn_gate_tid2eid` routing table from quantization (#25787).
  - **ggml**: version bumped to 0.17.0 (ggml/1568) plus a ggml sync; ggml-blas defaults hadamard mul_mat to the CPU routine (#25710); initialize all tensors in `test_dsv4_hc` to avoid NaNs in sentinel tensors (#25822).
  - **Vulkan**: Q2_0 support (#25430).
  - **OpenCL**: q6_K MoE GEMM kernel loaded from the binary kernel lib (#25797); MoE dp4a activation tiles read/written to local memory as 128-bit vectorized LD/ST for Adreno (#25810); transposed q4_K noshuffle scales for coalesced reads (#25805); q4_K/q5_K flat mv loads quants as uint for Adreno A7x (#25780); ABS op (#25115); Adreno 810 usage note in docs (#25786).
  - **SYCL**: fix row calculation when `K_QUANTS_PER_ITERATION` is 1 (#25690).
  - **common** (download helpers, not linked into the binding): auto-download dflash- and eagle3- HF sidecars (#25811).

## v0.8.36

Maintenance release: llama.cpp bump to b10052. Full suite against the rebuilt
NIF: 178 passed, 0 failures (4 skipped, 8 smoke tests excluded — they need real
GGUF models).

### Changed

- **llama.cpp submodule** — Updated from 4f37f5197 to b2dd28a3b (85 commits, tag b10052). No NIF changes were required: `include/llama.h` is untouched and every other binding-relevant header change in this range is additive only — `common/common.h` gains a `LLAMA_EXAMPLE_TOKENIZE` enum value, server CORS fields, `tokenize_*` fields, and an `id_task` field on `common_prompt_checkpoint`; `ggml/include/ggml.h` gains a `GGML_OP_LIGHTNING_INDEXER` op plus `ggml_is_contiguous_to_{1,2,3}` and `ggml_lightning_indexer`; `ggml-cpu.h` gains `ggml_cpu_has_sme2`; `gguf.h` gains a `gguf_get_tensor_ne` shape accessor; `ggml-rpc.h` only bumps its protocol patch version (not linked into the binding).
  - **llama core / models**: add Hy3 (`hy_v3`) with MTP speculative decoding (#25395); Minimax2 eagle3 speculative support; DeepSeek V4 fixes — reduce graph splits (#25702), fix `seq_rm` (#25588), clear cache per-seq rather than full (#25521); tensor-parallel fixes for Phi3/Bert/Plamo2-3/ChatGLM (#25536); fix crash with draft-simple (#25720); fix reasoning leak with force-opened bare `<think>` templates (#24674).
  - **ggml**: add `GGML_OP_LIGHTNING_INDEXER` for the DeepSeek V3.2/V4 lightning indexer (#24231); add inner-dimension contiguity check functions (#25650); uniformize im2col dst_type across conv ops (#23660); add f16 `out_prod` (CPU) and `out_prod` op for Vulkan (#23997); support f16 as `SET_ROWS` src for Vulkan/CPU (#25432).
  - **gguf**: add tensor shape accessor `gguf_get_tensor_ne` (#24405); reject empty metadata keys (#24917).
  - **Metal**: fuse snake activation (mul, sin, sqr, mul, add) (#25459); add Q2_0 support (#25419).
  - **CUDA**: LIGHTNING_INDEXER kernel — generic vector + wmma (#25545); CUDA Virtual Devices (#25228); CUDA graphs on Volta/Turing (#25749); dedup MoE gate/up activation quantization (#25441); relax tensor contiguity for quantized concat (#25678); MMQ kernel config refactor (#24127); don't crash querying memory on a device with no free memory (#25157).
  - **Vulkan**: native e2m1/e4m3 conversions for mxfp4/nvfp4 (#25338); sync on event_wait for transfer-queue async copies (#25229); route large matmuls to medium tile on Adreno (#24877).
  - **OpenCL**: int8 dp4 dense + MoE prefill optimization for Adreno (#25537); assorted Adreno a7x/850 flash-attention and MoE fixes/guards (#25745, #25698, #25697, #25671, #25640, #25639, #25673).
  - **SYCL**: flash attention via oneDNN XMX engine (#25222); fused top-k MoE (#25217); xielu op (#25550); get_rows Q2_K/Q4_K/Q5_K fix (#25656); Q2_K DMMV reorder path (#25064).
  - **kleidiai / hexagon**: SME2 f32 kernel (#24414) and SME-vs-SME2 dispatch (#25478); hexagon L2 cache rework with lazy flushing (#25762) and hmx-queue enum-narrowing fix (#25677).
  - **mtmd**: fix silent prompt truncation on embedded NUL (#25548).
  - **convert**: accept `BitNetForCausalLM` architecture name (#25769); fix dflash target tokenizer mismatch (#25733); split MTP export for HY V3 (#25641).
  - **tools/server/ui** (not linked into the binding): `--cors-*` options (#25655) and ignore empty `Origin` headers (#25756); per-request `reasoning_budget_tokens` in chat completions (#23116); prompt-cache state ownership refactor (#25649); text-only slot save/restore with mtmd (#25076); fix dropped image blocks in tool_result during Anthropic/OpenAI conversion (#22536); `tokenize` tool aligned to common args (#25516, #25672); assorted web UI / MCP fixes.

## v0.8.35

Maintenance release: llama.cpp bump to b9967. Full suite against the rebuilt
NIF: 252 tests, 0 failures.

### Changed

- **llama.cpp submodule** — Updated from a646006f0 to 4f37f5197 (35 commits, tag b9967). No NIF changes were required: the only header change in this range is additive — a new `ggml/include/ggml-et.h` for the initial ET backend; `llama.h`, `common/chat.h`, `common/common.h`, `common/sampling.h`, `common/speculative.h`, and `common/json-schema-to-grammar.h` are all untouched.
  - **llama core**: make all KQ masks f16 when flash attention is used, remove zero attention bias and raw_k repeats in DeepSeek V4 (#25370); make tensor-split regex patterns static (#24710); add llama-batch unit test (#25471).
  - **ggml**: initial ET backend (#24179); process data in smaller chunks in CUDA `ggml_top_k()`/`ggml_argsort()` to reduce temporary buffer memory (#24776); fix depthwise conv2d (#25490); ggml syncs including `ggml_vqtbl1q_u8` for 32-bit compat.
  - **Metal**: add CONV_2D_DW (depthwise convolution) support (#21565).
  - **CUDA**: mmvq indexing simplification with always multiply/add (#25445); align snake fusion matcher with other backends (#25460).
  - **HIP/WebGPU/OpenCL**: enable `-funsafe-math-optimizations` on HIP (#24668); tune subgroup split in WebGPU `flash_attn_vec` (#25418); OpenCL cluster-parallel decode FA for Adreno (#25473) and Q6_K GEMM/GEMV fix for weight `ne01` not a multiple of 128 (#25464).
  - **hexagon**: ARGSORT performance for small tensors (#25512); tiling, tracing and optimizations for unary ops (#25474); VISION RoPE support (#25216).
  - **mtmd**: deepseek-ocr v1 multi-tile (#24717).
  - **tools/server/ui** (not linked into the binding): accept null sampling params (#25538); respect min-step when splitting prompt batches (#25420); move chat-template thinking probe inside the init try/catch (#24093); prevent duplicate speculative model downloads (#25527); `llama-cli --output` option (#25484) and crash fix on wrong server base URL (#25497); assorted web UI improvements.

## v0.8.34

Maintenance release: llama.cpp bump to b9932. Full suite against the rebuilt
NIF: 191 tests, 0 failures.

### Changed

- **llama.cpp submodule** — Updated from cb295bf59 to a646006f0 (44 commits, tag b9932). No NIF changes were required: the header diffs in this range are additive only — new `GGML_TYPE_Q2_0`/`GGML_FTYPE_MOSTLY_Q2_0`/`LLAMA_FTYPE_MOSTLY_Q2_0` enum values in `ggml.h`/`llama.h`, a new `common_speculative_init_result` helper in `common/speculative.h`, and a `server_base` field plus `<fstream>` include in `common/common.h`.
  - **llama core**: fix allowed decreasing positions in a sequence in llama-batch (#25449); add `n_keep_tail` in `split_equal` for recurrent models (#25278); refactor fused ops (#24646); fix quantized KV cache for dsv4 (#25202); fix OOB reads in the UGM tokenizer's `precompiled_charsmap` handling (#18750).
  - **speculative**: fix out-of-bounds read in ngram-map on prompt shrink (#23936); fix draft-model fit vs load inconsistency in the server (#25056); naming/spacing cleanup (#25410).
  - **ggml**: add Q2_0 quantization type definition + CPU backend (#24448); CPU f16→f16 `GGML_OP_SET_ROWS` (#25344); fix A-indexing in the simd_gemm scalar tail-column path (#25390); make `ggml_time_init` idempotent (#24422); better default thread count on ppc/AIX (#25237).
  - **Metal**: add `set_rows` with f16 src0 (#25434); add `col2im_1d` op for f32/f16/bf16 (#25176).
  - **CUDA**: f16→f16 `SET_ROWS` (#25367); fuse MMVQ post-scale for NVFP4 (#24481); remove `-sm row`, refactor cuBLAS (#24216).
  - **Vulkan**: disable FA `mask_opt` on GCN (#24362); reduce submission threshold on small AMD GPUs by CU count (#25240); guard unimplemented f16 `SET_ROWS` (#25351).
  - **OpenCL**: ragged-tile MoE prefill FP16 GEMM optimization (#25433); flash-attention decode perf (#25366); fix potential crash in aos reconstruct (#25383).
  - **SYCL/HIP/hexagon**: eight SYCL commits (col2im_1d, cross-entropy-loss ops, argsort coverage, AOT double fix, env-var renames); HIP `-fno-finite-math-only` alongside `-ffast-math` (#25373); hexagon VTCM layouts + pipeline improvements for MUL_MAT/MUL_MAT_ID/FLASH_ATTN_EXT (#25425).
  - **tools/server/ui** (not linked into the binding): llama-cli moved to an HTTP-based implementation (#24948); SSE replay-buffer follow-up (#25047); timings/progress in `/responses` API streams (#25348); prompt-cache RAM limit enforcement (#25070); fix `load_models()` deadlock (#25358); context-usage gauge in the web UI (#25340).

### Added

- **Speculative-decoding docs** — README and `LlamaCppEx.MTP` moduledoc now state which upstream speculative types the binding exposes (MTP only) and document DFlash status on Apple Silicon: functional end-to-end on Metal at b9932 via upstream tools, but measured slower than plain decoding at small target sizes (Qwen3.5-4B + 0.6B drafter on M4 Max: 42 tok/s vs 85 tok/s plain, 30% acceptance greedy), with community drafter-GGUF conversions still incompatible across converter revisions (#25116, #25110).

## v0.8.33

Performance and robustness release for the batching `Server` and generation paths
(the `perf-batching-prefix-cache` plan: 22 tasks across hot-loop, prefix-caching,
API-routing, and correctness phases). Benchmarked before/after on an M1 Max
(`bench/results/v0.8.32-e6e1ef1-baseline-perf-m1max.md` vs
`bench/results/v0.8.32-perf-branch-final-m1max.md`). Headline: with 8 interleaved
conversations sharing a system prompt on 4 slots, TTFT median drops 115 → 35.5 ms
(3.2x) at a ~75% prefix-cache hit ratio; server-routed multi-turn `chat_completion`
is 1.6x faster than the stateless path. Full suite: 252 tests, 0 failures.

### Added

- **Cross-slot prefix sharing** — with unified KV (`kv_unified: true`, new `Server`
  and `Context.create/2` option), a prefix cached by any slot (idle or still
  generating) is adopted by other slots via a metadata-only `llama_memory_seq_cp`;
  a shared system prompt prefills once, ever. Gated off automatically on
  configurations where it is unsafe (split KV, hybrid-GDN models).
- **Session affinity** — `session:` request option pins a conversation to the slot
  holding its cache; freed slots serve queued requests session-first.
- **Level-2 RAM prompt cache** — `prompt_cache_ram_mb:` server option (default 0 =
  off): slot KV states about to be destroyed are serialized to RAM (new
  `llama_state_seq_get_size/get_data/set_data` NIFs) and restored later instead of
  re-prefilling; FIFO eviction under a byte budget checked before copying.
- **Per-request options** — `:cache_prompt`, `:session`, and all sampling params
  (`:temp`, `:seed`, `:top_k`, `:top_p`, `:min_p`, penalties, `:grammar`) now work
  per request on `Server.generate/stream/generate_tokens/stream_tokens`, overriding
  server defaults; each request gets a fresh sampler chain.
- **Server-routed chat completions** — `LlamaCppEx.chat_completion/3` and
  `stream_chat_completion/3` accept a running `Server` (pid or name); templating and
  tokenization happen in the caller and generation uses continuous batching with
  prefix caching. New `Server.complete_tokens/3` returns
  `%{text, completion_tokens, finish_reason}`. `ModelManager` gains matching
  `chat_completion/3` and `stream_chat_completion/3` routing.
- **Cancellation** — the server monitors each request's consumer and frees the slot
  when it dies; halting a `Server.stream/stream_tokens` early cancels generation
  (`Server.cancel/2` also public). Stateless `generate`/`stream`/MTP loops carry a
  cancel-flag NIF resource installed as `llama_set_abort_callback`, so even a long
  prefill aborts mid-decode instead of running to completion for a departed caller.
- **`:max_queue` backpressure** — the previously documented but unenforced option now
  rejects overflow with `{:error, :queue_full}` immediately; streams surface it (and
  mid-generation errors) as a single `{:error, reason}` element.
- **KV-pressure recovery** — on `llama_decode == 1` the fused tick NIF purges idle
  slots' cached KV, then recursively halves the batch; a sequence that still cannot
  fit one token fails alone with `{:error, :context_full}` while other slots keep
  generating. New `[:llama_cpp_ex, :server, :kv_pressure]`, `[:..., :ram_cache]`, and
  `[:..., :prefix_instability]` telemetry events.
- **Embeddings for Nx** — `format: :binary` on `Embedding.embed/embed_batch` returns
  the raw native-endian f32 binary (`Nx.from_binary(bin, :f32)`), skipping the boxed
  float list entirely.
- Tests: `test/server_test.exs` (pure slot-pick/session/donor/RAM-cache logic),
  `test/server_smoke_test.exs` (cache semantics, affinity, backpressure,
  cancellation, overflow isolation), `test/utf8_stream_test.exs`; two new bench
  scripts (`bench/prefix_cache_concurrent.exs`, `bench/chat_completion_server.exs`).

### Changed

- **llama.cpp submodule** — Updated from 2d973636e to cb295bf59 (18 commits, tag b9888). No NIF changes were required: every header the binding compiles against — `include/llama.h`, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/common.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/speculative.h` — has zero diff in this range (nothing under `include/` or `common/` changed at all). The range only touches `llama` core `src/` and the statically-linked `ggml` backend implementations (CUDA, HIP, Vulkan, CPU) plus webui/scripts/server-test code the NIF does not link against, so the bump is picked up purely by resyncing the static libraries on rebuild.
  - **llama core**: guard the K/V rotation input when the buffer is unallocated (#25215); fix stale tensor-split params for draft models (#24814).
  - **ggml**: fix a tensor-parallel + `-ncmoe` crash on MoE models (#25028); abort on a multi-buffer in `ggml-backend-meta` (#25276); fix the broken CPU concat implementation for quantized types (#25247).
  - **CUDA**: extend K-type validation to V-types for flash attention (#24403); add a concat implementation for quantized types (#25303); optimize `conv_transpose_1d` indexing (#25310); VMM-pool allocation Turing P2P access fix (#24491).
  - **ggml-cpu**: use a UE4M3 LUT in the ARM NVFP4 dot product (#25331); enable tiled matmul on AIX (#25199).
  - **ggml-hip**: enable `-ffast-math` for HIP builds (#23862).
  - **vulkan**: fix a 32-bit integer overflow in `CEIL_DIV` (#25245).
  - **ui/scripts/tests** (not linked into the binding): restore the Ctrl+B sidebar toggle (#25307), fake a 200 for proxy `DELETE` requests (#25298), and add sync blocks so display/behavior settings honor `--ui-config-file` (#25132); use `HF_TOKEN` when downloading UI assets (#25280); temporarily skip the model-downloading API test (#25355).
- **Server hot loop** — one fused dirty-CPU NIF per tick (`batch_eval_sample`:
  decode + per-slot sampling + detokenization + EOG check) replaces the previous
  1 dirty + (2 normal-scheduler NIFs + send) × generating-slots pattern; streamed
  pieces are delivered one tick earlier; a persistent `llama_batch` is reused across
  decode NIFs; `sampler_sample`, `sampler_sample_at`, `chat_apply_template_jinja`,
  `json_schema_to_grammar`, and `speculative_init` are dirty-flagged.
- **Defaults** — `cache_prompt` on the `Server` now defaults to `true` (llama-server
  parity; per-request overridable); `kv_unified` defaults to `true` (slots share the
  full `n_ctx` budget instead of a fixed `n_ctx/n_parallel` split; no measurable
  throughput cost in A/B); `n_batch` defaults to `min(n_ctx, 2048)` (bounds
  worst-case tick latency).
- **Error shapes** — failure replies are atoms/tagged tuples: `:context_full`,
  `:queue_full`, `:empty_prompt`, `{:inference_failed, reason}` (previously
  formatted strings). A batch failure now fails only the affected request instead of
  every active slot.
- **Tokenization moved to the caller** — `Server.generate/stream` encode client-side
  using a `:persistent_term`-cached model handle; `Server.get_model/1` no longer
  round-trips the server mailbox.
- `Context.decode/2` uses an explicit batch requesting logits only for the final
  token (previously the last token of every `n_batch` chunk).

### Fixed

- **UTF-8-safe streaming** — pieces that split a multibyte codepoint across tokens
  are buffered and emitted whole at every emission point (server streams, stateless
  streams, chat-completion chunks); previously consumers could receive invalid UTF-8.
- A prompt exactly matching its cached prefix hung the slot until the call timeout
  (nothing left to prefill, no logits to sample); cached reuse is now capped at
  `prompt_len - 1` like llama-server, so the last prompt token is always re-decoded.
- A tiny incidental prefix match (e.g. 2 tokens) could destroy a long cached prefix
  via trim; own-slot, donor-slot, and RAM-cache candidates now compete by match
  length with cost-ordered tie-breaking.
- Abandoned streams no longer burn batch budget to `max_tokens` (see cancellation).
- `stream_tokens` with an empty token list now returns `{:error, :empty_prompt}`
  instead of hanging until the stream timeout.
- Embedding decodes clear only their own sequences instead of the whole context —
  safe if pointed at a shared context.

## v0.8.32

### Changed

- **llama.cpp submodule** — Updated from f708a5b2c to 2d973636e (24 commits, tag b9870). No NIF changes were required. `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/common.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/speculative.h` are all unchanged. The only touched header the binding compiles against is `include/llama.h`, and its diff is purely additive: a new `llama_ftype_name()` helper that renders a `llama_ftype` quantization enum as a string (e.g. `"Q8_0"`) and a new `llama_model_ftype()` getter that returns a loaded model's file type (#25134) — the binding calls neither, so nothing it consumes changed shape. The full test suite passes (216 tests + 1 skipped, with the generation paths run against a Qwen3.5-0.8B model and the embedding paths against a Qwen3-Embedding-0.6B model), all 7 end-to-end smoke tests pass (generation, streaming, chat templates, JSON-schema grammar, raw GBNF, and embeddings), formatting is clean, and Dialyzer reports 0 errors.
  - **llama API**: add `llama_model_ftype()` / `llama_ftype_name()` for reading a model's quantization type (#25134).
  - **model**: register `t_layer_inp` for qwen3next (#25141).
  - **chat**: trim messages sent to the StepFun parser, fixing long reasoning loops (#25238).
  - **spec/dflash**: support `spec-draft-p-min` in DFlash (#25246).
  - **CUDA**: enable topk-moe fusion for 288 experts (#25267); remove redundant copies after `gated_delta_net` (#23940); consistent use of `__restrict__` + PDL for FlashAttention (#25185); prevent integer truncation/overflow in the `flash_attn_mask_to_KV_max` kernel's KQ mask strides (#24945); fix `get_rows_back` for tables with more than 65535 rows (#25103); fix Gemma E4B MTP FlashAttention (#25148).
  - **ggml-cpu**: add AVX2 optimization for the nvfp4 dot product using a UE4M3 LUT (#23961).
  - **opencl**: allow loading precompiled binary kernels from a library (#23042); initial q1_0 support (#25160).
  - **hexagon**: flash-attention rework with optimizations and accuracy improvements (#25085).
  - **common/server**: use the HF primary split as the model path (#25194); handle bracketed IPv6 literals in URL authorities (#25140); ping silent SSE streams every 1s and kick only after 3s so slow prefill never drops healthy connections (#25241); update the vendored cpp-httplib to 0.49.0 (#25218).
  - **ui**: improve streaming performance (#25225); strip path and weight extension from the model id in single-model mode (#25137); align persisted config with the strict server schema and enable thinking by default (#25242); add an MCP-servers opt-in for first-time visitors (#25239); prevent tool messages from appending to other conversations (#25177); remove the PWA navigate fallback to avoid caching API endpoint requests (#25174).

## v0.8.31

### Changed

- **llama.cpp submodule** — Updated from 9bebfcb4b to f708a5b2c (20 commits, tag b9846). No NIF changes were required. `include/llama.h`, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/speculative.h` are all unchanged. The only touched header the binding compiles against is `common/common.h`, and its diff does not reach anything the binding consumes: a block of internal `COM_*` logging macros (`COM_DBG`/`COM_TRC`/`COM_INF`/`COM_WRN`/`COM_ERR`/`COM_CNT`) is added, a new `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` value is appended to the `common_speculative_type` enum (the DFlash speculative-decoding work below), and `common_params_speculative::need_n_rs_seq()` is extended to also reserve a recurrent-state seq for that new DFlash draft type. The binding only constructs `common_params_speculative` for its MTP path (setting `types` and `draft.*`) and otherwise calls `common_chat_templates_*`, `common_context_can_seq_rm`, `common_batch_add`, and `json_schema_to_grammar` — so the new enum value and the internal `need_n_rs_seq` logic are inert for it. The full test suite passes (158 tests + 4 skipped, with the generation paths run against a Qwen3.5-0.8B model and the embedding paths against a Qwen3-Embedding-0.6B model), all 7 end-to-end smoke tests pass (generation, streaming, chat templates, JSON-schema grammar, raw GBNF, and embeddings), formatting is clean, and Dialyzer reports 0 errors.
  - **model**: add DeepSeek V4 (#24162); implement the MiniCPM5 chat parser (#24889).
  - **spec/dflash**: add DFlash speculative decoding (#22105); refactor DFlash draft-model conversion (#25110).
  - **CUDA/HIP**: add a `cudaMemcpy2DAsync` fast path to `ggml_cuda_cpy` (#25057); use hipBLAS for dense prefill on gfx900 while keeping MMQ for MoE (#24588).
  - **vulkan**: roll the bk loop in matmul for Asahi Linux (#24663); use flops instead of weight-tensor size for the submission heuristic (#25005).
  - **opencl**: flash-attention improvement (#25069).
  - **ggml-webgpu**: add NVFP4 support (#25143).
  - **sched**: revert "reintroduce less synchronizations during split compute" (#25138, reverting #20793 which had landed in v0.8.30).
  - **chat/jinja**: add a `--reasoning-preserve` flag (#25105); add `jinja --dump-prog` for debugging (#25086).
  - **common/server/cli**: dedup preset and cached model entries in `/v1/models` (#25131); remove the unused regex-partial helper (#25118); allow `--offline` in `llama download` (#25091); reduce logs (v2) (#25078).
  - **ui/tools**: restore Tailwind scanning in ignored worktrees (#24879); fix stop and reasoning skip in single-model mode (#25084); revert the hover-gated interactive-elements accessibility change (#25098, reverting #24727).

## v0.8.30

### Changed

- **llama.cpp submodule** — Updated from b3ce5cedf to 9bebfcb4b (37 commits, tag b9826). No NIF changes were required. `include/llama.h`, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/speculative.h` are all unchanged. The only touched header the binding compiles against is `common/common.h`, and its diff is server/CLI-only: a new `LLAMA_EXAMPLE_DOWNLOAD` enum value, `common_params_model::get_name()` gaining a `const` qualifier plus a new `empty()` helper, `common_params_speculative::has_dft()` refactored to call that helper, and the removal of the server-side `skip_download` flag — none of which the binding consumes (it only calls `common_chat_templates_inputs`, `common_chat_msg`, `common_chat_templates_apply`, and `json_schema_to_grammar`). This range also resyncs `ggml` (bumped to 0.15.3) underneath the statically-linked `ggml`/`ggml-backend` libraries the NIF links against, so the binding was rebuilt clean and re-exercised end to end. The full test suite passes (216 tests + 1 skipped, with the generation paths run against a Qwen3.5-0.8B model and the embedding paths against a Qwen3-Embedding model), all 7 end-to-end smoke tests pass (generation, streaming, chat templates, JSON-schema grammar, raw GBNF, and embeddings), formatting is clean, and Dialyzer reports 0 errors.
  - **ggml**: sync ggml and bump to version 0.15.3 (ggml/1550); address integer overflows in the binary-op CUDA implementation (#24706).
  - **CUDA**: add a `cublasSgemmBatched` mapping for the HIP/MUSA vendor headers (#25033); batch the `out_prod` broadcast (dps2>1) path with `cublasSgemmBatched` (#24426); various fixes to `cpy.cu` (#25000).
  - **vulkan**: fix the step operator for 0 input (#25036); optimize `mul_mat_vecq` for mi50 (#22933); add the `INTEL_XE1` arch enum and enable coopmat1 on Intel Xe-LPG Plus (#24404); work around a compiler bug in the conv2d coopmat2 path (#24924).
  - **SYCL**: fix the failed `norm` unit-test cases (#25044); clamp softmax input to avoid underflow (#24941).
  - **ggml-cpu**: fix the SVE leftover path in `ggml_vec_dot_f32` (#24699).
  - **opencl**: flush the profiling batch at shutdown for incomplete batches (#25016).
  - **model**: add a label for LFM2.5-230M (#25008); mamba2 — remove the hardcoded 2x expansion factor and the invalid `d_inner % d_state` check (#23082).
  - **sched**: reintroduce fewer synchronizations during split compute (#20793).
  - **common/server/mtmd**: refactor model handling (#24980); add more `mtmd` validations (#25013); server SSE replay buffer (#23226); return status code 403 for disabled server features (#24970).
  - **app/cli**: allow `--version`, `--licenses` & `--help` (#25054); add the `llama download` subcommand (#24982); fix handling of `--spec-draft-hf` and `--hf-repo-v` (#25043).
  - **build/devops**: include `libmtmd` in the Apple XCFramework (#21935); add `llama` to all docker images (#25035); update OpenVINO to OV 2026.2.1 with self-contained release packages (#24974); disable mtmd video on i/tv/visionOS in the xcframework (#25018); improve the `rpc-server` and `export-graph-ops` binary names (#25045).
  - **ui/ci/docs/tests/misc**: webui accessibility fix for hover-gated interactive elements (#24727) and the always-show-sidebar-on-desktop setting (#24979); CI windows-openvino in check-release (#25022); fix `test-chat-template --no-common` (#25075) and synchronize contexts at the end of `test-thread-safety` (#24935); Eagle3 qwen3 draft-model docs (#24977); labeler fixes (#25012, #24920).

## v0.8.29

### Changed

- **llama.cpp submodule** — Updated from dec5ca557 to b3ce5cedf (26 commits, tag b9789). No NIF changes were required. `include/llama.h`, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/speculative.h` are all unchanged. Two headers the binding compiles against did change, but neither touches the symbols the NIF uses: `common/common.h` only bumps a server-side default (`checkpoint_min_step` 256 → 8192), and `common/chat.h` reworks the chat message-splitting API (the `common_chat_msg_span.role` field becomes a `common_chat_role` enum, new `common_chat_msg_spans` / `common_chat_msg_delimiters` containers replace the free-standing `common_chat_split_by_role`, and `common_chat_params.message_spans` is renamed to `message_delimiters`) — all consumed by the server, not the binding, which only calls `common_chat_templates_inputs`, `common_chat_msg`, and `common_chat_templates_apply`. The implementation behind that last call, `common/chat.cpp`, was refactored as part of the same message-splitting work, so the chat-template smoke path was re-exercised against the freshly built NIF. The full test suite passes (158 tests + 4 skipped), all 7 end-to-end smoke tests pass (generation, streaming, chat templates, JSON-schema grammar, raw GBNF, and embeddings — the embedding paths exercised with a Qwen3-Embedding model), formatting is clean, and Dialyzer reports 0 errors.
  - **model**: add LFM2.5-ColBERT-350M and LFM2.5-Embedding-350M (#24913); Granite Speech Plus (#24818).
  - **quant**: fix quantizing MoE with MTP (#24986).
  - **chat**: harden the capabilities check (#24973).
  - **server**: check for draft-context creation errors (#24922); fix remote preset handling and add a test (#24938); improve user-message detection and create checkpoints at every user message (#24176).
  - **mtmd**: unlimited-OCR model converter plus a parity test (#24969).
  - **vulkan**: apply bias before softmax in flash attention to avoid overflow (#24909); allow reducing graph submission batches to avoid timeouts (#24872); support `GET_ROWS_BACK` (#24883) and `CONV_3D` (#24612); cover more backend tests for SQR/SQRT/SIN/COS/CLAMP/LEAKY_RELU/NORM (#24582); make `mul_mm` ALIGNED a spec constant (#24689); fail the build when a shader fails to compile (#24450); link ggml-cpu when `GGML_VULKAN_CHECK_RESULTS` / `RUN_TESTS` are enabled (#24444).
  - **SYCL**: support `--split-mode tensor` (#24152); fix the failed `conv_3d` unit-test cases (#24900).
  - **opencl**: support non-contiguous rows in `norm` (#24965); improve q8_0 gemv precision (#24923).
  - **hexagon**: MUL_MAT / MUL_MAT_ID rework — 32x32 tiled weight repack, kernel params, and cached graphs (#24954).
  - **ggml-webgpu**: improve MTP inference by using the mat-vec path for small batches (#24811).
  - **webui**: new logo plus navigation cleanup and mobile UI/UX improvements (#24897); loading bar below the model picker (#24931).
  - **common/build/docs**: remove the unused json-partial helper (#24968); add yomaytk to ggml-webgpu CODEOWNERS (#24930).

## v0.8.28

### Changed

- **llama.cpp submodule** — Updated from 845282461 to dec5ca557 (24 commits, tag b9763). No NIF changes were required. `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, `common/speculative.h`, and `common/common.h` are all unchanged. The only touched header the binding compiles against is `include/llama.h`, and the sole functional change there is one new accessor — `llama_model_n_layer_nextn` (the number of next-token / MTP prediction layers) — added alongside the existing `llama_model_n_layer`; the rest of the diff is whitespace realignment. The binding does not call the new function. Notably, this range refactors the grammar generators that the binding's `json_schema_to_grammar_nif` links against (the `common/peg` AC parser and the JSON-schema-to-GBNF spacing rules below) even though `common/json-schema-to-grammar.h` itself is unchanged, so both the JSON-schema and raw-GBNF smoke paths were exercised against the freshly built NIF. The full test suite passes (158 tests + 4 skipped), all 7 end-to-end smoke tests pass (generation, streaming, chat templates, JSON-schema grammar, raw GBNF, and embeddings — the embedding paths exercised with a Qwen3-Embedding model), formatting is clean, and Dialyzer reports 0 errors.
  - **model/quantization**: use `LLM_KV` for `quantization_version` & `file_type` (#24802).
  - **common/grammar**: implement an AC parser for stricter grammar generation (#24869); refactor the until→GBNF grammar generation (#24839); align `json-schema-to-grammar` spacing rules with the parsers (#24835) — these underlie the binding's structured-output / GBNF path.
  - **sampling**: remove the unconditional softmax+sort in the top-n-sigma sampler (#22645).
  - **jinja** (template engine used by chat templates): implement the `call` statement (#24847).
  - **MTP/speculative**: support Step3.5/3.7 flash mtp3 (#24340).
  - **mtmd**: fix `mtmd_get_memory_usage` (#24867); add a load-progress callback (#24865).
  - **server**: add an `id` to tool-call responses API (#24882); move model downloading to a dedicated process in the router (#24834); refactor/generalize the input-file schema (#24299); fix an `edit_file` crash on append at end of file (`line_start` -1) (#24893); report progress for loading spec models and add a "stages" list (#24870); refactor batch construction (#24843); add a "verbose" field to the schema (#24864); real-time model load-progress tracking via `/models/sse` (#24828).
  - **SYCL**: support bf16 on the `bin_bcast` op and unary ops (#24838).
  - **hexagon**: use a padded stride for ssm-conv weights (#24470).
  - **webui**: prioritize favorite models in model selection (#24766); show model status and load progress via the `/models/sse` feed (#24878).
  - **common/cli/build/docs**: stabilize the randomly-failing `test-args-parser` (#24826); add the `libandroid-spawn` dependency for Termux builds (#21812); whitespace clean-up (#24862).

## v0.8.27

### Changed

- **llama.cpp submodule** — Updated from 74ade5274 to 845282461 (67 commits, tag b9739). No NIF changes were required. `include/llama.h`, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, and `common/sampling.h` are all unchanged. `common/speculative.h` only gains two optional declarations — `common_speculative_get_state` / `common_speculative_set_state` (for stashing and restoring internal speculative state) — which the binding does not call. `common/common.h` changes do not touch the binding either: `common_params_model` swaps its `name` field for a `get_name()` method, the deprecated `webui` / `webui_mcp_proxy` / `webui_config_json` fields are dropped from `common_params`, a `models_preset_hf` field and an `fs_open_ifstream` helper are added, and `common_prompt_checkpoint` gains a `data_spec` blob — but the NIF constructs only `common_params_speculative` (setting `types` and `draft.*`), never `common_params` or `common_params_model`, and the sole `common_params_speculative` change is internal `need_n_rs_seq()` logic that now also reserves a recurrent-state seq for EAGLE3 drafts. The full test suite passes (158 tests + 4 skipped), all 7 end-to-end smoke tests pass against the freshly built NIF (generation, streaming, chat templates, JSON-schema grammar, raw GBNF, and embeddings — the embedding paths fully exercised with a Qwen3-Embedding model), formatting is clean, and Dialyzer reports 0 errors.
  - **model/convert**: load GLM-DSA indexer tensors as optional (#24770); more consistent `rope_parameters` handling in convert (#24833); skip `main_gpu` validation when no devices are available (#23405).
  - **MTP/speculative**: support EAGLE3 for Qwen3.5 & 3.6 (#24593); fix a segfault on long prompts for EAGLE3 (#24707).
  - **mtmd**: fix UTF-8 handling on Windows (#24779); several bug fixes (#24784); add batching support for InternVL (#24775) and for `mtmd-cli` plus video tests (#24778); refactor the preprocessor and add `mtmd_image_preproc_out` (#24736); refactor llava-uhd overview image handling to always use `ov_img_first` (#24769); stop using the batch dim in llava_uhd (#24732).
  - **server**: avoid forwarding auth headers in the CORS proxy (#24373); refactor child→router communication (#24821); optimize `get_token_probabilities` (#24796); fix an unbounded `n_discard` during context shifting (#24786); consolidate slot selection into `get_available_slot` (#24755); add an `X-Accel-Buffering: no` header to streaming endpoints (#24774); add request `schema` validation (#24150); return HTTP 400 on invalid grammar (#24154); add a last-5-seconds generation-speed display (#24291); router fixes — stopping-thread hang (#24728), forward args to child instances (#24760), rework `-hf` preset repos (#24739), add a model-management API (#23976); drop internal "webui" naming and add an `--agent` arg (#24817, #24801).
  - **ggml/ggml-cpu**: optimize AMX (#24806); sync ggml and bump to 0.15.2 (ggml/1548); power10 Q8/Q4 MMA matmul K-tail support (#24753); conditionally enable the power11 backend based on compiler support (#24687).
  - **CUDA**: add `col2im` 1D (#24417); revert "reset CUDA context after reading memory size" (#24715).
  - **metal**: check for BF16 support in the concat kernel (#24747); add f16/bf16 support to the concat operator (#24724); implement the `rope_back` operator (#24725).
  - **vulkan**: record actual memory properties during buffer creation (#24326).
  - **SYCL**: support `MUL_MAT`/`OUT_PROD` with Q1_0 (#24721); add `conv_2d`/`conv_2d_dw`/`conv2d_transpose` (#24600) and `conv_3d` (#24691); enable fp16 for SQR/SQRT/LOG/SIN/COS/CLAMP (#24692); add dev-to-dev memcpy via the SYCL API (#24476); fix a use-after-free with async memcpy in MoE prefill (#24676); optional USM system allocations (#22526); rename `GGML_SYCL_SUPPORT_LEVEL_ZERO` (#24719).
  - **webgpu**: add adapter toggles for F16 on Vulkan + NVIDIA (f449e0553).
  - **opencl**: optimize `mul_mat_f16_f32_l4` for decode (#24504).
  - **hexagon**: add op-trace — fine-grained HVX/HMX/DMA event tracing (#24592).
  - **openvino**: OV 2026.2, context-shift, Q5_1 support, Gemma4 dense/embedding, and `-fa off` (#24503).
  - **common/cli**: enforce `max_capacity` and optimize queue resizing in logging (#24490); support comment lines in `--api-key-file` (#23168); the `pi` interactive tool drops docs from its system prompt (#24791); enable app self-update only when built with `llama-install.sh` (#24754).
  - **webui**: export conversations as JSONL (#24688); touch-accessible model selection (#24604); fix SSE transport detection and routing through the CORS proxy (#24500).
  - **vendor/ci/docker/build/docs**: update cpp-httplib to 0.48.0 (#24787); build/prebuild the web UI in Docker (#24794, #24829) and fix the cmake UI build against a read-only source tree (#24752); CI fixes — check-release message parsing (#24751), Windows x64 OpenVINO release link (#24731), Vulkan docker images (#24595), Adreno arm64 OpenCL release link (#24809); fix the export-lora `--lora-scaled` docs (#24703).

## v0.8.26

### Changed

- **llama.cpp submodule** — Updated from 597b6672e to 74ade5274 (51 commits, tag b9672). No NIF changes were required. Every header the binding compiles against is unchanged: `include/llama.h`, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `common/chat.h`, `common/speculative.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/common.h` all have zero diff in this range — the cleanest bump in a while (even `common_params`, which gained a field in v0.8.24, is untouched). The full test suite passes (158 tests + 4 skipped), all 7 end-to-end smoke tests pass against the freshly built NIF (generation, streaming, chat templates, JSON-schema grammar, raw GBNF, and embeddings), formatting is clean, and Dialyzer reports 0 errors.
  - **model/vocab**: add arch support for Cohere2-MoE (#24260); register `cohere2moe` in llama-vocab for TINY_AYA (#24601); add a dedicated Cohere2MoE (North Code) chat parser (#24615).
  - **MTP/speculative**: add backend sampling support for EAGLE3 (#24655); add spec metrics — mean acceptance length and acceptance rate per position (#24536).
  - **common/chat**: fix LFM2 tool-call parsing double-escaping (#24667); harden peg-native tool-call parsing (#24329); fix an "oldie but goodie" grammar-generator bug (#24653); fix whitespace handling "once and for all" (#24624); include the full unparsed prompt in debug output (#24650).
  - **jinja** (template engine used by chat templates): add `count`/`d`/`e` filter aliases (#24606); fix negative-step slice with start/stop values (#24580); fix `split`/`replace` with an empty first arg (#24574).
  - **mtmd**: fix `n_tokens` miscount (#24656); add a post-decode callback (#24645).
  - **ggml/graph**: fix and restrict NVFP4 edge-cases in llama-graph (#24331).
  - **CUDA**: only support F32/F16 for `GGML_OP_REPEAT` (#24533).
  - **metal**: add `repeat` for bf16 (#24638).
  - **vulkan**: prefer host-visible memory buffers on UMA devices (#22930); support `gated_delta_net` with S_v=16 (#24581); add `col2im_1d` op (#24425); support more `CONCAT` types (#24579); support non-contig unary/glu ops (#24215).
  - **SYCL**: reordered Q4_K/Q5_K/Q6_K MoE `MUL_MAT_ID` (#24452); support `EXPM1` + all `FLOOR`/`TRUNC`/`ROUND` cases (#24363); make `GGML_SYCL_F16=ON` the default (#23996); native subgroup size for K-quant DMMV (#21700); fix `soft_max_f32` max reduction (#24451); fix reorder + add fp32/fp16 to build script (#24578); `set_rows` for q1_0/mxfp4/nvfp4 (#24564); add `pool_1d` (#24584); remove per-allocation Level Zero runtime checks (#23399).
  - **webgpu**: improve i-quant `mul_mat` performance and speed up prefill (#24530).
  - **wasm/convert/cli/bench**: fix wasm fallback symbol collision (#24639); fix lora base-model arch retrieval in convert (#24621); fix CLI not copying preserved tokens (#24258); add `--offline` to bench (#24511).
  - **webui**: mermaid/svg source toggle (#24652); svg block rendering (#24080); render thinking/reasoning blocks as markdown (#24611); HEIC/HEIF image support (#24137); mobile keyboard + PWA popup fixes (#24610); fix mobile UI clipping (#24605); fix `llama-ui-embed` crash with no asset dir (#24597); build-time gzip compression (#24571).
  - **vendor/ci/docker/docs**: update BoringSSL to 0.20260616.0 (#24693); specify registry for Podman builds (#24607); use the CUDA label for the cuda backend in CI (#24594); add SYCL to check-release (#24583); add conda-forge install docs (#22219); fix typos in CUDA-FEDORA.md and grammars/README.md (#24459).

## v0.8.25

### Fixed

- **`LlamaCppEx.Hub` honors HTTP/HTTPS proxy env vars** (#57) — `Hub` uses Req → Finch → Mint, which (unlike curl/wget) does not read proxy environment variables, so on proxy-only networks every HuggingFace request went direct and timed out. Added `proxy_request_options/2`, resolving a proxy from the `:proxy` option (URL string, Mint `{scheme, host, port, opts}` tuple, or `false`) or the standard env vars: `HTTPS_PROXY`/`HTTP_PROXY` (and lowercase) take precedence over `ALL_PROXY`, with `NO_PROXY`/`:no_proxy` host bypass and basic-auth userinfo support (redacted in logs). Wired into `search`, `list_gguf_files`, `get_model_info`, and the streaming download. SOCKS proxies are detected and skipped with an actionable warning (Mint supports HTTP/1 CONNECT proxies only), documented in a new Proxies moduledoc section with a Privoxy/gost bridge workaround. The llama.cpp submodule is unchanged (597b6672e, tag b9621).

## v0.8.24

### Changed

- **llama.cpp submodule** — Updated from 4c6595503 to 597b6672e (20 commits, tag b9621). No NIF changes were required. `include/llama.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/speculative.h`, and `common/sampling.h` are all unchanged. `common/common.h` adds an `mtmd_batch_max_tokens` field to `common_params` (a multimodal batching cap — not used by the binding, which constructs `common_params_speculative`, not `common_params`). The full test suite passes (196 tests), formatting is clean, and Dialyzer reports 0 errors.
  - **mtmd**: add a batching API (#24384).
  - **MTP/speculative**: add EAGLE3 speculative decoding support (#18039).
  - **ggml**: sync ggml and bump to 0.15.1 (ggml/1541); support `concat` for scalar types on the CUDA backend (#24011).
  - **vulkan**: add pipeline barriers for memcpy read operations (#23770).
  - **opencl**: add q5_0/q5_1 GEMM and GEMV kernels for Adreno (#24319).
  - **server**: clean up static asset handling (#24550); fix reasoning-budget WebUI precedence over `model.ini` (#24517).
  - **webui**: keep the original file name and path (#24568); add PWA support (#23871); honor JPEG EXIF orientation (#24196).
  - **fit** (finetuning): wrap `llama_device_memory_data` (#24522); avoid including `llama-ext.h` in `fit.h` (#24506).
  - **vendor/ci/docker**: update cpp-httplib to 0.47.0 (#24395); fix the SYCL CI build & release (#24387) and release-note links (#24527); unbreak the release workflow (#24544, #24545); allow specifying the CUDA GCC version in Docker (#24447).

## v0.8.23

### Added

- **Multi-model manager** (`LlamaCppEx.ModelManager` + `LlamaCppEx.ModelSupervisor`) — keep several models resident at once and route requests to them by id. Builds on the existing `Hub` downloader and batching `Server`; adds named load/unload, capability-based routing, and an advisory memory budget. Opt-in and additive: no existing API changes, no new dependencies, and no auto-started application.
  - **Routing** — a node-wide singleton GenServer owns an ETS table. State changes (load/unload/set_default) serialize through it; `generate`/`stream`/`chat`/`embed` read the ETS table directly from the caller, keeping the manager off the inference hot path. Route by explicit id or `:default`.
  - **Non-blocking loads** — `load/3` runs the Hub download and native model load in a supervised `Task`, so a slow load never blocks other lifecycle calls (concurrent `load`/`unload`/`set_default`) or ETS reads. The caller still blocks until the model is ready; a model in flight reports `status: :loading` and re-loading the same id is refused. The budget reservation and ETS commit stay serialized on the manager.
  - **Backing modes** — `:server` (default for generation/chat) backs the model with a supervised `LlamaCppEx.Server` for batching, streaming, prefix caching, and telemetry; `:direct` (auto-selected when `:embed` is in `:capabilities`) holds the model for stateless calls and is required for embeddings.
  - **Placement-aware memory budget** — knows whether a model lands in RAM or on specific GPUs (from `:n_gpu_layers`/`:split_mode`/`:tensor_split`/`:main_gpu`) and checks each pool independently. `:infinity` (default), an integer (combined RAM+VRAM pool), `:auto` (~80% system RAM + per-GPU free VRAM), or `%{ram: …, vram: [..]|%{i => ..}}`. Refuses over-budget loads naming the device: `{:error, {:insufficient_memory, device: :total | :ram | {:gpu, i}, required:, available:}}`. No automatic eviction.
  - **`LlamaCppEx.devices/0`** — lists ggml backend devices (GPUs, integrated GPUs, accelerators, CPU) with `:gpu_index`, `:memory_total`, and `:memory_free`, via a new backend-agnostic `device_list` NIF (CUDA/Metal/Vulkan). Powers the per-GPU `:auto` budget.
  - **Unload** — stops the backing server (dropping context + model refs) and forces a GC. Reclamation is by GC, so a caller still holding a `%Model{}` from `fetch_model/1` keeps it alive; this is documented.
  - See ADR 009 and the "Multiple Models (ModelManager)" section of the README. Runnable example: `examples/model_manager.exs`. Covered by tests for routing, capability-based dispatch, the memory budget, lifecycle, and async-load concurrency.

## v0.8.22

### Fixed

- **Precompiled NIF 2.18 artifacts** — `mix.exs` advertises precompiled NIFs for NIF versions 2.17 and 2.18, but the precompile workflow built each target on OTP 27 *and* OTP 28, and both of those report NIF 2.17 (2.18 only arrived with OTP 29). The two jobs therefore produced identically named `nif-2.17` tarballs that overwrote each other on upload, and no 2.18 artifact was ever published — so installing on OTP 29 (NIF 2.18) failed with a 404 when fetching the precompiled binary. The precompile matrix now builds on OTP 27 (NIF 2.17) and OTP 29 (NIF 2.18), publishing both NIF versions for each target. The llama.cpp submodule is unchanged (4c6595503, tag b9601).

## v0.8.21

### Changed

- **llama.cpp submodule** — Updated from 04eb4c446 to 4c6595503 (52 commits, tag b9601). No NIF changes were required. `include/llama.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, and `common/speculative.h` are all unchanged. `common/common.h` adds a `path_prompts_log_dir` field to `common_params` (server prompt logging — not used by the binding), and `common/sampling.h` drops the `allow_alt_names` parameter from `common_sampler_types_from_names` (the NIF does not call any `common_sampler_*` functions). The full test suite passes, formatting is clean, and Dialyzer reports 0 errors.
  - **vocab**: adopt leading TemplateProcessing special token as BOS (#24428); refactor normalizer flags into an options struct and add `strip_accents` (#24371).
  - **model/graph/convert**: fix plamo2 `attention_key/value_length` regression (#24317); fix Granite Speech inference by applying embedding scale when deepstack is not used (#24357); guard iswa `kq_mask` on its own buffer (#24294); fix conversion for Mistral-Medium-3.5-128B (#24268).
  - **mtmd**: add video input support (#24269); refactor video subproc handling (#24316); `build_vit` batching (#24352).
  - **MTP/speculative**: Gemma-4 E2B and E4B assistants (#24282); remove padding and multiple D2D copies (#24086); fix "ngram-map-k4v" name in logging (#24253).
  - **common/chat**: fix LFM2/LFM2.5 ignoring `json_schema` (#24377); relax sampler name matching (#23744).
  - **kv-cache**: avoid kv cells copies (#24277); follow the source cache size when sharing cells (#24267); skip checkpoints beyond `pos_next` (#24411); do not clear slots without unified KV cache (#24190).
  - **server**: log prompts to a directory (#22031); skip unused log lines in router mode (#24463); do not parse when flushing http headers (#24281).
  - **CUDA/HIP**: fix `ssm_scan_f32` data races (#24360); reset CUDA context after reading memory size (#23935); remove the GGML_TYPE_Q4_K case in mmvq.cu (#23528); add gfx1152/gfx1153 to RDNA3.5 (#24129).
  - **vulkan**: fast path for contiguous buffer transfers (#23973); medium matmul tile on Asahi Linux (#24306); reduce iq1 shared memory usage for mul_mm (#24287); `v_dot2_f32_f16` support in matmul and Flash Attention (#24123); cm2 `decode_vector` for `mul_mat_id` B-matrix loads (#23991); eMesaHoneykrisp ifdef build fix (#24479).
  - **metal**: fix im2col 1D case for audio models (#24220).
  - **webgpu**: improve prefill speeds for k-quants and refactor matmul for Q4/Q5/Q8 (#24225); handle buffer aliasing for concat (#24000); 2D workgroups for scale/binary/unary ops (#24044).
  - **ggml**: add `GGML_OP_COL2IM_1D` (#24206); fix `rms_norm_back` wrong output under in-place aliasing (#24305); version bumps to 0.14.0/0.15.0.
  - **webui/cli**: pinned conversations (#21387); opt-in `run_javascript` frontend tool (#24244); fix excessive style recalculation on hover (#24243); fix mobile chat form overflow and stale bundle cache (#24158); fix spinner during prompt processing (#24283).
  - **vendor/ci/docker**: update LibreSSL to 4.3.2 (#24397); install ffmpeg in released Docker images (#24302); SYCL compute runtime 26.x in Docker (#24070); fix Windows release CI (#24369); bump komac (#24396).

## v0.8.20

### Changed

- **llama.cpp submodule** — Updated from 6b80c74f2 to 04eb4c446 (7 commits, tag b9549). No NIF changes were required. `common/chat.h`, `common/json-schema-to-grammar.h`, `common/speculative.h`, `common/sampling.h`, and `common/common.h` are all unchanged. The only changed header the binding compiles against is `include/llama.h`, which appends a `ctx_other` field to `llama_context_params` (used by the new Gemma4 MTP path to share `llama_memory`/results between two contexts); the NIF initializes via `llama_context_default_params()` and sets fields by name, so the new field simply defaults to `nullptr` and the binding is unaffected. The full test suite passes, formatting is clean, and Dialyzer reports 0 errors.
  - **model/mtmd**: add Gemma4 MTP — multi-token prediction / speculative decoding for dense Gemma4, adding the `ctx_other` context-sharing mechanism (#23398); fix Gemma4 conversion when there is no audio encoder (#24242); support "frame merge" for qwen-vl-based models (#21858).
  - **common/chat**: fix LFM2/LFM2.5 reasoning round-trip and `<think>` leak (#24234).
  - **spec**: fix the vocab compatibility check (#24256).
  - **common/arg**: skip the mmproj download when the user supplied an mmproj (#24239).
  - **docker/ci**: bump cuda13 to 13.3.0 (#24228).

## v0.8.19

### Changed

- **llama.cpp submodule** — Updated from 166fe2949 to 6b80c74f2 (47 commits, tag b9542). No NIF changes were required. Every header the binding compiles against — `include/llama.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/speculative.h`, `common/sampling.h`, and `common/common.h` — is byte-for-byte unchanged across the range. The full test suite passes (147 tests), formatting is clean, and Dialyzer reports 0 errors.
  - **model/mtmd**: Granite4 Vision (#23545); fix Gemma 4 unified FPE (#24088) and audio projector embedding size (#24091); fix Gemma 4 Unified conversion (#24118); add a "placeholder bitmap" for counting tokens plus a `*/input_tokens` API (#23913); refactor `hparams.n_layer` (#24060); fix `llama_model::n_gpu_layers()` (#24188) and off-by-one comparisons to `n_gpu_layers` (#24208).
  - **common/chat**: unify and fix the LFM2/LFM2.5 tool parser (#24178).
  - **server**: disable on-device speculative checkpoints (#24108); avoid unnecessary checkpoint restore when new tokens are present (#24110); restore the memory-saving filter (#24125).
  - **CUDA / TP**: enroll `mul_mat_vec_q_moe` into PDL (#24087); round tensor-parallel granularity up to 128 (#24180).
  - **vulkan**: check coopmat2 features before reporting support (#24186); add FWHT support for Intel with shared-memory reduction (#23964).
  - **SYCL**: port multi-column MMVQ from the CUDA backend (#21845).
  - **opencl**: improve `get_rows`, `cpy`, `concat`, and q6_K flat gemv (#24160).
  - **ggml**: WASM SIMD128 vectorization of `ggml_vec_dot_q4_1_q8_1` (#22209); extend RVV quantization vec dot to higher VLENs (#22754); WebGPU FlashAttention refactor and standardized quantization support (#23834); KleidiAI dynamic chunk-based scheduling for hybrid execution (#23819).
  - **metal**: reduce rset heartbeat from 500ms → 5ms (#24074).
  - **common/arg**: fix double MTP downloads (#24128).
  - **build/ci**: use the umbrella Headers directory for the XCFramework module map (#23974); skip cvector-generator and export-lora when the CPU backend is disabled (#24053); consolidate duplicated imatrix code into `common/imatrix-loader.cpp` (#22445).

## v0.8.18

### Changed

- **llama.cpp submodule** — Updated from 0b7154066 to 166fe2949 (16 commits, tag b9495). No NIF changes were required. `include/llama.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, `common/sampling.h`, and `common/common.h` are all unchanged. The only changed header the binding compiles against is `common/speculative.h`, which renames `common_speculative_need_embd_pre_norm` → `common_speculative_need_embd_nextn`; the NIF only calls `common_speculative_need_embd` (not the renamed variant), so the MTP/speculative paths are unaffected. The full test suite passes.
  - **model/convert**: add Mellum architecture (#23966); support Granite multilingual embeddings R2 (ibm-granite/granite-embedding-{97,311}m-multilingual-r2) (#22716); add StepFun 3.5 MTP (#23274); qwen35 — use the post-norm hidden state for MTP (#24025).
  - **mtmd**: enable non-causal vision for Gemma 4 unified (#24082); allow skipping `build_vit()` (#24077).
  - **CUDA**: reserve space for the quantized KV cache at startup (#23907); avoid PDL race conditions by disabling `__restrict__` when PDL is used (#24030).
  - **opencl**: use flat variants of q4_K and q6_K gemv for very large M (#24006).
  - **hexagon**: profiler output fix and script updates (#24042).
  - **ggml-cpu**: use the runtime SVE width in FWHT (#24059).
  - **common/arg**: skip the unnecessary mmproj download when `--no-mmproj` is passed (#23425).
  - **webui**: Mermaid diagrams in chat with interactive preview (#24032).
  - **tests**: add support for qwen3 SSM archs (#24031).
  - **build/vendor/CI**: update BoringSSL to 0.20260526.0 (#23794); disable ccache for MSVC Windows release jobs (#23911).

## v0.8.17

### Changed

- **llama.cpp submodule** — Updated from d4c8e2c29 to 0b7154066 (37 commits, tag b9479). No NIF changes were required. `common/chat.h` and `common/json-schema-to-grammar.h` are unchanged; the changes to `include/llama.h`, `common/sampling.h`, and `common/common.h` are additive or touch symbols the binding does not use. `include/llama.h` adds an `n_outputs_max` field to `llama_context_params` (the NIF initializes via `llama_context_default_params()` and sets fields by name, so it defaults to `0` = `n_batch`) and marks `llama_set_warmup` as `DEPRECATED` (not called by the NIF). `common/sampling.h` adds `common_sampler_reasoning_budget_force` and `common/common.h` adds `reasoning_control`/`n_outputs_max`/`sse_ping_interval` fields plus a signature change to `common_prompt_batch_decode` — none of which the binding uses. The full test suite passes.
  - **model/vocab/convert**: add EXAONE 4.5 implementations (#21733); support Step3.7-Flash conversion (#23845); add `normalizer.lowercase` support to WPM tokenizer (#23899).
  - **llama**: deprecate `llama_set_warmup` (#24009); limit max outputs of `llama_context` via `n_outputs_max` (#23861); SWA checkpoints store only non-masked cells (#23981); tensor-parallel quantized KV cache support (#23792); speculative — fix `n_outputs_max` and remove draft-simple auto-enable (#23988).
  - **common**: fix state save in `common_prompt_batch_decode` (#23468); support manually triggering the reasoning budget end sequence (#23949).
  - **server**: add SSE ping interval (#24013); real-time reasoning interruption via control endpoint (#23971); handle `If-None-Match` weak ETags (#23916); disable private security disclosures (#23963).
  - **vulkan**: don't hold the device mutex while compiling pipelines (#23641); reduce host memory lock contention (#23376); block-load Q3_K/Q6_K block data and subtract on 32-bit ints (#23056); remove unused functions (#23175).
  - **SYCL**: support Q4_1/Q5_0/Q5_1 in Flash-Attention (#23812); add more types in `GET_ROWS` op (#23710); optimize Q3_K `mul_mat` by reorder (#23725).
  - **metal**: template GLU kernels to support f16/f32 (#23882).
  - **hexagon**: MUL_MAT, MUL_MAT_ID, FLASH_ATTN and GDN cleanup and optimizations for latest models (#23989); add `gelu_quick` (#24007).
  - **opencl**: add basic support for q5_0 and q5_1 (#23548); fix compiler warnings for the non-Adreno path (#23922); revert to using `global_invocation_id` for the cpy shader (#23955).
  - **webui**: add a Thinking-mode toggle with reasoning effort levels and Chat Form "Add Action" UI improvements (#23434); simplify network error handling (#23431).
  - **build/vendor/CI/docs**: update cpp-httplib to 0.46.1 (#23980); add nix-nodejs facilities to build the Web UI (#23846); clean up unused-variable warnings (#23975); CI job trimming and runner-label fixes (#24012, #23958, #23927, #23938); update `HOWTO-add-model.md` (#23883).

## v0.8.16

### Changed

- **llama.cpp submodule** — Updated from 19e92c33e to d4c8e2c29 (40 commits, tag b9442). No public API changes; `include/llama.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, and `common/sampling.h` are all unchanged. `common/common.h` has only additive/benign changes (a new `skip_download` field, the server `timeout_read` default bumped 600s→3600s, and a comment tweak) that don't touch any NIF binding. No NIF changes were required; the full test suite passes.
  - **model/vocab**: support for `DeepseekV32ForCausalLM` with a generic DeepSeek Sparse Attention (DSA) implementation (#23346); tokenizer support for jina-embeddings-v2-base-zh (#18756) and LFM2.5-8B-A1B (#23826).
  - **mtmd**: add DeepSeekOCR 2 support (#20975).
  - **llama**: use f16 mask for Flash Attention to save VRAM (#23764); only use one iGPU device by default (#23897) and don't skip iGPU when only RPC devices are present (#23868); add `llm_graph_input_mtp` (#23643).
  - **server**: in SSE mode, send HTTP headers when the slot starts (#23884); bump read timeout to 3600s (#23842); add speed-bench for speculative decoding (#23869); remove obsolete scripts (#23870).
  - **metal**: restore im2col implementation for large kernels (#23901).
  - **vulkan**: add Flash Attention support for BFloat16 KV cache (#23420).
  - **CUDA**: check PTX version on the host side to guard PDL dispatch (#23530); disable `launch_fattn` PDL enrollment due to a compiler bug (#23825).
  - **opencl**: support bf16 by converting to f16 (#23839).
  - **ggml**: add some LSX support (#23798); bump ggml version to 0.13.1; sync ggml; fix missing `buffer` set in the allreduce fallback (#23480).
  - **ggml-webgpu**: check earlier for required WebGPU features (#23879); add q4_0/q8_0 `SET_ROWS` (#23760).
  - **graph**: ensure DS32 `kq_mask_lid` is F32 (#23864).
  - **tensor-parallel**: fix granularity for Qwen 3.5/3.6 with 3 GPUs (#23843).
  - **download**: add option to skip downloading (#23059); **llama-bench**: support `-fa auto` (#23714).
  - **app/ui**: add `llama update` self-updater (#23865); move licences to llama-app (#23824); custom CSS injection via config (#23904); handle `audio/vnd.wave` as WAV (#23754); fix ETag truncation with MSVC (#23917); exclude generated build dirs from prettier/eslint (#23910).
  - **CI/docs/build**: macOS and iOS release jobs moved to macos-26 runners (#23878, #23906); fix s390x release job (#23898); cache-key fixes (#23895); update ZenDNN docs for Q8 support (#23791); `ngram-mod` missing include (#23857).

### Fixed

- **`LlamaCppEx.Server` double-accept.** The server's per-slot sampling called
  `sampler_accept` after `sampler_sample_at`, but `sampler_sample_at` (like
  `llama_sampler_sample`) already accepts the selected token. The redundant
  accept double-advanced sampler state — the same class of bug fixed in the
  direct generation loops in v0.8.15. It broke grammar-constrained server
  inference and double-counted repeat/frequency/presence penalties. Removed both
  redundant accepts (first-token-after-prefill and each decode step). Output for
  penalty-based sampling through the server changes (now correct); greedy/temp
  sampling is unaffected.

### Performance

- **Server batch loop — removed super-linear hot paths.** The three batching
  strategies (`DecodeMaximal`, `PrefillPriority`, `Balanced`) shared identical
  per-slot/per-token assembly code with several costly patterns: `batch_idx =
  length(entries)` inside the prefill token loop (**O(n²)** per prompt),
  `length(slot.prompt_tokens)` (O(n)) called twice per chunk, `Enum.slice/3` on a
  list (O(prefill_pos)) per chunk, and `accumulated_text <> piece` per decode
  token (**O(n²)** over a generation). The shared logic now lives in
  `LlamaCppEx.Server.Strategy.Batch` and: threads a running entry counter instead
  of `length/1`, uses the cached `slot.n_prompt_tokens`, slices from a
  `prompt_tokens_tuple` (O(1) indexing), and accumulates token pieces as iodata
  joined once at completion. Behavior is unchanged (guarded by
  `test/server_batch_test.exs`).
- **`common_prefix_length/2`** rewritten as a single-pass tail recursion (was
  `Enum.zip |> Enum.take_while |> length`, allocating an intermediate list). Runs
  on every prefix-cache lookup.
- **`embed_batch/3` no longer allocates a context per text.** It now packs texts
  into a single context as distinct sequences (greedy bin-packing within the
  context budget, capped by `:max_batch_sequences`, default 64) and decodes each
  group in one batch via the new `embed_batch_decode` NIF, retrieving pooled
  per-sequence embeddings. Falls back to one-context-per-text only for
  `:pooling_type: :none`. Equivalence with the per-text path is guarded by a smoke
  test.
- **Streaming NIF loops** (`generate_tokens`, MTP `generate_mtp_tokens`) intern
  the hot result atoms once instead of per token and reuse the detokenize fallback
  buffer across iterations.

## v0.8.15

### Changed

- **llama.cpp submodule** — Updated from 0d18aaa9d to 19e92c33e (51 commits). No public API changes; `include/llama.h`, `common/common.h`, `common/chat.h`, `common/json-schema-to-grammar.h`, and `common/sampling.h` are all unchanged. The only header diffs are internal (`src/llama-chat.h`, `src/llama-vocab.h`, `ggml-cpu/vec.h`, the ggml-hexagon op headers, `clip-graph.h`, `server-http.h`, and the vendored `cpp-httplib`). No NIF binding changes were required.
  - **chat**: add Granite 4.1 chat template (#23518) — picked up automatically by `chat_apply_template`.
  - **model/mtmd**: fix gemma 4 projector pre_norm (#23822) and audio rms norm eps (#23815); `n_head_kv` defaults to `n_head` (#23782); mtmd-debug color and rainbow mode (#23829).
  - **convert**: add FP8 to Q8 conversion (#23250); add MiniCPM5 tokenizer support (#23384).
  - **arg/common**: add `LLAMA_ARG_API_KEY_FILE` env var for `--api-key-file` (#23167); fix env names to all have the `LLAMA_ARG_` prefix (#23778).
  - **server**: add support for HTTP ETags in llama-server (#23701); minor tweaks to use more cpp features (#23785); fix the log message when using SSL (#23393).
  - **ggml**: auto-apply iGPU flag for CUDA/HIP on integrated devices (#23007); fix Arm SVE usage bug in `vec.h`/`vec.cpp` (#22841).
  - **CUDA**: route batch>=4 quantized matmul to MMQ on AMD MFMA hardware (#23227); add `MMVQ_PARAMETERS_TURING` (#23729); fix KQ mask offset integer overflow in the fattn MMA kernel (#23610); restrict PDL to CTK >= 12.3 due to MSVC issues (#23742).
  - **vulkan**: fast path for Walsh–Hadamard transform (#23687); use `GL_NV_cooperative_matrix_decode_vector` for faster matmul (#23541); switch `MUL_MAT_VEC` to 4 K per iteration for F16/32 (#22887); add `REPEAT` op support for f16→f16 (#23298); avoid preferring transfer queue on AMD UMA devices (#22455); fix inner-loop index variable (#23665) and memory-logger unsafe iterator access (#23667).
  - **hexagon**: basic/generic op fusion + `RMS_NORM`+`MUL` fusion (#23835); `OP_GATED_DELTA_NET` K>1 support (#23531); add Q4_1 in `MUL_MAT`/`MUL_MAT_ID` (#23647); minor refresh for HMX FA and MM (#23796).
  - **opencl**: `OP_GATED_DELTA_NET` (#23312); move backend info printing into its own function (#23702).
  - **ggml-webgpu**: remove legacy constants (#23672); fix workgroup dispatch for some ops (#23750).
  - **ggml-zendnn**: fix naming of matmul function (#20964).
  - **ui**: fix audio and video modality detection (#23756).
  - **app**: improve help output (#23805).
  - **perplexity**: fix format specifier in `LOG_ERR` (#23788).
  - **vendor**: update `cpp-httplib` to 0.46.0 (#23650).
  - **docker**: add ZenDNN Dockerfile (#23716).
  - **pyproject**: add conversion folder and update dependencies (#23746).
  - **docs**: fix duplicated "the" in granitevision and model-conversion docs (#23767).
  - **CI**: numerous build/runner changes — UI publish on ubuntu-slim (#23818), releases use GitHub-hosted builds for the UI (#23823), Vulkan builds switched to Release (#23820), CI refactor (#23789), move ARM jobs to self-hosted (#23780), bump CUDA release to 13.3 (#23749), add ccache to server builds (#23763), fix windows ccaches (#23777), remove wasm test (#23733).

### Fixed

- **Grammar / structured output crash (double-accept)** — Constrained generation
  (`:json_schema` and `:grammar` options on `generate`/`stream`/`chat`/`stream_chat`)
  crashed on the **first** generated token with
  `RuntimeError: Unexpected empty grammar stack after accepting piece: ...`.
  The generation loops in the NIF called `llama_sampler_accept/2` after
  `llama_sampler_sample/3`, but `llama_sampler_sample/3` already accepts the
  selected token internally. The redundant accept advanced grammar state twice,
  so the grammar tried to match the just-consumed token against the *next*
  position and emptied its stack. For unconstrained sampling the double-accept
  was mostly harmless (it double-counted repeat/frequency/presence penalties),
  which is why it went unnoticed. Removed the redundant `llama_sampler_accept/2`
  from all five sampling sites: `generate`, `generate_tokens` (streaming),
  `decode_batch`, and both MTP/speculative loops. Structured output now returns
  schema-valid JSON, and penalty-based sampling is no longer double-applied.

### Testing

- **Added an end-to-end smoke test** (`test/smoke_test.exs`, tagged `:smoke` and
  **excluded by default**) covering generation, streaming, chat templating,
  structured output (JSON-schema + raw GBNF grammar — a regression guard for the
  double-accept bug), and embeddings against real GGUF models. Run with
  `LLAMA_SMOKE_GEN_MODEL=... [LLAMA_SMOKE_EMB_MODEL=...] mix test --include smoke`.

## v0.8.14

### Changed

- **llama.cpp submodule** — Updated from b22ff4b7b to 0d18aaa9d (52 commits). No public API changes; the only `include/llama.h` diff is a doc comment on `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` clarifying that getting a per-seq state with the flag invalidates prior on-device states for the same `seq_id`. `common/common.h` replaces `checkpoint_every_nt` with `checkpoint_min_step` (server-only field) and drops the `LLAMA_UI_DEFAULT_ENABLED` ifdef (UI still defaults to `true`); `common/chat.h` adds additive `common_chat_msg_span`/`common_chat_msg_delimiter` structs, a `message_spans` field on `common_chat_params`, an `is_continuation` field on `common_chat_parser_params`, and a new `common_chat_split_by_role` helper. None of these are used by the NIF, so no binding changes were required.
  - **llama**: document that only one on-device state can be saved per sequence (#23520).
  - **server**: fix checkpoints creation (#22929); MTP layer kv-cache should respect draft type ctk (#23646); expose prompt token counts in `/slots` (carried from v0.8.13 lineage); add margin for draft model for `fit` (#23485).
  - **convert**: support `Gemma4ForCausalLM` (#23682); add compressed-tensors NVFP4 support (#21095); minor fixes for numpy 2.x (#23571).
  - **model**: add support for `talkie-1930-13b` (#22596); tag `ffn_latent` as `MUL_MAT` to fix buft probe (#23664); attach Mistral3 NVFP4 weight scales (#23629).
  - **vocab**: fix `HybridDNA` tokenizer (#23466) (carried).
  - **ggml**: bump to 0.13.0 (ggml/1510) and 0.12.1 (ggml/1508); `gguf_init_from_callback` and `gguf_init_from_buffer` (#22341); parallelize quant LUT init (#23595); ggml-alloc out-of-bounds read fix in `ggml_dyn_tallocr_remove_block` (ggml/1492); TP fix ggml context size calculation (#22616); `ggml_silu_back` docstring fix (ggml/1500).
  - **metal**: add apple device id (#23566).
  - **CUDA**: add fast Walsh–Hadamard transform (#23615); missing PDL sync for FWHT + better fallback (#23690).
  - **vulkan**: optimize `conv2d` and implement `coopmat1` support (#22620).
  - **SYCL**: implement `ggml_sycl_pool_vmm` (#22862).
  - **hexagon**: add `CONCAT` op (#23648); flash-attn softmax repl optimization (#23455).
  - **ggml-webgpu**: add MMVQ path for Q4/Q8/Q2_K/Q4_K and clean up legacy `MUL_MAT` pipeline (#23594); check `batch_compute_passes` before sending passes when not GPU profiling (#23457).
  - **opencl**: batch profiling to improve speed and prevent memory leaks (#23495).
  - **perplexity**: fix even more integer overflows (#23623).
  - **TP**: fix entirely zero-sized slices per device (#23525).
  - **ui**: fix stop/continue during an agentic loop (#23356); media attachments before text (#23467).
  - **vendor**: update `cpp-httplib` to 0.45.1 (#23639).
  - **snapdragon**: bump toolchain docker to v0.7 to fix UI build issues (#23680); update windows toolchain to use `hsdk` v6.6.0.0 (#23552).
  - **cmake**: fix UI build (#23592).
  - **tests**: `test-backend-ops -j <N>` to run tests in parallel (#23637).
  - **CI**: many self-hosted runner migrations, `[no release]` keyword support, and macOS/apple workflow consolidation (#23705, #23713, #23715, #23718, #23721, #23728, #23730, #23734, #23619, #23616, #23675, #23642, #23651, #23630).

## v0.8.13

### Changed

- **llama.cpp submodule** — Updated from 52fb93a2b to b22ff4b7b (25 commits). No public API changes; `include/llama.h` is unchanged. `common/chat.h` adds an additive `is_continuation` field (default `false`) on `common_chat_parser_params`; `common/common.h` simplifies the `ui` default (removes the `LLAMA_UI_DEFAULT_ENABLED` ifdef, still defaults to `true`). No NIF changes required.
  - **model**: add NVFP4 MTP scale tensors (#23563).
  - **server**: only parse empty message if continuing an assistant message (#23506); expose prompt token counts in `/slots` endpoint (#23454).
  - **vocab**: fix HybridDNA tokenizer (#23466).
  - **perplexity**: fix integer overflow (#23496).
  - **ggml**: check the right iface method before using the fallback 2D get (#23514).
  - **flash-attn**: replace `f32` with `kv_type` and `q_type` (#23372).
  - **metal**: optimize concat kernel and fix `set` kernel threads (#23411).
  - **CUDA**: fix PDL CC check for JIT compilation (#23471).
  - **vulkan**: fuse snake activation `mul + sin + sqr + mul + add` (#22855); fix windows `find_package` of `SPIRV-Headers` (#23215).
  - **SYCL**: improve MoE prefill throughput (#23142); Level Zero detection in `ggml_sycl_init` (#23097); `gated_delta_net` K>1 (#23174); add BF16 to DMMV kernel path (~4x tg speedup on Intel Arc) (#21580).
  - **opencl**: generalize Adreno MoE kernels on M (#23449).
  - **ggml-zendnn**: add Q8_0 quantization support (#23414).
  - **cmake**: refactor UI build (#23352); add `install()` for impl libraries + fix Apple builds (#23511); remove `STATIC` from impl libraries, enable `LLAMA_BUILD_APP` by default (#23462); build router app only during standalone builds (#23521).
  - **tests**: move `save-load-state` from examples to tests (#23336).
  - **docs**: update documentation with Granite 4.0/4.1 (#23404); update WebGPU support and add link to blog/demo (#23483).
  - **requirements**: bump torch to 2.11.0 (#23503).

### Fixed

- **`mix.exs` `@version` drift** — `@version` is bumped from `0.8.11` to `0.8.13` to re-align with the published Hex/tag stream. Tag `v0.8.12` was cut against a `@version "0.8.11"` source tree, so this release skips `0.8.12` to avoid republishing under a stale version.

## v0.8.12

### Changed

- **llama.cpp submodule** — Updated from b28a2f372 to 52fb93a2b (30 commits). No public API changes; existing NIF and `LlamaCppEx.MTP` bindings continue to work unchanged.
  - **MTP / speculative**: move draft sampling to the backend (`backend_sampling` defaults `true` on the new `common_params_speculative_draft` field — additive) (#23287); skip logit computation via `inp_out_ids` (#23433); fix `nullptr` crash in `common_speculative_get_devices_str` (#23386); free draft/MTP resources on server slot sleep to fix a VRAM leak (#23461); doc typo (#23435).
  - **llama**: fix `llm_graph_input_attn_kv_iswa` null-buffer crash on SWA-only models (#23131).
  - **vocab**: add Carbon-3B `HybridDNATokenizer` support (#23410).
  - **server**: re-inject subcommand when the router spawns children under the unified binary (#23442).
  - **app**: introduce the `llama` unified executable (#23296); add `batched-bench`, `fit-params`, `quantize`, and `perplexity` subcommands (#23459); show version (#23426).
  - **mtmd**: merge HunyuanOCR into HunyuanVL and fix OCR vision precision (#23329); DeepSeek-OCR image-processing fixes + `img_tool::resize` padding refactor (#23345); `fit_params` now accounts for `mmproj` (#21489); WAV MIME-type variants and improved audio format detection (#23396).
  - **ggml**: check the right iface method before falling back to the 2D get (#23306).
  - **metal**: optimize `pad` + `cpy` (#23354).
  - **CUDA**: Programmatic Dependent Launch (PDL) for Hopper+ (#22522); tune RDNA3 Q6_K MMVQ nwarps (#23349).
  - **vulkan**: optimize `IM2COL` shader (#22685).
  - **opencl**: refactor backend initialization (#23318).
  - **hexagon**: `ssm-conv` fix for large prompts (#23307); HMX quantized matmul rework (#23368).
  - **snapdragon**: update toolchain to v0.6 (#23369).
  - **webui**: max image size option (#22849); reactive `isMobile` in viewport store (#23330); div-wrapper pointer-events fix on hidden (#23390); move text attachments before message content in chat-completions payload (#23406); improve UI dev git hooks (#23403).
  - **docker**: copy conversion files (#23370).

## v0.8.11

### Changed

- **llama.cpp submodule** — Updated from 0253fb21f to b28a2f372 (57 commits).
  - **llama**: MTP clean-up (#23269); initialize pre-norm embedding mask flag (#23256); avoid copying logits during prompt decode in MTP (#23198).
  - **common**: delegate assistant continuation to underlying template handlers (#23089) — new `common_chat_continuation` enum and `continue_final_message` field on `common_chat_templates_inputs` (default `COMMON_CHAT_CONTINUATION_NONE`, additive); enable streaming JSON argument values (#23173); remove hf cache migration (#23266); fix `--help` and `--fit` `--verbosity` output (#23278, #23282).
  - **server**: guarantee at least 1 token to decode in server-context (#23280); print graphs reused in slot timings (#23279); honor `--embd-normalize` CLI arg (#23125); router allocates tmp buffer on heap (#23159); skip device enumeration in router mode to avoid creating CUDA primary context (#23137).
  - **model**: clarify MTP layer comment in qwen35.cpp (#23338); update bid to match each layer's MTP source (#23237).
  - **vulkan**: add cpy bf16 → f32 pipelines (#22677); support unaligned tensors for ROPE (#22637); fuse `SSM_CONV + BIAS + SILU` (#22653); add `SPIRV-Headers` cmake check (#22009); remove duplicate `#include <memory>` (#23144).
  - **hexagon**: add MROPE and IMROPE in HTP rope op (#23317); enable NORM op (#23319); add TRI op (#22822); ggml-hexagon PAD op HVX kernel (#23078).
  - **opencl**: add MoE support for q4_k, q5_k, q6_k on Adreno (#23303).
  - **CUDA**: continue directly including `cuda/iterator` (#23102); support `d_conv=15` for `ssm-conv.cu` (#23017).
  - **SYCL**: add `GGML_SYCL_USE_ASYNC_MEM_OP` env toggle (#22153); scalar SWAR byte-subtract in Q6_K MMVQ dot product (#22156); route small f32 matmuls to oneMKL, bypass oneDNN (#22150); fix error when using `-mg 1` (#23140); performance reference in SYCL.md (#23315).
  - **ggml-webgpu**: extend GDN for K>1 (#23299).
  - **rpc**: keep `last_graph_uid` in the device context (#23273).
  - **webui**: chat screen UI refactor (#23333); bump packages + address build warnings (#23300); update KaTeX + clean `sass` warnings (#23275); scroll-to-bottom button + prevent forced scroll (#23270); refactor models store / MCP service / gate logs behind `VITE_DEBUG` (#23236); centralize monospace font styles (#23272); fix Tailwind v4 utility classes missing when built via cmake (#23253); support video files as input (#22830).
  - **convert**: update MTP-related help (#23334); filter LoRA tensor names (#23077).
  - **save-load-state**: refactor tests and improve readability (#23196).
  - **llama-eval**: add per-task summary stats (#23151).
  - **ngram**: reduce noisy logs (#23185).
  - **build/CI**: install libssl-dev (#23325); install server kleidiai runner dependencies (#23259); add kleidiai-server to server-self-hosted workflow (#22435); cmake — do not check for bin install dir (#23234), fix `LLAMA_BUILD_UI` logic (#23190), do not install conversion script (#23204); docker — add OCI image labels for version and build date (#21653).

## v0.8.8

### Fixed

- **Server prefix-cache crash on hybrid GDN models** (#38) — On hybrid Gated Delta Net architectures (Qwen 3.5 / 3.6) `llama_memory_seq_rm` silently no-ops on partial-range trims, leaving the KV cache at the old positions while the next prefill tried to write tokens at lower positions. This produced an M-RoPE positional-consistency abort (`X = 56 >= Y = 46 ... requires X < Y`). `LlamaCppEx.Server` now probes `common_context_can_seq_rm` once at init and falls back to a full reset (n_match=0) when the model only supports `:full` seq_rm. Includes a regression test for sequential same-prefix requests under `cache_prompt: true`.

### Added

- **`LlamaCppEx.NIF.context_can_seq_rm`/1** — exposes `common_context_can_seq_rm`, returning `:no | :part | :full | :rs`. Clears KV memory as a side effect, so call once before any decode.

## v0.8.7

### Added

- **Multi-Token Prediction (MTP) speculative decoding** (#37) — new `LlamaCppEx.MTP` module exposing `init/2`, `stream/3`, `stream_events/3`, `generate/3`, `stats/1`, and `print_stats/1`. Drives a target/draft speculative loop where the draft model is the MTP head embedded in the same GGUF (e.g. [`ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF), or the `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` UD-Q4_K_XL quant). On hybrid models (GDN + attention, e.g. Qwen 3.6) the loop wraps each iteration in a recurrent-state checkpoint save/restore so partial draft rejections are recoverable. See README "Speculative decoding (MTP)" and `examples/mtp_speculative.exs` / `examples/mtp_benchmark.exs`.

  **Performance status (Apple Silicon):** the lack of speedup on Metal is intrinsic to the hardware, not the binding. Direct comparison on M1 Max with upstream's own `llama-server --spec-type draft-mtp`: 39.80 tok/s MTP vs 39.14 tok/s plain (1.02×) on Qwen 3.6 35B-A3B. Pair this with `n_draft: 1` and our binding reaches 39.7 tok/s at 79% acceptance for a ~1.06× speedup — see upstream [#23011](https://github.com/ggml-org/llama.cpp/issues/23011) and the Metal MTP follow-up [#23114](https://github.com/ggml-org/llama.cpp/pull/23114). On NVIDIA, the upstream-quoted 2× should hold with `n_draft: 3`.
- **Live MTP statistics** — `MTP.stats/1` returns a lock-free snapshot of speculative counters (`iters`, `drafts_generated`, `drafts_accepted`, `acceptance_rate`, `tokens_emitted`, `tokens_per_sec`, per-stage `timing_us`). Safe to call mid-stream from any process; optional `:emit_stats_every` flag streams periodic snapshots over the token channel.
- **Context options for speculative decoding** — `LlamaCppEx.Context.create/2` accepts `:ctx_type` (`:default` / `:mtp`) and `:n_rs_seq` (rollback snapshot count), plus new `Context.n_rs_seq/1` getter.

### Changed

- **llama.cpp submodule** — Updated from 834a24366 to 0253fb21f (31 commits), pulling in MTP and related speculative-decoding work.
  - **llama + spec**: MTP Support (#22673) — multi-token prediction speculative decoding, new `llama_context_type` enum (`LLAMA_CONTEXT_TYPE_DEFAULT` / `LLAMA_CONTEXT_TYPE_MTP`), new `llama_context_params.ctx_type` and `n_rs_seq` fields, new `llama_n_rs_seq()` API, new `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`.
  - **spec**: allow partial seq_rm for GDN models for speculative decoding (#22400).

## v0.8.6

### Changed

- **llama.cpp submodule** — Updated from 1e5ad35d5 to 834a24366 (63 commits).
  - **model**: fix model type check for granite/llama3 and deepseek2/glm4.7 lite (#22870).
  - **spec**: parallel drafting support (#22838); update CLI arguments for better consistency (#22964).
  - **server**: accept `continue_final_message` flag for vLLM API compat (#23012); support continue generation on reasoning models (#22727); expose modalities to `/v1/models` (#22952); print warning when HTTP timeout exceeded (#22907).
  - **mtmd**: add MiMo v2.5 vision (#22883).
  - **CUDA**: handle `OW > 65535` in `im2col` (2D and 3D) (#22944); snake fusion hardening (#22912); directly include `cuda/iterator` (#22936); internal AllReduce kernel for CUDA provider (#22299).
  - **SYCL**: fix multi-GPU system RAM exhaustion by using Level Zero allocations (#21597); add OP `im2col_3d` (#22903).
  - **vulkan**: fix matmul integer pipeline selection (#23005); fix Windows performance regression on Intel GPU BF16 for Xe2+ (#22461); check shared memory size for MMQ shaders (#22693); support asymmetric FA in scalar/MMQ/coopmat1 paths (#22589).
  - **hexagon**: add unary tanh op (#22999); eliminate scalar VTCM loads via HVX splat helpers (#22993).
  - **opencl**: add q5_0/q5_1 MoE for Adreno (#22985); fix crash when warming up MoE on Adreno (#22876); add opt-in Adreno xmem F16xF32 GEMM for prefill (#22755); add q4_1 MoE for Adreno (#22856).
  - **ggml-webgpu**: enable NVIDIA self-hosted CI (#22976); subgroup-aware flash attn vec path (#23040); restrict subgroup-matrix path to compatible head dims (#23020); enable running gpt-oss-20b (#22906); precision fixes for multimodal (#22808); cast intermediate results to float to avoid half+half ambiguity (#22994); flush GPU profile timestamp before queryset overflow (#22995).
  - **ggml-cpu**: add IME2 instruction support for the SpacemiT backend (#22863).
  - **ggml-zendnn**: adaptive fallback to CPU backend for small batch sizes (#22681).
  - **ggml-virtgpu**: add a GHA build check (#22943); include missing mutex header (#22810).
  - **ggml**: bump version to 0.11.1; sync ggml.
  - **metal**: promote `mul_mv`/`mul_mm` batch divisors to function constants (#22711).
  - **backend sampling**: support returning post-sampling probs (#22622).
  - **unicode**: add Qwen3.5 non-backtracking tokenizer handler and regression test (#22110).
  - **logs**: reduce verbosity (#23021).
  - **download**: do not `exit()` on error (#23008).
  - **convert**: fix Pixtral 12B `--mistral-format` conversion (3 bugs) (#22981); add `split()` to `LoraTorchTensor` in LoRA converter (#22832); add image break token fallback (#22914).
  - **webui**: move static build output from repo code to HF Bucket (#22937); deduplicate model aliases (#22979); preserve system message on edit cancel (#22911); fix chat screen form box disappearing + autoscroll issues on WebKit (#22977); autoscroll detection (#23026); propagate version tag to WebUI asset download in self-hosted CI (#23051).
  - **examples**: add `llama-eval` (#21152); enable type check in `llama-eval` (#22988); update speculative-simple README (#22938).
  - **model-conversion**: add `causal-convert-mmproj` target (#22969).
  - **vendor/deps**: update cpp-httplib to 0.44.0 (#22919, #22888).
  - **build/CI**: revert docker intel compute-runtime to stable (#22968); validate model naming convention (#22680); bump `ty` to 0.0.35 (#22961).
  - **docs**: update OPENVINO.md (#22959); fix metrics endpoint description in server README (#22879).

## v0.8.5

### Changed

- **llama.cpp submodule** — Updated from eff06702b to 1e5ad35d5 (68 commits).
  - **model**: add sarvam_moe architecture (#20275); support Gemma4_26B_A4B_NVFP4 (#22804); add Mimo v2.5 (#22493); support sarashina2.2-vision-3b (#22103); don't crash on unsupported architecture (#22742).
  - **llama**: add option to save memory in device buffers, with new `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` flag (#22679); fix device state save/load (#22805); remove unnecessary seq_id check during state restore (#22797); add missing `ggml_backend_load_all()` call (#22752).
  - **common**: do not wrap raw strings in schema parser for tagged parsers (#22827); revert reasoning budget +inf logit bias (#22740); preserve media markers for typed-content templates (#22634); do not fit to unknown device memory (#22614); only load backends when required (#22290); fix missing-noreturn warnings on clang 21 (#22702).
  - **server**: support Vertex AI compatible API (#22545); router exposes child model info from `/v1/models` (#22683); validate `--tools` CLI argument against known tool names (#22538).
  - **mtmd**: support MiniCPM-V 4.6 (#22529); add granite-speech support (#22101); fix whisper audio tail truncation by exposing padded buffer to FFT (#22770).
  - **CUDA**: fuse snake activation (#22667); batch `out_prod` inner loop with `cublasSgemmStridedBatched` (#22651); lower-case PCI bus id, standardize for ggml (#22820).
  - **SYCL**: reduce allocation overhead during flash attention (#22732); BF16 support in `GET_ROWS` (#21391); Q5_K reorder MMVQ/dequant + Q8_0 reorder MMVQ (#22152); Battlemage AOT build via `spir64_gen` (#22147); add FILL, CUMSUM, DIAG, SOLVE_TRI, SSM_SCAN, GATED_DELTA_NET (#22149); non-contiguous input in PAD op (#22148).
  - **vulkan**: flash attention MMA / Tiles for MiMo-V2.5 (#22812); fix spv shadowing (#22760).
  - **hexagon**: HTP kernel for `GGML_OP_GATED_DELTA_NET` (#22837); l2 norm (#22816); process M-tail rows on HMX instead of HVX (#22724).
  - **opencl**: q4_0 MoE GEMM for Adreno (#22731); refactor Adreno q4_0 (#22335); use `CL_DEVICE_GLOBAL_MEM_SIZE` for `--fit` memory estimate (#22688); add opfilter regex for debugging (#22782).
  - **ggml-cpu**: fuse `RMS_NORM + MUL` on CPU backend (#22423); optimized risc-v q1_0 dot.
  - **ggml**: fast Walsh-Hadamard transform for KV rotation (#22631); bump version to 0.11.0; update `SCHED_DEBUG` output to use `ggml_op_desc()` (#22825).
  - **graph**: handle non-contiguous Q/K/V in `mul_mat_aux` (#22630).
  - **rpc**: use graph uid instead of graph cache (#22701).
  - **convert**: fix RuntimeError when stripping FP8 KV-cache scales (#22818); ignore non-language tensors for Gemma4Model (#22753); add `filter_tensors` method (#22597).
  - **gguf-py**: bump to 0.19.0 (#22664); migrate to PEP 621 and add uv support (#21907).
  - **webui**: import/export of settings (#22803); LLM title generation for agentic conversations (#22840); fix `?model=` URL param race in router mode (#22771); remove Google favicons (#22719); accessibility fixes (#22699, #22773).
  - **build/deps**: update BoringSSL to 0.20260508.0 (#22839); cpp-httplib 0.43.3 (#22686); upgrade default intel compute-runtime in docker (#22567); update Nix systems (#22869).

## v0.8.4

### Changed

- **llama.cpp submodule** — Updated from e48034dfc to eff06702b (12 commits).
  - **model**: move `load_hparams` and `load_tensors` to per-model definition (#22004)
  - **server**: implement `/models?reload=1` (#21848); add a simple `get_datetime` server tool (#22649)
  - **CUDA**: use fastdiv for batch index split in `get_rows` (#22650)
  - **vulkan**: delete dead `GGML_VK_MAX_NODES` def (#22621)
  - **ggml-webgpu**: add layer norm ops (#22406)
  - **kleidiai**: update to v1.24.0 and use release archive (#22549)
  - **common/autoparser**: fixes for newline handling / forced tool calls (#22654)
  - **webui**: fix circular dependency between `chat.service.ts` and `models.svelte.ts` (#22625); restore missing settings (#22666)
  - **examples**: refactor diffusion generation (#22590)
  - **docs**: update speculative decoding parameters after refactor (#22539)

## v0.8.3

### Changed

- **llama.cpp submodule** — Updated from b97ebdc98 to e48034dfc (14 commits).
  - **common**: determine generation prompt using longest common prefix (#22657)
  - **convert**: Mistral format yarn `apply_scale` support (#22612); apply Q/K RoPE permutation in NVFP4 repack path (#22611); disable uint types (#18908)
  - **CUDA**: fix device PCI bus ID de-dupe OOMing (ignoring other 3 GPUs entirely) (#22533)
  - **server**: avoid checkpoint data host copies (#22558)
  - **ggml-virtgpu**: fix circular dependency in headers (#22557)
  - **opencl**: Adreno optimization for MoE - MxFP4 (#22301)
  - **hexagon**: HMX flash attention (#22347)
  - **ggml**: bump version to 0.10.2; sync ggml; try fix win32 build

## v0.8.2

### Changed

- **llama.cpp submodule** — Updated from d77599234 to b97ebdc98 (18 commits).
  - **llama-quant**: fix `--tensor-type` when default `qtype` is overriden (#22572); add fast matmul iquants (#22504)
  - **CUDA**: fix tile FA kernel on Pascal (#22541)
  - **vulkan**: support asymmetric FA in coopmat2 path (#21753); add get/set tensor 2d functions (#22514)
  - **ggml-webgpu**: fix vectorized handling in mul-mat and mul-mat-id (#22578); add the upscale shader (#22419); improve performance of mat-vec and mat-mat for `MUL_MAT_ID` (#22464)
  - **hexagon**: enable non-contiguous row tensor support for unary ops (#22574)
  - **llama-mmap**: use `ftello`/`fseeko` (#22497)
  - **spec**: fix draft model checkpoints (#22521); fix vocab compat checks in spec example (#22426); fix argument typo (#22552)
  - **common**: check for null `getpwuid` in hf-cache (#22550)
  - **webui**: Spring Cleaning Refactor v1 (#22505)
  - **vendor**: update cpp-httplib to 0.43.2 (#22548)
  - **ci**: bump ty to 0.0.33 (#22535)
  - **scripts**: add `wc2wt.sh` - create worktree from current HEAD (#22513)

## v0.8.1

### Changed

- **llama.cpp submodule** — Updated from 98dc1418e to d77599234 (49 commits).
  - **server**: use `pos_next` instead of `n_tokens` for m-rope (#22439); (router) forward form-data to model server (#22118)
  - **CUDA**: fuse SSM_CONV + ADD(bias) + SILU (#22478); refactor fusion code (#22468); Blackwell native NVFP4 support (#22196); flash-attn support for DKQ=320/DV=256 with `ncols2=32` (#22286); better coalesce data-access for contiguous concat (#22330)
  - **ggml-cpu**: disable tiled matmul on AIX to fix page boundary segfault (#22293); append `xsmtvdotii` march for SpacemiT IME (#22317); re-enable fast `gelu_quick_f16` (#22339); optimize avx2 q6_k (#22345); SVE-tuned `gemm_q8_0_4x8_q8_0` kernel (#21916)
  - **ggml-webgpu**: fix FlashAttention support check (#22492); fix buffer aliasing for `ssm_scan` (#22456); add Q1_0 support (#22374)
  - **vulkan**: coalesce Q4_K/Q5_K scale loads (#21751); add barrier after `writetimestamp` (#21865)
  - **ggml**: bump version to 0.10.1; use 64-byte aligned tile buffers (#21058); skip already-registered backends and devices (#22296); revert to `-lm` linking instead of `find_library` (#22355); improve SPIR-V headers detection with `__has_include` (#21918)
  - **hexagon**: make vmem and buffer-size configurable (#22487); guard HMX clock request for v75+ platforms (#22377)
  - **spec**: discard last drafted token with low prob (#22506); refactor params (#22397)
  - **common**: do not pass prompt tokens to reasoning budget sampler (#22488); re-arm reasoning budget after DONE on new `<think>` (#22323); intentionally leak logger instance to fix hanging on Windows (#22273); fix missing exports in `llama-common` (#22340)
  - **chat**: fix handling of space in reasoning markers (#22353); handle gemma4 parsing edge cases (#22420)
  - **convert**: add support for Nemotron Nano 3 Omni (#22481); remove `input_scale` for dequantized fp8 modelopt (#22356)
  - **model**: remove duplicate `wo_s` scale after `build_attn` (Qwen3, LLaMA) (#22421)
  - **opencl**: add iq4_nl support (#22272)
  - **CANN**: add new ops, optimize existing ops (#21204)
  - **TP**: fix delayed AllReduce + zero-sized slices (#22489)
  - **rpc**: fix rpc-server cache on Windows (#22394)
  - **download**: prefer q8_0 when q4_k not available (#22428)
  - **webui**: fix slow mic stop and WAV encode (#22480); add Server tools (#21237)

## v0.8.0

### Changed

- **llama.cpp submodule** — Updated from 550d684bd to 98dc1418e (30 commits).
  - **server**: fix swa-full logic (#22288); rename debug tags to match `--cache-idle-slots` (#22292); `convert_anthropic_to_oai` also copy `chat_template_kwargs` (#22154); fix heap-buffer-overflow from negative `n_discard` (CVE-2026-21869) (#22267); (anthropic API) fix prefix caching (#21793)
  - **CUDA**: reduce MMQ stream-k overhead (#22298)
  - **metal**: optimize Metal Tensor API usage for `GGML_OP_MUL_MAT` (#20962); print GPU description (#22318)
  - **SYCL**: optimize Q4_0 `mul_mat` for Arc770, add scripts (#22291); fix build number for SYCL release (#22283)
  - **hexagon**: bump HMX frequency to max corner (#22334); use DIRID 13 in `libggml-htp.inf` for modern InfVerif (#22306); add SOLVE_TRI op (#21974); add basic and extended op profiling (#22269)
  - **ggml-webgpu**: support for SSM_SCAN and disable `set_rows` error checking (#22327); enable `FLASH_ATTN_EXT` on browser without subgroup matrix (#22199)
  - **llama-quant**: default ftype param `Q5_1` → `Q8_0` (#20828)
  - **spec**: fix vocab compat checks (#22358)
  - **parser**: fix structured output bug (#22302)
  - **common**: fix jinja warnings with clang 21 (#22313)
  - **vendor**: update LibreSSL to 4.3.1 (#22285)

## v0.7.9

### Changed

- **llama.cpp submodule** — Updated from 45cac7ca7 to 550d684bd (69 commits).
  - **server**: Enable transcriptions API for LFM2-Audio (#22000); ignore reasoning content from transcription api (#21905); allow cancel loading model (#21814); fix hardcoded proxy connection timeout in router mode (#22003)
  - **metal**: fix event synchronization (#22260); workaround macOS GPU interactivity watchdog (#22216)
  - **ggml-base**: use `MATH_LIBRARY` variable instead of hardcoded `m` (#22239)
  - **ggml**: bump version to 0.10.0
  - **SYCL**: update oneapi 2025.3.3, separate SYCL build, release Ubuntu 24 package (#22078); fused MoE `mul_mat_vec_q` for TG (#21920); improve `mul_mat_id` memory efficiency and add BF16 fast path (#22119)
  - **CUDA**: fuse relu + sqr (#22249); flush legacy pool on OOM and retry (#22155)
  - **HIP**: flip `GGML_HIP_GRAPHS` to default on (#22254)
  - **ggml-webgpu**: add support for im2col (#22259); implement async tensor api and event api (#22099); fused RMS_NORM + MUL (#21983); conv2d kernels (#21964); reset CPU/GPU profiling time when freeing context (#22050)
  - **vulkan**: Support F16 OP_FILL (#22177)
  - **hexagon**: add support for FILL op (#22198); DAIG op (#22195); fix missing v79 entry in `libggml-htp.inf` (#22194)
  - **mtmd**: also support `LLAMA_ROPE_TYPE_NONE` (#22242); update HunyuanVL vision-language model support (#22037); correct `mtmd_decode_use_mrope()` (#22188); add support for Reka Edge 2603 (#21616)
  - **chat**: fix `parallel_tool_calls` default setting based on model capabilities, add tests for parallel tool calls and structured outputs (#22217)
  - **common**: refactoring sampler parameters (#22233); refactor, move all conversion functions to common, add tests (#20690)
  - **speculative**: add checkpoint support (#22227); reset `i_last` when low acceptance streak occurs (#22168); `--spec-default` arg (#22223)
  - **convert**: handle ModelOpt produced mixed precision model during convert to GGUF (#22247)
  - **openvino**: driver setup, CI split, thread safety, and NPU optimizations (#21944)
  - **llama-ext**: fix exports (#22202)
  - **vendor**: update cpp-httplib to 0.43.1 (#22143)

### Fixed

- **build**: Added `-DLLAMA_OPENSSL=OFF` to suppress upstream HTTPS dependency pulled in by the new `LLAMA_OPENSSL=ON` default.

## v0.7.8

### Changed

- **llama.cpp submodule** — Updated from 30dce2cf2 to 45cac7ca7 (7 commits).
  - **model**: Gemma4 model type detection (#22027)
  - **mtmd**: add missing struct tag (#22023)
  - **libs**: rename `libcommon` → `libllama-common` (#21936)
  - **CUDA**: use LRU based eviction for cuda graphs (#21611)
  - **OpenCL**: refactor q8_0 `set_tensor` and `mul_mat` host side dispatch for Adreno (#21938)
  - **ggml-webgpu**: fix compiler warnings and refactor FlashAttention encoding (#21052)
  - **ci**: add android arm64 build and release (#21647)

## v0.7.7

### Changed

- **llama.cpp submodule** — Updated from 408225bb1 to 30dce2cf2 (18 commits).
  - **model**: using single llm_build per arch (#21970), refactor QKV into common `build_qkv` and `create_tensor_qkv` helpers (#21245), support NVFP4 tensors for Gemma4 (#21971)
  - **cli**: use `get_media_marker` (#22017)
  - **server**: tests fetch random media marker via `/apply-template` (#21980)
  - **convert**: fix NemotronH config parsing (#21664)
  - **ggml**: add `graph_reused` (#21764)
  - **ggml-cpu**: 128-bit RVV implementation for Quantization Vector Dot (#20633), SIMD gemm kernel for RISC-V vector extension (#20627)
  - **Metal**: implement ROLL op (#21946)
  - **OpenCL**: add q5_K gemm and gemv kernels for Adreno (#21595)
  - **SYCL**: fix Q8_0 reorder garbage on 2nd prompt + crash on full VRAM (#21638)
  - **hexagon**: optimize HMX matmul operations (#21071)
  - **ggml-webgpu**: compute pass batching and remove profiling overhead (#21873)
  - **cmake**: use glob to collect `src/models` sources (#22005)
  - **ci**: use ggml-org/ccache-action on RISC-V (#21632)
  - **devops**: add spirv-headers to nix (#21965)

## v0.7.6

### Changed

- **llama.cpp submodule** — Updated from a8bad3842 to 408225bb1 (28 commits).
  - **server**: use random media marker (#21962), support OAI `/v1/audio/transcriptions` API (#21863)
  - **chat**: dedicated DeepSeek v3.2 parser + "official" template (#21785)
  - **autoparser**: support case of JSON_NATIVE with per-call markers (test case: Reka-Edge) (#21892)
  - **common**: handle gemma4 parsing edge cases (#21760), skip reasoning budget sampler when no budget is requested (#21870)
  - **mtmd**: add `mtmd_image_tokens_get_decoder_pos()` API (#21851)
  - **llama**: read `n_ctx` back after making `llama_context` (#21939)
  - **CUDA**: Q1_0 initial backend (#21629), require explicit opt-in for P2P access (#21910), manage NCCL communicators in context (#21891)
  - **Metal**: fix FA support logic (#21898), add XIELU unary op (#20802)
  - **Vulkan**: optimize im2col (#21713), support GGML_TYPE_NVFP4 (#21455), programmatically add RoundingModeRTE to all shaders when the device supports it (#21572)
  - **ggml-webgpu**: fix dequantization helpers to not pass in pointers (#21872), update register tiling matmul to use f32 accumulation (#21644)
  - **ggml**: remove `ggml-ext.h` (#21869), fix ARM NEON nvfp4 dot product on non-dotprod targets (#21559)
  - **hexagon**: optimization for HMX mat_mul (#21554)
  - **rpc**: add native RDMA transport for RPC backend (RoCEv2) (#20590)
  - **vendor**: update BoringSSL to 0.20260413.0 (#21881)
  - **cmake**: fix CMP0194 warning on Windows with MSVC (#21630)
  - **ci**: re-enable mac workflows (#21894), disable test-backend-ops on Vulkan llvmpipe run and restore default timeout (#21901)

## v0.7.5

### Changed

- **llama.cpp submodule** — Updated from 073bb2c20 to a8bad3842 (18 commits).
  - **mtmd**: add Gemma 4 audio conformer encoder support (#21421), qwen3 audio support (qwen3-omni and qwen3-asr) (#19441), use causal attn for gemma 4 audio (#21824), fix crash when sending image under 2x2 pixels (#21711)
  - **Vulkan**: Flash Attention DP4A shader for quantized KV cache (#20797)
  - **CUDA**: limit DeviceSegmentedSort to immediate mode (#21718), skip compilation of superfluous FA kernels (#21768)
  - **common**: add download cancellation and temp file cleanup (#21813)
  - **server**: expose build_info in router mode (#21835)
  - **convert**: force f16 or f32 on step3-vl conv weights (#21646)

## v0.7.4

### Changed

- **llama.cpp submodule** — Updated from d12cc3d1c to 073bb2c20 (42 commits).
  - **model**: make Gemma 4 shared-KV tail attn_k tensors optional on load (#21739), fix multimodal padding token for gemma3n/gemma4 (#21625)
  - **mtmd**: add MERaLiON-2 multimodal audio support (#21756), support dots.ocr (#17575)
  - **common**: better align to the updated official gemma4 template (#21704), enable reasoning budget sampler for gemma4 (#21697), add callback interface for download progress (#21735), fix when loading cached HF models with unavailable API (#21670), mark `--split-mode tensor` as experimental (#21684), add fluidity to the progress bar (#21671), fix ambiguous grammar rule in gemma4 (#21661), simplify autoparser tagged parser rules (#21216), skip non-primary GGUF split files when selecting model (#21633)
  - **server**: ignore `--alias` when using `--models-preset` (#21380), fix grammar commandline args (#21543)
  - **jinja**: support `ensure_ascii=true`, string repetition and int/float self-filtering (#21623)
  - **vocab**: add gemma4 tokenizer tests, fix edge case (#21534)
  - **structured output**: fix broken structured output when using `$refs` in json_schema (#21699)
  - **ggml**: backend-agnostic tensor parallelism (experimental) (#19378), fix missing GGML_TYPE_Q1_0 cases (#21716), check return value of CUB calls in argsort and top-k (#21676)
  - **CUDA**: fuse muls (#21665), also store `node->src` ne/nb for graph equality (#21736)
  - **Metal**: add missing mm-id specializations for q1_0 (#21662)
  - **Vulkan**: support Q1_0 (#21539), unify type macros to use Vx instead of _VECx (#21605)
  - **SYCL**: add flash-attn support for head size 512 (#21654)
  - **HIP**: add CDNA4 (gfx950) architecture support for MI350X/MI355X (#21570)
  - **OpenCL**: add basic support for q5_k (#21593)
  - **WebGPU**: support non-square subgroup matrix configs for Intel GPUs (#21669), address quantization precision and backend lifecycle management (#21521)
  - **hexagon**: add support for linux on snapdragon (#21707), improved Op queuing, buffer and cache management (#21705)
  - **TP**: fix Qwen 3 Next data split (#21732)
  - **webui**: static build output improvements (#21667), add "Send message on Enter" setting (#21577), add option to pre-encode conversation for faster next turns (#21034), fix Model Selector choice sync (#21628)

## v0.7.3

### Changed

- **llama.cpp submodule** — Updated from b8635075f to d12cc3d1c (55 commits).
  - **model**: add HunyuanOCR support (#21395), support step3-vl-10b (#21287)
  - **llama**: remove per-arch tensor name lists (#21531), correct platform-independent loading of BOOL metadata (#21428)
  - **server**: respect the ignore eos flag (#21203), fix model params not propagated (#21509), fix restore for checkpoints with `pos_min == 0` (#21510), handle unsuccessful sink.write in chunked stream provider (#21478), fix logging of build + system info (#21460)
  - **kv-cache**: extend cache quantization checks (#21586), support attention rotation for heterogeneous iSWA (#21513)
  - **vocab**: remove `</s>` eog token for gemma4 (#21492), add byte token handling to BPE detokenizer for Gemma4 (#21488)
  - **gemma**: perform per-layer projections in the first layer (#21612)
  - **unicode**: add custom Qwen2 regex handler to fix segfault on long input (#21257)
  - **parser**: fix MiniMax handling (#21573)
  - **convert**: set `add bos == True` for Gemma 4 (#21500), fix `block_ff_dim` retrieval for lfm2 (#21508)
  - **ggml**: add Q1_0 1-bit quantization support (CPU) (#21273), deprecate `GGML_OP_ADD1` (#21363), free `ctx_copy` in `ggml_opt_free` to plug per-training-session leak (#21592)
  - **metal**: Q1_0 backend (#21528)
  - **CUDA**: also store `node->src->data` ptrs for equality check (#21635), check for buffer overlap before fusing (#21566), make cuda graphs props check faster (#21472), write an optimized `flash_attn_stream_k_fixup` kernel (#21159), `ds_read_b128` for q4_0 and q4_1 mmq kernels (#21168), fix CDNA2 compute capability constant for gfx90a/MI210 (#21519)
  - **SYCL**: Add Q8_0 reorder optimization (~3x tg speedup on Intel Arc) (#21527), handle other FA case (#21377)
  - **Vulkan**: add FA dequant for q4_1, q5_0, q5_1, iq4_nl (#21029), Linux output error string for errno on fork failure (#20904)
  - **WebGPU**: query for adapter support when registering backend (#21579), parameterize submission size and add iOS specific limits (#21533), add support of `MUL_MAT_ID` (#21147)
  - **hexagon**: slight optimization for argsort output init (#21463)
  - **webui**: store reasoning_content so it is sent back in subsequent requests (#21249), fix syntax highlighting lost after streaming (#21206), detect streaming state in reasoning content blocks (#21549), fix RTL text rendering (#21382), send both `backend_sampling == false/true` (#18781)
  - **cli**: fix stripping of `\n` in multiline input (#21485)
  - **llama-bench**: add `-fitc` and `-fitt` arguments (#21304)
  - **devops/ci**: provide KleidiAI-enabled ARM release artifact (#21259), lower cuda12 floor to 12.8.1 for broader host compatibility (#21438), fix vulkan workflow referencing non-existent action (#21442), use default RISE RISC-V Runners (#21263)

## v0.7.2

### Fixed

- **NIF signature mismatch on precompiled builds** — When `LLAMA_BACKEND` is set, the build now forces compilation from source instead of downloading a precompiled NIF that may have a stale function signature. (#23)
- **Precompile workflow CI failures** — The CI Checks job in the precompile workflow used a stale cached NIF (arity 9 vs 10 for `model_load`) because the cache key didn't include C source hashes and `mix compile` ran under the wrong `MIX_ENV`. Aligned with `ci.yml` by adding `c_src/**` to the cache key, compiling for `MIX_ENV=test`, and running `mix clean` before compile.
- **Precompile archive version mismatch** — The precompile and checksum jobs now set `@version` from the git tag (via `sed`), matching what the publish job already did. Previously, archives were named with the old version from `mix.exs`, causing the publish job to fail when looking for archives matching the tag version.

## v0.7.1

### Added

- **Full llama.cpp optimization parameters** — Exposed 17 new context parameters and 1 model parameter:
  - KV cache quantization: `type_k`, `type_v` (f16, q8_0, q4_0, etc.) for 2-4x memory savings
  - Flash attention & GPU offload: `flash_attn`, `offload_kqv`, `op_offload`
  - RoPE scaling: `rope_scaling_type`, `rope_freq_base`, `rope_freq_scale`, YaRN parameters
  - Misc: `attention_type`, `no_perf`, `swa_full`, `check_tensors`

## v0.7.0

### Added

- **Prefix caching** — Same-slot KV cache reuse for multi-turn chat. When a new request shares a prefix with the slot's previous request, the common prefix is skipped during prefill. 1.23x faster for multi-turn conversations. Controlled by `cache_prompt` option (default `false`, opt-in). Includes prefix-affinity slot selection. See [ADR 007](docs/adr/007-prefix-caching.md).

- **Pluggable batching strategies** — Extracted batch building into `BatchStrategy` behaviour with three built-in strategies: `DecodeMaximal` (default, generation-latency optimized), `PrefillPriority` (throughput optimized), `Balanced` (fair split). Custom strategies can implement the behaviour. See [ADR 008](docs/adr/008-batching-strategies.md).

- **Pre-tokenized API** — `Server.generate_tokens/3`, `Server.stream_tokens/3`, and `Server.get_model/1` allow callers to tokenize outside the GenServer, reducing mailbox contention under concurrent load.

- **HuggingFace Hub integration** — New `LlamaCppEx.Hub` module with `search/2` (find GGUF models), `list_gguf_files/2` (with file sizes via tree API), `download/3` (with local caching, ETag support, offline mode via `LLAMA_OFFLINE=1`), and `get_model_info/2`. Authentication via `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env vars. New `LlamaCppEx.load_model_from_hub/3` convenience wrapper. Requires optional `:req` dependency.

- **Performance guide** — New `docs/performance.md` with server tuning, prefix caching patterns, strategy selection guide, and optimization recipes.

- **Benchee benchmarks** — New `bench/prefix_cache.exs`, `bench/strategies.exs`, `bench/tokenize_overhead.exs` for measuring prefix cache impact, strategy comparison, and tokenization overhead.

### Changed

- **Graceful batch_eval error handling** — The server now fails active slots with error replies instead of crashing the GenServer when `batch_eval` returns an error (e.g., KV cache overflow).

### Fixed

- **CI warning suppression** — Suppress `-Wunused-function` warnings from vendored llama.cpp jinja headers (`runtime.h`, `lexer.h`).

## v0.6.14

### Changed

- **llama.cpp submodule** — Updated from 50e0ad08f to b8635075f (7 commits).
  - **common**: add Gemma 4 specialized parser (#21418), respect specified tag fallback when tag is empty (#21413)
  - **llama-model**: read `final_logit_softcapping` for Gemma 4 (#21390)
  - **llama**: add custom newline split for Gemma 4 (#21406)
  - **server**: fix undefined timing measurement errors in server context (#21201)
  - **ggml-webgpu**: move from parameter buffer pool to single buffer with offsets (#21278)
  - **ci**: add Windows Vulkan backend testing on Intel (#21292)

## v0.6.13

### Changed

- **llama.cpp submodule** — Updated from 95a6ebabb to 50e0ad08f (32 commits).
  - **server**: save and clear idle slots on new task (`--clear-idle`) (#20993)
  - **common/parser**: fix call ID detection (Mistral parser mostly) + atomicity for tag-json parsers (#21230)
  - **common**: fix tool call type detection for nullable and enum schemas (#21327), add commentary rules for gpt-oss-20b (#21286)
  - **chat**: avoid including json in chat.h (#21306), add Granite 4.0 chat template (#20804), Gemma4 tool response support
  - **jinja**: coerce input for string-specific filters (#21370)
  - **vocab**: fix Gemma4 tokenizer (#21343)
  - **ggml**: bump to 0.9.11 (ggml/1456)
  - **ggml-webgpu**: add vectorized flash attention (#20709)
  - **ggml-zendnn**: add MUL_MAT_ID op support for MoE models (#21315)
  - **rpc**: reuse compute graph buffers (#21299)
  - **kv-cache**: do not quantize SWA KV cache (#21277)
  - **SYCL**: fix llama_kv_cache hang when kv_cache is huge: 5GB (#21283)
  - **hexagon**: add cumsum op support (#21246)
  - **model/mtmd**: fix gguf conversion for audio/vision mmproj (#21309)
  - **tests**: add unit test coverage for llama_tensor_get_type (#20112), allow exporting graph ops from HF file without downloading weights (#21182)
  - **fix**: remove stale assert (#21369), fix gemma 4 template (#21326)

## v0.6.12

### Changed

- **llama.cpp submodule** — Updated from 08f21453a to 95a6ebabb (37 commits).
  - **CUDA**: add FA support for head dim 512 (#20998), fix FA kernel selection logic (#21271), add generic NVFP4 MMQ kernel (#21074), fix kernel selection for mmvq mmid kernel (#21238)
  - **opencl**: fix leak in Adreno q8_0 path (#21212)
  - **ggml**: bump to 0.9.10 (ggml/1454), fix RWKV ops thread assignment (#21226)
  - **ggml-cpu**: fix fallback for RVV kernels without zvfh (#21157)
  - **ggml-webgpu**: quantized buffers to u32 + wider browser/device support (#21046), port AOT operators to JIT (#20728)
  - **kleidiai**: add CPU feature detection to CI run script (#20394)
  - **hexagon**: improve RMS_NORM and DIV accuracy (#21251)
  - **SYCL**: support nvfp4 in mul_mat (#21227), enhance fattn perf (#21185)
  - **CANN**: fix multi-thread set_tensor race conditions (#20151)
  - **memory**: respect unified KV cache in hybrid memory for eval tasks (#21224)
  - **llama**: rotate activations for better quantization (#21038), refactor llama_model_quantize_params to pure C interface (#20346)
  - **common**: gpt-oss handle builtin/unsolicited tool calls (#21213), cleanup logs and modernize progress bar (#21215), disable backend sampling if reasoning budget enabled (#21209), add bounds check to prevent segfault on failed model load (#21082), move up common_init() and fix Windows UTF-8 logs (#21176)
  - **server**: bypass API key validation for WebUI static assets (#21269), no more gzip compression for webui (#21073), cleanup dual representation to openai-compat (#21090)
  - **fix**: tool call parsing for LFM2/LFM2.5 (#21242), correct misspellings (#21217), use lower-case proxy headers (#21235), include API key in CORS proxy for MCP (#21193)
  - **vendor**: update BoringSSL to 0.20260327.0 (#21211)

## v0.6.11

### Changed

- **llama.cpp submodule** — Updated from 82b703f8b to 08f21453a (21 commits).
  - **opencl**: add q4_K gemm and gemv kernels for Adreno (#20919)
  - **CUDA**: fix CUB's argsort when nrows % block_size == 0 (#21181), optimize MOE GEMV kernel for BS > 1 (#20905)
  - **jinja**: handle empty expressions correctly (#20913)
  - **common/parser**: fix handling of tool definition with missing properties key (#21128), add reasoning_format = none support to gpt-oss (#21094)
  - **common/json-schema**: fix non-capturing groups in pattern converter (#21124)
  - **common**: add character class support to glob_match (#21111)
  - **server**: wrap headers for mcp proxy (#21072), fix processing of multiple back-to-back mtmd chunks (#21107)
  - **model**: add missing ROPE_FACTORS_LONG/SHORT for MiniCPM (#21150)
  - **llama-model-loader**: print warning when using overrides with mmap (#20978)
  - **hexagon**: dma optimizations (#21137)
  - **SYCL**: enhance build script to use half cores to avoid OS hang (#21093)
  - **rpc**: fix misleading error log (#21184)

## v0.6.10

### Changed

- **llama.cpp submodule** — Updated from 5c1a7b835 to 82b703f8b (7 commits).
  - **vendor**: update cpp-httplib to 0.40.0 (#21100)
  - **vulkan**: add noncontiguous GLU support (#21081)
  - **common/parser**: fix reasoning whitespace bugs + extra parser tests (#21085)
  - **cli**: add /glob command (#21084)
  - **webui**: conversation forking + branching improvements (#21021)
  - **docker**: fix and enable ARM64 image build (#20929)

## v0.6.9

### Changed

- **llama.cpp submodule** — Updated from 9f102a140 to 1743d9805 (38 commits).
  - **model**: F2LLM-v2 support, allow causal_attn and pooling_type on all architectures (#20973)
  - **convert**: register Qwen3Model architecture (#20967), support Qwen3.5/Qwen3.5 Moe NVFP4 and add input scales (#20505), add RuGPT3XL support (#21011)
  - **ggml-cuda**: add NVFP4 dp4a kernel (#20644), support F32 kernel type for CONV_TRANSPOSE_2D (#17094)
  - **hip**: use fnuz fp8 for conversion on CDNA3 (#21040)
  - **opencl**: allow large buffer for Adreno (#20997)
  - **jinja**: fix macro with kwargs (#20960)
  - **common**: make LLAMA_CACHE the one cache for everything (#21009), fix split model migration (#21019), fix verbosity setup (#20989), add getpwuid fallback for HF cache (#21035), filter out imatrix when finding models (#21023)
  - **llama**: fix llama-model-saver (#20503)
  - **mtmd**: add DeepSeekOCR support (#17400), refactor image preprocessing (#21031), fix quant and im2col ops on Metal for deepseek-ocr (#21027)
  - **imatrix**: fix crash with --show-statistics and zero counts (#19532)

## v0.6.8

### Changed

- **llama.cpp submodule** — Updated from 1772701f9 to 9f102a140 (15 commits).
  - **models**: move the token embedding norms to the first layer (#20943)
  - **ggml-backend**: re-enable graph reuse with pipeline parallelism (#20927)
  - **metal**: add FLOOR, CEIL, ROUND, TRUNC unary ops (#20930), add FA instantiations for HSK=512, HSV=512 (#20902)
  - **common**: add standard Hugging Face cache support (#20775), add a WARNING for HF cache migration (#20935), fix get_gguf_split_info (#20946), replace wrap_for_generation with a prefix convenience function (#20912)
  - **hexagon**: general DMA and Binary Op fixes for large strides (#20918)
  - **llama-fit**: fix regex pattern for gate_up tensors (#20910)
  - **vendor**: update cpp-httplib to 0.39.0 (#20933)

## v0.6.7

### Changed

- **llama.cpp submodule** — Updated from eac9c6ea8 to 1772701f9 (30 commits).
  - **rpc**: RCE patch (#20908), prevent division by zero in deserialize_tensor (#20712)
  - **memory**: fix seq_id bounds in llama_memory_recurrent::state_read_meta() (#20887)
  - **server**: use httplib dynamic threads (#20817), allow router to report child instances sleep status (#20849), fix Host header (#20843)
  - **metal**: add CONV_3D (#19927)
  - **common/autoparser**: detect reasoning markers when enable_thinking changes system prompt (#20859)
  - **common/grammar**: fix grammar parsing issues to prevent stack overflow and hangs (#18604)
  - **context**: use n_embd_out for pooled embedding extraction (#20840)
  - **jinja**: refactor token advancement (#20864)
  - **CUDA**: fix BF16 FA compilation (#20865), native bf16 flash attention for vec kernel (#20525), increase output elements per-thread block for small K-dimension (#20635)
  - **CANN**: add RoPE cache preload before ACL graph capture (#20747)
  - **opencl**: add q6_K gemm and gemv kernels for Adreno (#20089), add flattened Q4_K mv and general Q4_K mm (#20773)
  - **openvino**: explicit memset in buffer_context allocation (#20857)
  - **mtmd**: add dynamic high-resolution image preprocessing for InternVL model (#20847), fix LightOnOCR image preprocessing (#20877)
  - **ggml**: support bf16 and quantized type (#20803)
  - **webui**: improve chat form positioning (#20901), fix --webui-config-file settings not applied on load (#20823)

## v0.6.6

### Changed

- **llama.cpp submodule** — Updated from 6729d4920 to eac9c6ea8 (47 commits).
  - **context**: zero output buffer on allocation (#20781)
  - **model**: assert nextn_predict_layers to prevent underflow (#20783), fix Granite Hybrid type check for 7B.A1B (#20795)
  - **jinja**: fix heap OOB read in value equality comparison (#20782)
  - **common/parser**: fix nasty bug causing subtle corruption of generation prompt (#20825), fix out_of_range crash in throw path (#20777), add proper reasoning tag prefill reading (#20424), fix gpt-oss content removal (#20745)
  - **chat**: handle tool calls with no required args in TAG_WITH_TAGGED format (#20764)
  - **server**: fix router mode deadlock on child crash and TOCTOU race (#20763), add cached_tokens info to oaicompat responses (#19361), improve mtmd ctx checkpoints (#20726), become source of truth for sampling defaults (#20558)
  - **vulkan**: change gated_delta_net to shard across subgroup (#20662), dequantize iq4_xs 4 at a time (#20657)
  - **hip**: avoid compiler bug in RDNA code generation during debug builds on Windows (#20655)
  - **hexagon**: add Matrix Extensions (HMX) for NPU backend (#20693)
  - **CANN**: add BF16 support for core operators (#20152), handle in-place ROPE on non-contiguous f32 tensors (#20274), support flash attention for head dim not multiple of 16 (#20031)
  - **ggml-cpu**: add always_inline to tinyBLAS_PPC accumulator saves (#20791)
  - **ggml-webgpu**: ops support for qwen3.5 (SET, TRI_SOLVE, SSM_CONV, GATED_DELTA_NET) (#20687), add DIAG/TRI ops (#20664), update RMS_NORM/L2_NORM (#20665)
  - **vocab**: assert array size of scores and toktypes (#20737)
  - **convert**: support is_causal hyperparameter (#20746), make NVFP4/MXFP4 say correct type (#20730)
  - **cmake**: fix build warning when kleidiai is enabled (#20457), guard KleidiAI DOWNLOAD_EXTRACT_TIMESTAMP for cmake < 3.24 (#20767)

## v0.6.5

### Changed

- **llama.cpp submodule** — Updated from b6c83aad5 to 6729d4920 (26 commits).
  - **model**: add control vector support where missing (#20653)
  - **ggml**: bump version to 0.9.8 (ggml/1442), restore ggml_type_sizef() to avoid major version bump (ggml/1441)
  - **ggml-cpu**: fix RVV checks in quants and repacking (#20682), fix unused changemask warning in repack (#20692)
  - **ggml-blas**: set MKL threads from thread context (#20602)
  - **Vulkan**: async and event fixes (#20518), disable MMVQ on Intel Windows driver (#20672), allow graphics queue only through env var (#20599)
  - **HIP**: ignore return of hipMemAdvise (#20696)
  - **hexagon**: add neg, exp, sigmoid, softplus, cont, repeat ops (#20701)
  - **kleidiai**: fix MUL_MAT support for batched (3D) inputs (#20620)
  - **server**: fix ctx checkpoint invalidation (#20671)
  - **context**: fix graph not resetting when control vector changes (#20381)
  - **llama**: re-enable manual LoRA adapter free (#19983)
  - **common**: rework gpt-oss parser (#20393), add `--skip-chat-parsing` to force pure content parser (#20289)
  - **webui**: fix duplicated messages on q param (#20715), improve tooltip wording for attachment requirements (#20688)
  - **OpenCL**: no timeout for WaitAny in graph submission to avoid deadlocks on llvm-pipe backends (#20618)

## v0.6.4

### Changed

- **llama.cpp submodule** — Updated from 463b6a963 to b6c83aad5 (56 commits).
  - **model**: Mistral Small 4 support (#20649), Nemotron-H NVFP4 tensors (#20561), Qwen3.5/Qwen3.5MoE NVFP4 tensors (#20506)
  - **ggml**: OpenVINO backend (#15307), native AVX512-FP16 support for F16 operations (#20529), extend im2col f16 (#1434), guard against sumq2 being 0 in IQ4_NL (#20460)
  - **CUDA**: GDN shared mem latency hiding (#20537), limit FA stream-k block count (#20586), RDNA4-specific MMVQ for bs=1 decode (#19478), FP32 cuBLAS for V100 to avoid overflows (#19959), fix data race in cpy kernel (#20507), avoid creating CUDA context during device init (#20595)
  - **metal**: FA specialization for HSK=320, HSV=256 (#20549)
  - **Vulkan**: fix flash attention dot product precision (#20589), use graphics queue on AMD (#20551)
  - **HIP**: APU compatibility — soft error handling for hipMemAdviseSetCoarseGrain (#20536)
  - **SYCL**: fix untransposed GDA recurrent state (#20583), enhance UPSCALE to support all UT cases (#20637)
  - **OpenCL**: fix l2_norm (#20480)
  - **server**: support refusal content for Responses API (#20285), fix wait in test_cancel_requests() (#20601), fix model selector locked to first loaded model (#20580)
  - **tools/cli**: fix disable reasoning (#20606)
  - **convert**: support mixed-precision ModelOpt NVFP4/FP8 quantization (#20539), support contiguous method on lora tensors (#20489)
  - **kv-cache**: fix reading llama_kv_cell_ext during state read (#20273)
  - **common**: fix iterator::end() dereference (#20445)
  - **vendor**: cpp-httplib 0.37.2 → 0.38.0 (#20484, #20578)
  - **webui**: model information dialog (#20600), MCP CORS proxy detection (#20167), code preview iframe isolation (#20477)
  - **hexagon**: Q4_0 and MXFP4 repack fixes (#20527)

## v0.6.3

### Added

- **CI workflow** — New `.github/workflows/ci.yml` runs `mix compile --warnings-as-errors`, `mix format --check-formatted`, `mix test`, and `mix dialyzer` on push/PR to master.
- **Dialyzer** — Added `dialyxir` dependency for static analysis. All modules pass with zero warnings.
- **Example scripts** — New `examples/` directory with 6 runnable scripts: `basic_generation.exs`, `streaming.exs`, `chat.exs`, `structured_output.exs`, `embeddings.exs`, and `server.exs`.
- **Expanded test coverage** — New `test/schema_test.exs` covering `embeds_one`, `embeds_many`, additional Ecto types (`:date`, `:utc_datetime`, `:decimal`, `:map`), empty schemas, and end-to-end nested schema to GBNF conversion. Added edge case tests to `test/thinking_test.exs` for unicode content, nested/malformed tags, and very long content.

### Fixed

- **`Chat.apply_template/3`** — Now accepts string-keyed message maps (`%{"role" => ..., "content" => ...}`) in addition to atom-keyed maps and tuples.
- **`Schema.to_json_schema/1`** — Fixed Dialyzer opaque type warning (replaced `MapSet.member?/2` with `in` operator).
- **GitHub Actions Node.js 20 deprecation** — Updated `actions/checkout` to v5 and added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` env to precompile workflow, preparing for the June 2026 Node.js 24 migration.
- **Stream test reliability** — Fixed `stream with early halt` test to use a prompt compatible with instruction-tuned models.

### Changed

- **llama.cpp submodule** — Updated from fdb17643d to 463b6a963 (31 commits).
  - tools: enable kvu in perplexity for hellaswag, winogrande, multiple-choice (#19954)
  - graph: remove redundant GDN state transposes (#20443)
  - llama: fix pooling assertion crash in chunked GDN detection path (#20468), disable graph reuse with pipeline parallelism (#20463)
  - metal: fix l2 norm scale (#20493), avoid divisions in bin kernel (#20426)
  - Vulkan: add GATED_DELTA_NET op support (#20334), fix l2_norm epsilon handling (#20350), fix OOB check in flash_attn_mask_opt (#20296), fix ErrorOutOfHostMemory on Intel GPU with --no-mmap (#20059)
  - OpenCL: add cumsum op (#18981), use larger workgroup size for get_rows (#20316)
  - HIP: compile debug builds with -O2 to avoid compiler bug (#20392)
  - ggml-cpu: add RVV vec dot kernels for quantization types (#18859)
  - server: reset counter related to kill-switch on client error (#20513), auto-select first loaded model for new conversations (#20403)
  - common/parser: gracefully handle undetected tool parser (#20286), add GigaChatV3/3.1 models support (#19931)
  - grammar: fix root symbol check (#19761)
  - vendor: update cpp-httplib to 0.37.1 (#20390)
  - convert: better mtp check and fix return (#20419)

## v0.6.1

### Changed

- **llama.cpp submodule** — Updated from c5a778891 to fdb17643d (70 commits).
  - model: add support for Phi4ForCausalLMV, Nemotron 3 Super, Qwen3VL reranker text
  - ggml: add NVFP4 quantization type support
  - llama: chunked fused GDN path, dynamic head_dim and n_rot for SWA
  - metal: extend mul_mv_ext to BF16/Q2_K/Q3_K, fix q5_k register spill, add upscale, handle command buffer failures gracefully
  - CUDA/HIP: GDN shared mem for HIP, fix loop unrolling in ssm-conv, display VRAM capacity on init
  - Vulkan: add SGN and ELU ops, fix data races in coopmat1, skip zero size tensors in copies
  - SYCL: Flash Attention support for fp32/fp16/Q4/Q5/Q8
  - WebGPU: add REPEAT op, faster quant matrix operations
  - KleidiAI: concurrent SME and NEON kernel execution
  - ggml-cpu: add RVV repack GEMM/GEMV for quantization types
  - server: kill switch when stuck, fix checkpoints and OAI completion stream index
  - common: fix --n-cpu-moe/--cpu-moe for fused gate+up models, gracefully handle incomplete output
  - vendor: update cpp-httplib to 0.37.0, miniaudio to 0.11.25
  - llama-quant: fail early on missing imatrix, refactor type selection

## v0.6.0

### Added

- **Qwen 3.5 support** — llama.cpp updated to c5a778891 (35 commits since v0.5.0).
- **`reasoning_content` in ChatCompletion** — `chat_completion/3` now splits `<think>...</think>` blocks from the response when `enable_thinking: true`. The choice message includes `reasoning_content` (the thinking text) and `content` (the final answer). Returns `nil` when thinking is not enabled or no thinking block is present.
- **`reasoning_content` in ChatCompletionChunk** — `stream_chat_completion/3` emits chunks with `reasoning_content` in the delta while the model is thinking, then switches to `content` after `</think>`.
- **`LlamaCppEx.Thinking`** — New module with `parse/1` for one-shot parsing and `stream_parser/1` + `feed/2` for streaming token-boundary-safe parsing of think blocks. Handles the real-world Qwen3/3.5 template behavior where `<think>` is opened by the template itself.

### Changed

- **llama.cpp submodule** — Updated from 7f5ee54 to c5a778891.
  - ggml: add GATED_DELTA_NET op for Qwen 3.5 hybrid architecture
  - model: update Qwen 3.5 model type detection
  - convert: register Qwen 3.5 ForCausalLM for text only
  - CUDA: use shared mem for ssm_conv, improve performance via fewer synchronizations
  - Hexagon: add f32 ssm_conv, fp16 binary ops, Flash Attention optimizations
  - OpenCL: add l2_norm, neg, exp, diag ops
  - CPU: skip redundant ROPE cache updates, fix data race for debug asserts
  - quants: add memsets and other fixes for IQ quants
  - kv-cache: fix M-RoPE checkpoints, checkpoint every n tokens
  - server: preserve Anthropic thinking blocks in conversion

### Unchanged

- `chat/3` and `stream_chat/3` continue returning raw text (no breaking change).

## v0.5.0

### Added

- **Structured output via JSON Schema** — New `:json_schema` option on `generate/3`, `stream/3`, `chat/3`, `stream_chat/3`, `chat_completion/3`, and `stream_chat_completion/3`. Pass a JSON Schema map and the model output is automatically constrained to valid JSON matching the schema. Uses llama.cpp's built-in `json_schema_to_grammar()` under the hood.

  ```elixir
  schema = %{
    "type" => "object",
    "properties" => %{"name" => %{"type" => "string"}, "age" => %{"type" => "integer"}},
    "required" => ["name", "age"],
    "additionalProperties" => false
  }
  {:ok, json} = LlamaCppEx.chat(model, messages, json_schema: schema, temp: 0.0)
  ```

- **`LlamaCppEx.Grammar`** — New module for JSON Schema to GBNF conversion.
  - `from_json_schema/1` — returns `{:ok, gbnf_string}` or `{:error, reason}`
  - `from_json_schema!/1` — returns the GBNF string or raises

- **`LlamaCppEx.Schema`** — New module for converting Ecto schema modules to JSON Schema maps. Maps all standard Ecto types (`:string`, `:integer`, `:float`, `:boolean`, `:date`, `{:array, inner}`, etc.) and supports nested `embeds_one`/`embeds_many`. Automatically excludes `:id` and timestamp fields.

- **NIF: `json_schema_to_grammar_nif/1`** — Exposes llama.cpp's `json_schema_to_grammar()` via `nlohmann::ordered_json`.

### Changed

- **Elixir requirement** bumped to `~> 1.18` (for built-in `JSON.encode!/1`).
- **Dependencies** — added `{:ecto, "~> 3.0", optional: true}` for optional Ecto schema integration.

## v0.4.4

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8198).
  - ggml: fix `ggml_is_contiguous_n` for ne == 1
  - ggml: use simple `std::thread` in AMX without OpenMP
  - KleidiAI: add SME fp16 compute path for q4_0 GEMM on aarch64
  - OpenCL: add optimized q4_1 mm kernel for Adreno
  - Vulkan: tune MMVQ for Intel Windows
  - WebGPU: fix workgroup dispatch limit for large batch sizes
  - Fix locale-dependent float printing in GGUF metadata

## v0.4.3

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8185).
  - Vulkan: improve partial offloading performance on AMD
  - CUDA: cap grid.y at 65535 in non-contiguous dequantize/convert kernels
  - ggml-cpu: optimise s390x multiply extend instructions
  - Vendors: update cpp-httplib to 0.35.0, miniaudio to 0.11.24

## v0.4.2

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8179).

## v0.4.1

### Improved

- **Error handling** — `Chat.apply_template/3`, `Tokenizer.encode/3`, and `Tokenizer.decode/2` now return `{:error, reason}` instead of crashing when NIFs raise.
- **Telemetry documentation** — Server moduledoc documents all telemetry events, measurements, and metadata.
- **Typespecs** — Added `@spec` to `Server.start_link/1`.

### Changed

- **llama.cpp submodule** — Updated to latest upstream (b8157).

## v0.4.0

### Added

- **Full model loading params** — `main_gpu`, `split_mode`, `tensor_split` for multi-GPU placement; `use_mlock` and `use_direct_io` for memory control; `vocab_only` for cheap model introspection without loading weights.
- **Server GPU forwarding** — `Server.start_link/1` now forwards `main_gpu`, `split_mode`, `tensor_split`, `use_mlock`, and `use_direct_io` to `Model.load/2`.

## v0.3.0

### Added

- **Jinja chat templates** — switched from `llama_chat_apply_template()` C API to the full Jinja-based `common_chat_templates_apply()` engine from llama.cpp's common library.
- **`enable_thinking` option** — pass `enable_thinking: false` to `Chat.apply_template/3`, `chat/3`, `stream_chat/3`, `chat_completion/3`, and `stream_chat_completion/3` to disable CoT reasoning for models like Qwen3/3.5.
- **`chat_template_kwargs` option** — pass arbitrary key-value pairs to the Jinja template engine.
- **Penalty parameters** — `penalty_repeat`, `penalty_freq`, and `penalty_present` options for repetition/frequency/presence penalties in sampling.
- **OpenAI-compatible response format** — `chat_completion/3` and `stream_chat_completion/3` return `ChatCompletion` and `ChatCompletionChunk` structs.
- **Qwen3.5 benchmark results** in README — Qwen3.5-27B and Qwen3.5-35B-A3B on Apple M4 Max.

### Changed

- `Chat.apply_template/3` now uses the Jinja engine and takes the model ref directly (no longer accepts `:template` option for raw template strings).
- Linked `libcommon.a` from llama.cpp build (previously excluded).
- `LlamaModel` RAII wrapper now caches `common_chat_templates` at model load time.

## v0.2.0

### Added

- **Continuous batching server** (`LlamaCppEx.Server`) — GenServer with slot pool for concurrent multi-sequence inference. One forward pass per tick with decode tokens and prefill chunks mixed in a single batch.
- **Embeddings** (`LlamaCppEx.Embedding`) — `embed/3` and `embed_batch/3` with L2 normalization and configurable pooling type.
- **Grammar-constrained generation** — GBNF grammar support via `grammar` and `grammar_root` options in `Sampler.create/2` and `generate/3`.
- **Batched inference primitives** — `prefill/3`, `decode_batch/3`, `decode_token/4`, `batch_eval/2`, `sampler_sample_at/3` NIFs for building custom inference loops.
- **Streaming via Server** — `LlamaCppEx.Server.stream/3` for token-by-token streaming through the batched server.
- **Telemetry events** — `[:llama_cpp_ex, :server, :tick]` and `[:llama_cpp_ex, :server, :request, :done]` for observability.
- **Benchmark suite** (`bench/`) — Benchee-based benchmarks for single-sequence and server generation, plus a custom continuous batching harness measuring throughput scaling.

### Changed

- `Sampler.create/1` now requires the model as the first argument: `Sampler.create(model, opts)`.
- `Context.create/2` accepts new options: `:embeddings`, `:pooling_type`, `:n_seq_max`.

## v0.1.0

Initial release.

- Model loading and introspection
- Text generation with configurable sampling
- Streaming token generation via `Stream.resource/3`
- Chat template support
- Tokenization and detokenization
- Metal, CUDA, Vulkan, and CPU backends
- RAII resource management via `fine`
