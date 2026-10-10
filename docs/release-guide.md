# Release Guide

How to upgrade the llama.cpp submodule and publish a new release.

## Prerequisites

- Elixir 1.18+, Erlang/OTP 26+ (OTP 26/27/28 report NIF 2.17, OTP 29 reports
  2.18 — those are the two artifact flavours the release builds)
- cmake and git
- A GGUF model file for testing (e.g. Qwen3.5-0.8B)
- An embedding model file for embedding tests (e.g. Qwen3-Embedding-0.6B)

## 1. Update the submodule

```bash
# Fetch latest upstream commits
git -C vendor/llama.cpp fetch origin

# Check what's new since the current pin
git -C vendor/llama.cpp log --oneline HEAD..origin/master

# Checkout the target commit
git -C vendor/llama.cpp checkout <commit-hash>
```

Then update the Makefile's `LLAMA_COMMIT` to the same SHA. It is what a Hex
source build clones when `vendor/llama.cpp` is absent, so leaving it behind means
source builds get the old llama.cpp while git checkouts get the new one:

```bash
git -C vendor/llama.cpp rev-parse HEAD
# paste into LLAMA_COMMIT in Makefile
```

## 2. Check API compatibility

Before building, verify the llama.cpp APIs used by the NIF haven't changed:

Re-derive the header list if you add an include -- a hand-kept list drifts, and
that is exactly how the `common/speculative.h` break below reached a build:

```bash
grep -hoE '^#include [<"](llama|ggml|chat|common|json|speculative|\.\./)[^">]*' \
  c_src/llama_cpp_ex/*.cpp c_src/llama_cpp_ex/*.h | sort -u
```

```bash
for h in include/llama.h src/llama-ext.h \
         ggml/include/ggml-backend.h ggml/include/ggml-rpc.h \
         common/common.h common/json.h \
         common/chat.h common/json-schema-to-grammar.h common/speculative.h; do
  echo "##### $h"
  git -C vendor/llama.cpp diff <old-commit>..<new-commit> -- "$h"
done
```

A signature change is the easy case -- it fails to compile with a clear message.
Watch for two harder ones:

- **A function removed outright.** The compiler's "did you mean" is actively
  misleading: when `common_speculative_need_embd` was deleted in `f785fc9ea`,
  both GCC and clang suggested the unrelated `common_speculative_n_max`, whose
  first parameter happens to be a different pointer type. Check the header diff
  for `-` lines before believing the suggestion.
- **A default value changed** with the signature intact. Nothing fails to
  compile. `llama_model_default_params()` moved `load_mode` from
  `LLAMA_LOAD_MODE_MMAP` to `LLAMA_LOAD_MODE_AUTO`; because the NIF always sets
  that field explicitly, behaviour did not change -- but a field we left at its
  default would have shifted silently. Diff
  `llama_model_default_params` / `llama_context_default_params` on every bump.

The NIF uses these key APIs (grep `llama_nif.cpp` for the full list):
- `llama_model_*`, `llama_context_*`, `llama_vocab_*` — model/context/vocab management
- `llama_tokenize`, `llama_detokenize`, `llama_token_to_piece` — tokenization
- `llama_batch_*`, `llama_decode` — inference
- `llama_sampler_*` — sampling chain
- `llama_memory_*` — KV cache / memory management
- `llama_get_embeddings_*`, `llama_pooling_type` — embeddings
- `llama_chat_apply_template` — legacy chat templates
- `common_chat_templates_init`, `common_chat_templates_apply` — Jinja chat templates
- `json_schema_to_grammar` — grammar generation
- `common_speculative_*` — speculative decoding and MTP draft models
- `ggml_backend_dev_*`, `ggml_backend_reg_*` — device enumeration for `:devices`
- `ggml_backend_rpc_add_server`, `ggml_backend_rpc_start_server` — RPC backend

If any signatures changed, update `c_src/llama_cpp_ex/llama_nif.cpp` and/or `llama_nif.h`.

### Decision engine port

`c_src/llama_cpp_ex/decision.cpp` is a port of upstream code this build does not
compile: `tools/server/server-decision.{h,cpp}` (parsing, prompts, answers) and
the decision paths of `tools/server/server-context.cpp` (`send_decision`, the
"outputs of a decision are read from one batch" rules, the `/v1/systemone`
handler). It also mirrors the decision branch of `common_init_result` in
`common/common.cpp`, which decides the context shape, and uses
`llama_decision_order` from `src/llama-ext.h`, a staging header outside
`include/`. None of this is a public API, so a bump can change it with no
header break and the port keeps compiling against the old behaviour. Diff it:

```bash
git -C vendor/llama.cpp diff <old-commit>..<new-commit> -- \
  tools/server/server-decision.h tools/server/server-decision.cpp \
  src/llama-ext.h
git -C vendor/llama.cpp log --oneline <old-commit>..<new-commit> -- \
  tools/server/server-context.cpp common/common.cpp | grep -i -E 'decision|systemone'
```

Ported functions carry their upstream name in a trailing comment
(`// server_decision_context::format_answer`), so a hunk maps to one function.
Port every change except image input (libmtmd is not linked), update the
"Ported at" commit at the top of `decision.cpp`, then check the result against
upstream's own server with `scripts/decision_compare.exs` (its header has the
build and serve commands) on tinylaya, tinyopenjev and any decision model on
disk. A new decision *type* is a new branch in `init`, in `@types` in
`lib/llama_cpp_ex/decision.ex`, and in the two type lists beside it.

Run the comparison on the **Metal** build (NIF and reference server both). Two
CPU builds with identical cmake flags still differ by up to 9e-4 on
tinylaya's near-uniform random-weight outputs, which is inside the script's
1e-3 bar but enough to flip the argmax of its 12-option request and trip the
choice check; and at `c479922ac` the CPU-only `llama-server` aborts on Clef
Flash at startup (`GGML_ASSERT(*cur_backend_id != -1)` in
`resolve_fused_ops`, with or without `-fit off`, `-np 1`, `-dev none`), while
the NIF runs the same model fine. On Metal all three agree to 1.5e-6 or
exactly. Keep question *order* identical on both sides: clef decides jointly,
so `[route, angry]` and `[angry, route]` are different prompts — build the
server's JSON from the same keyword list, not from a map (which sorts).

### Upstream defects we work around

Three known llama.cpp defects have workarounds in this repo. A bump is the only
time anyone looks at them, so check each one here — if upstream has fixed it, the
workaround should come out rather than quietly accumulate.

Each was measured against `4801e3c567d5` (b10362) on NVIDIA DGX Spark (GB10,
aarch64, GCC 13.3, CUDA 13.0). Full reports, with reproductions and suggested
upstream fixes, are drafted in
`.claude/plans/dgx-spark-2node/upstream-issues.md` — not yet filed, so there are
no issue URLs to link. **When they are filed, put the URLs in this table.**

Re-checked at `e85caa81e` (b10582): all three still stand. Every re-check so far
has been a source diff, not a re-measurement. At `a94d563ed801` the three files
involved (`ggml/src/ggml-cpu/CMakeLists.txt`, `ggml_backend_cuda_comm_init`,
`ggml_backend_rpc_start_server`) were untouched. At `e85caa81e` the CUDA and RPC
ones are still untouched — #26502 moved the tensor-split meta backend and was
reverted in `f20395dae` — while `ggml-cpu/CMakeLists.txt` did change: OpenMP
target variables, KleidiAI SME2 GEMV sources, and IntelLLVM fast-math gating,
none of it near the `-mcpu=native` probe. A source diff is enough to say a defect
is *still there*; it is not enough to say it is *gone*, so if a diff ever touches
the probe itself, run the command in the last column.

Re-checked at `465e49b9c` (b10830), again as a source diff: still all three.
`ggml_backend_rpc_start_server` and `ggml_backend_cuda_comm_init` are
untouched — the RPC diff is #26500 (do not serialise buffers that belong to
another server) and #27960 (`ggml_op_alloc_size_may_expand`), and the RPC
buffer's `set_tensor_2d`/`get_tensor_2d` hooks are still `NULL`. The
`ggml-cpu/CMakeLists.txt` diff adds `iqp.cpp` and gates the SpacemiT IME
kernel sources; the `-mcpu=native` probe is untouched.

Re-checked at `b6b003d2c` (b10944), covering the gap from `465e49b9c` in one
source diff: still all three. `ggml-rpc.cpp` was not touched at all — the RPC
buffer's `set_tensor_2d`/`get_tensor_2d` hooks are still `NULL` — and the
`ggml-cuda.cu` diff is #28079 (`GGML_FA_QUANTS`), #28604 (HIP `prop.integrated`
revert) and #26454 (gfx90c), none of them near `ggml_backend_cuda_comm_init`.
The `ggml-cpu/CMakeLists.txt` diff is #28091 (PCH and unity build, with GCC PCH
gated to x86) and #28667 (s390x `repack.cpp`); the `-mcpu=native` probe is
untouched.

Re-checked at `e117148a4` (b11424+1), covering the gap from `c85b92c69` in one
source diff: still all three. `ggml-cpu/CMakeLists.txt` was not touched. The
`ggml-rpc.cpp` diff only adds the new `alloc_buffer_n`/`get_alloc_size_n`
buffer-type slots (#23671), both `NULL`; `ggml_backend_rpc_start_server` and
the `NULL` `set_tensor_2d`/`get_tensor_2d` hooks are unchanged. The
`ggml-cuda.cu` diff is MMVQ/MMVF/fusion work plus the same `alloc_buffer_n`
slots, none of it near `ggml_backend_cuda_comm_init`.

Re-checked at `c479922ac` (b11458), covering the gap from `e117148a4`: #1 and
#3 still stand — `ggml-cpu/CMakeLists.txt` is untouched and
`ggml_backend_rpc_start_server` still returns `void` with the same signature.
**#2 has moved.** #26610 (RPC `-sm tensor`) implements exactly what the "Still
needed?" column names: the RPC buffer's `set_tensor_2d`/`get_tensor_2d` hooks
(and their `_async` backend variants) are no longer `NULL`, and the RPC backend
now exposes `ggml_backend_comm_init` with `RPC_CMD_COMM_INIT` /
`RPC_CMD_COMM_ALLREDUCE` /
`RPC_CMD_COMM_FREE`, so an all-CUDA-plus-RPC set no longer has to fall back to
the generic butterfly over 1-D copies. `ggml_backend_cuda_comm_init` itself is
untouched. The workaround is documentation only (`:layer` across hosts), so
there is nothing to delete; the tp=2 verdict in [DGX Spark](dgx-spark.md) is
now a claim about an older build and has to be re-measured with
`mix run bench/spark_tensor_split.exs remote` on Spark hardware before it is
changed. This bump was done on an M1 Max, so that measurement is still owed.
`RPC_PROTO_MAJOR_VERSION` went 7 → 8 with it: a worker and a client must come
from the same build.

Re-checked at `88dcc460d` (b11479+4), covering the gap from `c479922ac`: #1,
#2 and #3 are as above — none of the three files changed in the 25 commits.
This range also **fixes a fourth defect found at the previous pin**: the Metal
`MUL_MAT+ADD` fusion picked the residual as "the ADD operand that is not a
MUL_MAT", which is both operands when the residual is itself a mat-mul — the
clef head's `options = proj_option_context @ ctx + proj_option_lexical @ lex`
— so the fused kernel added a never-written buffer and Clef Flash routed
`billing` 0.28 where the CPU gave 0.977. Found by `test/decision_clef_test.exs`
(loaded with `n_gpu_layers: -1` for exactly this reason), bisected with
`GGML_METAL_FUSION_DISABLE=1` then one fusion-table row at a time, fixed
upstream in #30100 (`7e8324f5f`, with a `mm2+mm` mode in `test-backend-ops`
that fails 27 of 28 cases without it). Nothing to remove here: the workaround
was a README note, now gone.

Re-checked at `23b0202a1` (b11552), covering the 69 commits from `88dcc460d`
in one source diff: #1, #2 and #3 are as above. `ggml-rpc.cpp` and
`ggml-rpc.h` were not touched (`RPC_PROTO_MAJOR_VERSION` stays 8). The
`ggml-cpu/CMakeLists.txt` diff is #30297, which maps the s390x z17
cross-compile target to `-march=arch15`; the `-mcpu=native` probe is
untouched. The `ggml-cuda.cu` diff is GDN cache fusion and `supports_op`
changes, none of it near `ggml_backend_cuda_comm_init`.

| # | Upstream defect | Our workaround | Still needed? |
|---|---|---|---|
| 1 | `GGML_NATIVE=ON` makes ggml's `-mcpu=native` probe resolve to **base ARMv8-A** on Cortex-X925/A725 with GCC 13.3 — silently, with a soft CMake warning and exit 0. Costs every `sdot`/`smmla`/SVE kernel. | `LLAMA_CPU_ARM_ARCH` + `LLAMA_CUDA_ARCH` in the `Makefile`, which must be set together. See [DGX Spark](dgx-spark.md) and [Cross-Platform Builds](cross-platform-builds.md). | `scripts/spark/verify-build-flags.sh` on an aarch64 host. If a default build (no `LLAMA_CPU_ARM_ARCH`) now reports non-zero `sdot`/`smmla`, upstream fixed the probe. |
| 2 | `-sm tensor` with a non-CUDA device in the set **runs and is correct but ~2.7× slower on decode**: `ggml_backend_cuda_comm_init` returns `nullptr` on any non-CUDA member, so the generic meta-backend butterfly runs instead, and the RPC backend's `NULL` 2-D tensor hooks degrade it to a loop of 1-D transfers. | Documented, not coded around: `Model.load/2` maps `:tensor` to its upstream value and the docs say to use `:layer` across hosts. See the tp=2 verdict in [DGX Spark](dgx-spark.md). | `mix run bench/spark_tensor_split.exs remote`. If `:tensor` comes within range of `:layer`, upstream implemented the 2-D hooks or the all-reduce — update the verdict section. |
| 3 | `ggml_backend_rpc_start_server` returns `void`, never returns on success, and prints failures to stderr, so an **embedded** caller cannot tell "listening" from "port in use". | `rpc_start_server` in `llama_nif.cpp` pre-`bind()`s the endpoint for a real `errno`, then polls `connect()` until something accepts. A TOCTOU window and one wasted connection per start. | Check whether the signature gained a return value or a listening callback. If so, delete `rpc_preflight_bind` and `rpc_wait_until_listening` and drop the poll. |

Not a defect and not going away: `RPC_STATUS_ASSERT` is `GGML_ABORT`
(`ggml-rpc.cpp:30`), so a peer failure terminates the client process — the BEAM
included. That is upstream's deliberate design. `LlamaCppEx.RPC` documents it;
see also the `:row` split mode, which throws on CUDA at this version and which we
deliberately do not work around.

## 3. Build and test

```bash
# Setting LLAMA_BACKEND forces a source build, so no version bump is needed to
# stop the precompiler downloading the old binary. The build stamp is keyed on
# the llama.cpp commit, so the bump from step 1 already forces a rebuild.
LLAMA_BACKEND=cpu mix compile

# Run the suite. The default run needs no model; each opt-in tag names the env
# var for the model it loads (see test/test_helper.exs). GGML_METAL_NO_RESIDENCY
# is Metal-only, and only stops a post-suite assert in llama.cpp's Metal device
# destructor from aborting the VM after a green run.
mix test

GGML_METAL_NO_RESIDENCY=1 \
LLAMA_SMOKE_GEN_MODEL=~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
LLAMA_SMOKE_EMB_MODEL=~/Downloads/Qwen3-Embedding-0.6B-f16.gguf \
LLAMA_SMOKE_MTP_MODEL=~/Downloads/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  mix test --include smoke --include embeddings --include slow --include mtp

# A second embedding model worth running: embeddinggemma-2 has n_embd_out
# (768) != n_embd (512) and defaults to mean pooling, which together caught
# two bugs in v0.8.56 that Qwen3-Embedding (last pooling, equal widths) cannot.
GGML_METAL_NO_RESIDENCY=1 \
LLAMA_SMOKE_EMB_MODEL=~/Downloads/embeddinggemma-2-Q8_0.gguf \
  mix test --include embeddings

# The decision tags: the tiny laya/openjev pair that CI uses, and the real
# Clef Flash (9B) behind its own tag because no tiny clef exists.
GGML_METAL_NO_RESIDENCY=1 \
LLAMA_SMOKE_DECISION_LAYA_MODEL=~/Downloads/tinylaya-for-testing-Q8_0.gguf \
LLAMA_SMOKE_DECISION_OPENJEV_MODEL=~/Downloads/tinyopenjev-for-testing-Q8_0.gguf \
  mix test --include decision

GGML_METAL_NO_RESIDENCY=1 \
LLAMA_SMOKE_DECISION_CLEF_MODEL=~/Downloads/Cloudflare_clef-flash-Q8_0.gguf \
  mix test --include decision_clef

# lfm2-d1 and lfm2-d1-omni have no tag: check them with
# scripts/decision_compare.exs (below) on d1-3B and d1-omni-600M. The d1-3B
# quants published before upstream's rename say "d1" and are refused by both
# sides; LiquidAI/d1-omni-600M-GGUF already says "lfm2-d1-omni".

GGML_METAL_NO_RESIDENCY=1 \
LLAMA_SMOKE_MTP_MODEL=~/Downloads/Qwen3.8-27B-Q4_K_M.gguf \
LLAMA_SMOKE_MTP_DRAFT_MODEL=~/Downloads/mtp-Qwen3.8-27B-Q4_0.gguf \
  mix test --include mtp_sidecar

# The Qwen pair is required: MTPSidecarTest has no skip and calls path!/1.
# Add the E4B pair to also run MTPE4BSidecarTest. Unset E4B vars skip that
# module, so the Qwen-only command above stays green.
GGML_METAL_NO_RESIDENCY=1 \
LLAMA_SMOKE_MTP_MODEL=~/Downloads/Qwen3.8-27B-Q4_K_M.gguf \
LLAMA_SMOKE_MTP_DRAFT_MODEL=~/Downloads/mtp-Qwen3.8-27B-Q4_0.gguf \
LLAMA_SMOKE_MTP_E4B_MODEL=~/Downloads/gemma-4-E4B-it-Q4_K_M.gguf \
LLAMA_SMOKE_MTP_E4B_DRAFT_MODEL=~/Downloads/mtp-gemma-4-E4B-it-Q8_0.gguf \
  mix test --include mtp_sidecar

# :rpc_live needs an RPC build AND a reachable worker, and must run with no
# model tag beside it — see test/test_helper.exs for why combining them aborts.
# The worker can be local: another BEAM running LlamaCppEx.RPC.Server, or
# `LLAMA_RPC=1 make rpc-server` and upstream's ggml-rpc-server binary.
LLAMA_RPC=1 LLAMA_BACKEND=metal MIX_ENV=test mix run --no-halt -e \
  'LlamaCppEx.RPC.Server.start_link(endpoint: "127.0.0.1:50052")' &
GGML_METAL_NO_RESIDENCY=1 LLAMA_RPC=1 LLAMA_RPC_ENDPOINT=127.0.0.1:50052 \
  mix test --include rpc_live

# :mtp_cancel is the one tag that is not expected to pass; run it to confirm
# how it fails, and update test/mtp_model_test.exs if the failure mode moved.
GGML_METAL_NO_RESIDENCY=1 \
LLAMA_SMOKE_MTP_MODEL=~/Downloads/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
  mix test --only mtp_cancel

# Verify formatting and types
mix format --check-formatted
mix dialyzer
```

Then check that a Hex **source** build still works, which is the path every
`LLAMA_BACKEND` user and every unlisted target takes. It exercises the Makefile's
llama.cpp clone, so it catches a `LLAMA_COMMIT` that drifted from the submodule:

```bash
mix hex.build
d=$(mktemp -d) && tar xf llama_cpp_ex-*.tar -C "$d" && tar xzf "$d"/contents.tar.gz -C "$d"
(cd "$d" && mix deps.get && LLAMA_BACKEND=cpu mix compile)
git -C "$d"/vendor/llama.cpp rev-parse HEAD   # must equal the submodule SHA
```

## 4. Update version and changelog

1. **`mix.exs`**: bump `@version` on `LlamaCppEx.MixProject` (e.g. `"0.8.42"` → `"0.8.43"`)
2. **`CHANGELOG.md`**: add a new `## vX.Y.Z` section at the top with:
   - The submodule commit range and count
   - Notable changes categorized by subsystem (follow existing format)

To list commits for the changelog:

```bash
git -C vendor/llama.cpp log --oneline <old-commit>..<new-commit>
```

## 5. Commit

```bash
git add vendor/llama.cpp mix.exs CHANGELOG.md
git commit -m "Bump llama.cpp to <short-hash>, release vX.Y.Z"
```

## 6. Tag and push

```bash
git tag vX.Y.Z
git push origin master
git push origin vX.Y.Z
```

The tag push triggers the **precompile workflow**
(`.github/workflows/precompile.yml`), which does everything including the Hex
publish. The jobs run in this order:

1. **`prepare_release`** creates the GitHub Release as a *draft*, so nothing is
   visible while assets are still arriving.
2. **`precompile`** (4 legs: macOS/Metal and Linux/CPU × OTP 27 and OTP 29)
   builds each NIF with `LLAMA_PORTABLE=1` and uploads its `.tar.gz` into the
   draft. Only the tarballs are uploaded — the `.sha256` sidecars stay on the
   runner so the next job hashes the bytes it actually downloads.
3. **`checksum`** verifies every artifact `mix.exs` declares is present, flips the
   release out of draft, runs `mix elixir_make.checksum --all`, verifies the
   resulting `checksum.exs` has an entry for each of them, and commits it to
   `master`.
4. **`publish`** checks out the **tag** (not `master`), takes only `checksum.exs`
   from `master`, compiles once to verify the published artifact against those
   checksums, and runs `mix hex.publish --yes`.

So there is nothing to do by hand after the tag push. Watch the run; if a leg
fails, the release stays a draft and nothing reaches Hex.

If you ever need to publish manually — a workflow outage, say — reproduce what
`publish` does rather than publishing from `master`:

```bash
git checkout vX.Y.Z
git fetch origin master
git checkout origin/master -- checksum.exs
mix hex.publish
```

## Troubleshooting

### Compilation errors after upgrade

- **Missing function**: check if the API was renamed or removed in `include/llama.h`
- **Struct field changes**: check `llama_model_params`, `llama_context_params`, `llama_batch` structs
- **Common library changes**: `common/chat.h` is the most volatile dependency — check `common_chat_templates_inputs` and `common_chat_msg`

### Build downloads precompiled binary instead of compiling from source

Set `LLAMA_BACKEND` (to `cpu` if you do not care which). Any value flips
`make_force_build` in `mix.exs` and skips the download entirely. Bumping
`@version` also works, but only because no artifact exists for the new version
yet.

### CI precompile fails

Check `.github/workflows/precompile.yml`. Common issues:

- New llama.cpp dependencies not available in CI runners
- CMake flag changes requiring updates to the `Makefile`
- **The tag is not strict semver.** Every job re-derives the version from
  `GITHUB_REF` and refuses anything that is not `X.Y.Z[-pre][+build]`, because
  that value is interpolated into a `sed` script. `vX.Y.Z-rc1` is fine,
  `v1.2` and `vlatest` are not.
- **A matrix leg failed.** The release then stays a draft and nothing is
  published to Hex. Fix the leg and re-run the workflow; `prepare_release`
  reuses the existing draft and the uploads use `--clobber`.
- **`checksum.exs` came back incomplete.** `mix elixir_make.checksum` prints an
  error but still exits 0 when an artifact download fails, so the workflow
  re-checks the file against the artifact list derived from `mix.exs` and fails
  the release itself. Re-running is usually enough.
