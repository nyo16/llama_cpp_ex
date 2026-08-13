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
grep -hoE '^#include [<"](llama|ggml|chat|json-schema|speculative)[^">]*' \
  c_src/llama_cpp_ex/*.cpp c_src/llama_cpp_ex/*.h | sort -u
```

```bash
for h in include/llama.h \
         ggml/include/ggml-backend.h ggml/include/ggml-rpc.h \
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

### Upstream defects we work around

Three known llama.cpp defects have workarounds in this repo. A bump is the only
time anyone looks at them, so check each one here — if upstream has fixed it, the
workaround should come out rather than quietly accumulate.

Each was measured against `4801e3c567d5` (b10362) on NVIDIA DGX Spark (GB10,
aarch64, GCC 13.3, CUDA 13.0). Full reports, with reproductions and suggested
upstream fixes, are drafted in
`.claude/plans/dgx-spark-2node/upstream-issues.md` — not yet filed, so there are
no issue URLs to link. **When they are filed, put the URLs in this table.**

Re-checked at `a94d563ed801` (61 commits later): all three still stand. That
check was a source diff, not a re-measurement — the files each defect lives in
(`ggml/src/ggml-cpu/CMakeLists.txt`, `ggml_backend_cuda_comm_init`, and
`ggml_backend_rpc_start_server`) were untouched by the bump. A source diff is
enough to say a defect is *still there*; it is not enough to say it is *gone*,
so if a diff ever shows movement, run the command in the last column.

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

# Run full test suite
LLAMA_MODEL_PATH=~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
LLAMA_EMBEDDING_MODEL_PATH=~/Downloads/Qwen3-Embedding-0.6B-f16.gguf \
mix test

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
