# Release Guide

How to upgrade the llama.cpp submodule and publish a new release.

## Prerequisites

- Elixir 1.18+, Erlang/OTP 27+
- cmake
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

## 2. Check API compatibility

Before building, verify the llama.cpp APIs used by the NIF haven't changed:

```bash
# Diff the public header between old and new commits
git -C vendor/llama.cpp diff <old-commit>..<new-commit> -- include/llama.h

# Diff common headers used by the NIF
git -C vendor/llama.cpp diff <old-commit>..<new-commit> -- common/chat.h
git -C vendor/llama.cpp diff <old-commit>..<new-commit> -- common/json-schema-to-grammar.h
```

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

If any signatures changed, update `c_src/llama_cpp_ex/llama_nif.cpp` and/or `llama_nif.h`.

## 3. Build and test

```bash
# Bump version first (so it builds from source instead of downloading precompiled)
# Edit mix.exs @version

# Clean build
mix clean && mix compile

# Run full test suite
LLAMA_MODEL_PATH=~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \
LLAMA_EMBEDDING_MODEL_PATH=~/Downloads/Qwen3-Embedding-0.6B-f16.gguf \
mix test

# Verify formatting and types
mix format --check-formatted
mix dialyzer
```

## 4. Update version and changelog

1. **`mix.exs`** line 40: bump `@version` (e.g. `"0.6.5"` → `"0.6.6"`)
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

The tag push triggers the **precompile workflow** (`.github/workflows/precompile.yml`) which:
1. Builds precompiled NIFs for macOS (Metal) and Linux (CPU) across OTP 27 and 28
2. Uploads `.tar.gz` artifacts to the GitHub release
3. Runs `mix elixir_make.checksum --all --ignore-unavailable` and auto-commits `checksum.exs` to master

## 7. Publish to Hex

After the CI checksum commit lands:

```bash
git pull origin master   # get the updated checksum.exs
mix hex.publish
```

## Troubleshooting

### Compilation errors after upgrade

- **Missing function**: check if the API was renamed or removed in `include/llama.h`
- **Struct field changes**: check `llama_model_params`, `llama_context_params`, `llama_batch` structs
- **Common library changes**: `common/chat.h` is the most volatile dependency — check `common_chat_templates_inputs` and `common_chat_msg`

### Build downloads precompiled binary instead of compiling from source

The precompiler fetches binaries by version. If you haven't bumped the version, it'll find and use the old binary. Bump `@version` in `mix.exs` before running `mix compile`.

### CI precompile fails

Check `.github/workflows/precompile.yml` for the build matrix. Common issues:
- New llama.cpp dependencies not available in CI runners
- CMake flag changes requiring updates to the `Makefile`
