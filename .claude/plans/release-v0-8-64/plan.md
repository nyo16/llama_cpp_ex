# Plan: Release v0.8.64 (no llama.cpp bump)

**Depth**: quick · **Status**: completed · **Date**: 2026-10-31 → 2026-11-01

## Goal

Publish v0.8.64 to Hex with precompiled NIFs for all 8 targets. It ships the
changes merged after v0.8.63 (CHANGELOG "Unreleased"), with llama.cpp staying
at `418f905a4`:

- `LlamaCppEx.set_log_level/1`, `config :llama_cpp_ex, log_level:`, DEBUG
  filtered by default (#114)
- `:tools` tag in CI against pinned Qwen3.5-0.8B (#114)
- `elixir_make ~> 0.10.1` (#114); req `~> 0.7` allowed, dep lock updates (#113)
- lock: decimal 3.2.0 / ecto 3.14.2 (#114)

Done means: release not a draft, 8 assets, `checksum.exs` on master, Hex
0.8.64, and the `/tmp` end-to-end smoke passing against the published
package with the `~/Downloads` models.

## Decisions

- **No llama.cpp bump** (user). There are 22 upstream commits on master;
  they belong in a separate bump release. So: no header diff, no
  decision-port diff, no tokenizer-partition re-diff, no
  `decision_compare.exs` rerun. No decision code changed since v0.8.63, which
  was compared ALL MATCH.
- **Merge #115 first** (user). It is docs only and `.claude/` and `CLAUDE.md`
  are not in the Hex `files` list, so the package is unaffected.
- **Release via PR then tag**, as for v0.8.60–v0.8.63; the version bump and
  changelog go through CI before the tag.

## Phase 1 — Prepare master

- [x] [git] Merge #115 (squash, delete branch); `git pull` master — first
      Inference smoke failed on a pre-existing test race (next item); re-run
      green; merged as 3b63b52
- [x] [test] Fix flaky `ServerSmokeTest` "halting a stream early …":
      `attach_collector/1` was a global handler; it caught the previous
      test's lingering server's `:done` (39 prompt tokens, `max_tokens`),
      which `finish_slot/3` emits after replying. Collectors in
      `server_smoke_test.exs` and the telemetry test in `llama_cpp_ex_test.exs`
      now pin `metadata.server`; test-only, no CHANGELOG entry
- [x] [git] Branch `release-v0.8.64` from master (3b63b52)
- [x] [config] `mix.exs` `@version "0.8.63"` → `"0.8.64"`
- [x] [docs] CHANGELOG: rename `## Unreleased` → `## v0.8.64`, add an
      intro line ("llama.cpp stays at `418f905a4`"), keep the existing
      Added/Changed entries; #115 is docs only and gets no entry
- [x] [verify] `mix compile --warnings-as-errors`, `mix format --check-formatted`,
      `mix credo --strict`, `mix dialyzer` (0 errors) — all clean

## Phase 2 — Local test matrix (Metal, `GGML_METAL_NO_RESIDENCY=1`)

Unset `LLAMA_BACKEND` only where a precompiled path is under test; here
everything is a local build (`LLAMA_BACKEND=metal`).

- [x] [test] default: `mix test` — 436 passed, 194 excluded
- [x] [test] CI's set: `--include smoke --include embeddings --include slow
      --include decision --include tools` with SmolLM2, bge-small, tinylaya,
      tinyopenjev, Qwen3.5-0.8B — 601 passed, 29 excluded
- [x] [test] release-guide set: `--include smoke --include embeddings
      --include slow --include mtp` with Qwen3.5-0.8B, Qwen3-Embedding-0.6B-f16,
      Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL — 585 passed, 45 excluded
- [x] [test] `--include embeddings` with embeddinggemma-2-Q8_0 — 447 passed
- [x] [test] `--include decision_clef` with Cloudflare_clef-flash-Q8_0 — 440
      passed (4/4 clef)
- [x] [test] `--include tools` with Llama-3.2-1B-Instruct (expect 10/11:
      the round trip calls the tool again; documented, not a regression) —
      446/447, the round-trip test as expected; Qwen3.5 on Metal 447
- [x] [test] `--include mtp_sidecar` with Qwen3.8-27B-Q4_K_M +
      mtp-Qwen3.8-27B-Q4_0 (E4B pair not on disk, so that module skips) —
      442 passed, 6 skipped
- [x] [test] LLAMA_BACKEND=cpu `--include tools` (what CI runs) — 447 passed
- [x] [verify] Hex source build: `mix hex.build`, unpack into a temp dir,
      `LLAMA_BACKEND=cpu mix compile`, confirm `vendor/llama.cpp` HEAD ==
      `418f905a4` and `input_tokenize.{h,cpp}` are in the tarball — 332K
      package, build fetched `418f905a4`, NIF built
- [x] [docs] CHANGELOG v0.8.64: add the "Verified on …" paragraph with the
      exact counts from this phase (format as in v0.8.63)

## Phase 3 — PR and CI

- [x] [git] Commit "Release v0.8.64", push, open PR — #117 (2 commits:
      e081cbf test fix, 008f7bf release)
- [x] [ci] Watch CI: Setup, Tests, Credo, Dialyzer, Formatting, Inference
      smoke (now including `:tools`), CUDA link cu12/cu13 — all green
      (run 41250391316)
- [x] [ci] Confirm in the Inference smoke log: `Including tags: [... :tools]`
      and 0 failures; no `xref: [exclude` annotation — 630 tests, 0 failures;
      no xref annotations
- [x] [git] Merge PR (ask the user before merging), pull master — user chose
      squash; 4f47d41

## Phase 4 — Tag and publish

- [x] [release] `git tag v0.8.64` on the merge commit; `git push origin v0.8.64`
- [x] [release] Watch `Precompile NIFs`: 2× macos-15 Metal (log shows
      `minos: 14.0`), 2× CPU, 4× CUDA (cu12/cu13 legs take ~50–110 min),
      then "Checksum and publish the release" and "Publish to Hex" — run
      41255669230 all green in 85 min; both Metal legs `minos: 14.0`
- [x] [release] Verify: `gh release view v0.8.64` → not draft, 8 assets;
      master has "Update checksum.exs for v0.8.64"; `mix hex.info llama_cpp_ex`
      lists 0.8.64 — draft=false, 8 assets, 1041099, `~> 0.8.64`
- [x] [git] `git pull` master (picks up the checksum commit)

## Phase 5 — End-to-end smoke from Hex (`/tmp`, `~/Downloads`)

- [x] [e2e] Fresh project: `rm -rf /tmp/llama_e2e_0864 && mix new`, deps
      `{:llama_cpp_ex, "== 0.8.64"}`, `{:req, "~> 0.7"}`
- [x] [e2e] `env -u LLAMA_BACKEND mix compile` → "Downloading precompiled NIF"
      for `nif-2.18-aarch64-apple-darwin-0.8.64`, checksum passes, `otool`
      minos 14.0; `mix.lock` resolves elixir_make 0.10.1 — all as expected
      (httpc `max_header_size` notice, known)
- [x] [e2e] Copy `/tmp/llama_e2e/e2e.exs`; add log-level checks:
      `set_log_level(:none)` before `init/0` → a load + tool call writes 0
      stderr lines; default level → the tool call writes no "awaiting trigger"
      line; invalid level raises `ArgumentError` — child VM via
      `loglevel.exs`, stderr to file
- [x] [e2e] Run with `GGML_METAL_NO_RESIDENCY=1`: all checks PASS (generate,
      stream, chat, json_schema, #108 ×3 incl. server and Llama 3.2, #110 ×2,
      #111 ×6, embeddings, #109 ×3 incl. Clef noul > 0.5, log level ×3) —
      23 passed, 0 failed; clef noul 0.98; :none 0 stderr lines, default 198
- [x] [e2e] Optional: repeat compile on a fresh project with
      `LLAMA_BACKEND=cpu` to confirm the Hex source build from the published
      tarball — built at `418f905`, CPU generate " Paris."

## Verification summary

| Gate | Command / evidence |
|---|---|
| Compile/lint | `mix compile --warnings-as-errors`, format, credo --strict, dialyzer |
| Local tests | Phase 2 matrix, counts recorded in CHANGELOG |
| CI | PR checks all green, Inference smoke ran `:tools` |
| Release | workflow green, release published with 8 assets, Hex 0.8.64 |
| Published package | `/tmp` e2e: precompiled download + all checks PASS |

## Iron Law check

No application code changes in this plan (version, changelog only). Iron Law
#27 (no special-parsed tokenization of rendered prompts) is unaffected; the
e2e #108/#109 checks re-verify it on the published artifact.

## Risks

- **CUDA legs are slow** (47–110 min) and can hit runner capacity; the
  release stays a draft until all 8 legs pass, so nothing partial reaches Hex.
- **macos-15 arm64 capacity**: the workflow annotated queueing delays for
  v0.8.63, so expect waits.
- **Flaky small-model tool tests on CI CPU**: Qwen3.5-0.8B passed 11/11 on CPU
  locally and in CI once. If CI flakes, investigate before re-running;
  don't loosen assertions to get green.
- **Local Hex auth expired** (`mix hex.user auth`): this only matters for a
  manual publish, since the workflow publishes with its own key.
- **Tag hygiene**: move a tag only while the release is still a draft and
  Hex has nothing (see `.claude/solutions/deployment-issues/macos-14-runner-brownout-precompile-20261031.md`).
