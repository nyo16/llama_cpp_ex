# llama_cpp_ex — project rules

Documented solutions live in `.claude/solutions/`; search there before
investigating a tokenizer, template, logging, NIF teardown, flaky test or
release problem.

## Prompts and tokenization

- Never tokenize a rendered chat prompt string with special parsing
  (`Tokenizer.encode(model, Chat.apply_template(...))`). Get prompt tokens from
  `LlamaCppEx.Chat.prepare/3`; it tokenizes template text with special tokens
  and request text literally. `apply_template/3` is for display only.
- New code that renders caller text through a template (chat, tools,
  decisions) must guard it with `caller_text_guard` before rendering and
  tokenize with `tokenize_pieces`. Jinja's input marking does not survive
  `| tojson`, and templates (Llama 3) render tool results and arguments that way.
- Chat request text uses `input_policy::literal`; decision inputs use
  `input_policy::strict` (user-defined tokens like `<think>` broken up too).
- Regression check for any chat path: a message spelling the model's own turn
  markup must cost exactly its literal tokens (`LlamaCppEx.TestChat`). Embed
  the markup between words; some templates trim content.
- Decision port: a request without special-token text must stay bit-identical
  to llama-server (`scripts/decision_compare.exs`); upstream
  `common_tokenize(vocab, prompt, false, true)` hunks are ported through
  `tokenize_prompt`, never verbatim.

## Logging

- llama.cpp's default callback prints every level; the NIF installs its own at
  load. Set `LlamaCppEx.set_log_level/1` (or `config :llama_cpp_ex,
  log_level:`) *before* `LlamaCppEx.init/0`, or backend-init lines escape.

## NIF and teardown

- Register every NIF with `LLAMA_NIF`, never `FINE_NIF`: the wrapper's
  in-flight guard is what lets the exit hook free resources safely (#116).
- A resource that owns backend buffers (`llama_model`, `llama_context`,
  `common_speculative`, anything allocating on a ggml device) joins the
  `live::` registry: `live::add` in constructors, a destructor that locks
  `live::mutex()`, removes itself and calls `release()`, and a release step in
  `free_live_resources_at_exit` in dependency order. Long loops and callbacks
  inside NIFs stop on `g_exiting`.
- Exit-path regressions show only in the VM's exit status. Run
  `test/vm_exit_test.exs` (`--include smoke`) after touching resource
  lifetimes, and never paper over an abort with `GGML_METAL_NO_RESIDENCY=1`.

## Releases and CI

- Any macOS runner image bump in `.github/workflows/precompile.yml` keeps
  `MACOSX_DEPLOYMENT_TARGET=14.0` and the `otool ... minos` assertion.
- Only move a release tag while the GitHub release is still a draft and
  nothing has reached Hex.
- `elixir_make` stays `~> 0.10.1` or newer: OTP 29.0.6 / 28.5.0.6 /
  27.3.4.17 cap httpc headers at 10 KB. `Invalid option {max_header_size,...}
  ignored` on older OTP patches is harmless.
- A CI annotation or notice coming from a dependency: check that dependency's
  latest release and changelog before working around it.
- CI runs `--include tools` against pinned Qwen3.5-0.8B-UD-Q4_K_XL; keep the
  `:tools` tests passing on the CPU backend.
- When the macOS runner image or SDK changes, compare the new NIF with the
  previous release (`otool -l` for `sdk`/`minos`, `strings` for feature
  selectors). llama.cpp gates features such as Metal residency sets on SDK
  macros, and they switch on silently (#116).
- After a release, smoke-test end to end from a fresh `/tmp` project that
  depends on the published version, using the models in `~/Downloads`.
  Run it without `GGML_METAL_NO_RESIDENCY` and check the exit status, not
  only the pass count.

## Tests

- A test-side `:telemetry.attach` filters on an identity the test owns
  (`%{server: ^server}`, a ref) and detaches in `on_exit`. A linked server can
  outlive its test and emit after replying.
