---
module: "LlamaCppEx.NIF"
date: "2026-10-31"
problem_type: integration_issue
component: nif_logging
symptoms:
  - "\"Grammar still awaiting trigger after token N (`...`)\" printed to stderr once per generated token during tool calling"
  - "\"Grammar triggered on token ...\" lacks a newline, so the next log line ran onto it (\"...(`<tool_call>`)set_abort_callback: call\")"
  - "a 100-token tool call printed 1,663 stderr lines"
root_cause: "the NIF never installed a log callback, so llama.cpp's llama_log_callback_default (src/llama-impl.cpp) wrote every level, DEBUG included, to stderr; lazy tool-call grammars log at DEBUG per token"
severity: low
tags: [logging, llama-log-set, ggml, stderr, lazy-grammar, tool-calling, nif]
---

# llama.cpp DEBUG logs flooded stderr during tool calling

## Symptoms

With `:tools`, the lazy grammar logs `LLAMA_LOG_DEBUG("Grammar still awaiting
trigger ...")` from `src/llama-grammar.cpp` for every sampled token until the
trigger. 41 such lines in a 100-token reply; 1,663 stderr lines total.

## Investigation

1. **Is there an existing log hook in the NIF?** — `grep llama_log_set|
   ggml_log_set|common_log_` in `c_src` returned nothing.
2. **Default callback** — `llama_log_callback_default` ignores `level` and
   `fputs(text, stderr)`. `llama_log_set` also calls `ggml_log_set`, so one
   install covers llama and ggml (Metal/CUDA device setup).
3. **Common library** (chat templates) has its own logger, gated by
   `common_log_set_verbosity_thold` (DEBUG 5, INFO 3, WARN 2, ERROR 1).
4. **Ordering trap**: setting the level *after* `LlamaCppEx.init()` still
   printed 14 Metal device-init lines; they are emitted during backend init.

## Root Cause

No callback installed → upstream default prints every level. `CONT`-level
messages continue the previous message and must inherit its level, and
`LLAMA_LOG(...)` uses level NONE for model summaries (treat as INFO).

## Solution

```cpp
// c_src/llama_cpp_ex/llama_nif.cpp
std::atomic<int> g_log_min{GGML_LOG_LEVEL_INFO};
thread_local int t_log_last = GGML_LOG_LEVEL_INFO;
void log_callback(ggml_log_level level, const char* text, void*) {
    int effective = level == GGML_LOG_LEVEL_CONT ? t_log_last
                  : (level == GGML_LOG_LEVEL_NONE ? GGML_LOG_LEVEL_INFO : level);
    if (level != GGML_LOG_LEVEL_CONT) t_log_last = effective;
    if (effective >= g_log_min.load(std::memory_order_relaxed)) fputs(text, stderr);
}
const bool log_callback_installed = (llama_log_set(log_callback, nullptr), true); // at dlopen
```

Elixir: `LlamaCppEx.set_log_level(:debug | :info | :warning | :error | :none)`
(sets ggml min level and common threshold), and
`config :llama_cpp_ex, log_level: ...` applied in `Application.start/2`
(only when configured, so app start doesn't touch the NIF otherwise).

Measured: debug 1,663 lines / info (default) 199 / warning & none 0 when set
before `init`.

### Files Changed

- `c_src/llama_cpp_ex/llama_nif.cpp`, `lib/llama_cpp_ex.ex`,
  `lib/llama_cpp_ex/application.ex`, `lib/llama_cpp_ex/nif.ex`
- `test/log_level_test.exs`

## Prevention

- Native libraries with a "default logger prints everything" policy need an
  explicit callback at load time; set the level before backend init.
- Default kept at `:info` so existing output only loses DEBUG lines.
- Shipped in PR #114 (unreleased at time of writing).
