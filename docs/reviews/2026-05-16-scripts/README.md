# Code review experiment scripts (2026-05-16)

Throwaway scripts used to confirm/deny the findings in
`../2026-05-16-code-review.md`. Kept here for reproducibility.

## How they were run

All scripts originally lived under `/tmp/llama_review/` and reference
`/tmp/llama_review/_common.exs` by absolute path (see the `Code.require_file`
call at the top of each `case_*.exs`). To re-run from this directory:

```sh
# From the repo root
mkdir -p /tmp/llama_review
cp docs/reviews/2026-05-16-scripts/*.{exs,sh} /tmp/llama_review/
MIX_ENV=dev mix run --no-start /tmp/llama_review/exp1_sampler_stall.exs
```

Or update the `Code.require_file` paths inline before running.

## What's here

| Script | Purpose | Finding it confirms |
|--------|---------|---------------------|
| `exp1_sampler_stall.exs` | BEAM scheduler stall from non-dirty `sampler_sample` | H1 |
| `exp2_caller_death.exs` | Server slot leak when stream consumer dies | C3 |
| `exp3_load_unload.exs` | Model load/unload RSS growth | (no leak — clean) |
| `exp5_edge_cases.exs` | First-pass edge-case sweep | (superseded by `case_*.exs`) |
| `exp5_runner.sh` | Driver for the per-case scripts | runs all `case_*.exs` |
| `case_*.exs` | One edge case per file, run in its own BEAM | C1, M2, M3, M4, M5, M1 |
| `exp6_concurrency.exs` | Server concurrency stress | (clean — 16/16) |
| `exp7_mtp_unit.exs` | MTP determinism + stats + reuse safety | C2 |
| `_common.exs` | Shared model-loader helper for the `case_*` scripts | — |

## Model paths

All scripts default to `~/Downloads/<model>.gguf` and accept env-var overrides:

- `MODEL_PATH` — defaults to `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` (fast iteration)
- `LLAMA_MTP_MODEL_PATH` — defaults to `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf`
