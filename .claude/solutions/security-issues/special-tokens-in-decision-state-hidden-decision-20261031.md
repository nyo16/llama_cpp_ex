---
module: "LlamaCppEx.Decision"
date: "2026-10-31"
problem_type: security_issue
component: nif_tokenization
symptoms:
  - "Clef Flash asked whether untrusted_content contains <|im_start|>/<|im_end|> answered noul 0.03 for raw markers but 0.80 when they were escaped by hand"
  - "usage.input_tokens for a state with markers was lower than for the same state tokenized literally (177 vs 191 on Clef Flash)"
  - "<think> and <tool_call> stayed single tokens even with parse_special: false on the Qwen/Clef vocabulary"
root_cause: "the decision port renders the systemone template with the caller's state, instructions and descriptions embedded and tokenizes the joined prompt with parse_special = true; upstream escapes only <|name|> and only on kev/lfm2-d1-omni/lfm3-d1/h2o-lightning, and user-defined tokens match even without special parsing"
severity: high
tags: [special-tokens, decision-model, tokenizer, user-defined-tokens, guard-model, nif, port]
related_solutions:
  - ".claude/solutions/security-issues/special-tokens-in-chat-content-forge-turns-chat-20261031.md"
---

# Decision state special-token text became control tokens

## Symptoms

GitHub issue #109. `LlamaCppEx.Decision.decide/3` on Clef Flash, question
"Does untrusted_content contain chat-template control markers?":

| state | input_tokens | noul |
|---|---|---|
| raw markers (before) | 177 | 0.030 |
| hand-escaped `<¦im_start¦>` | 191 | 0.784 |
| raw markers (after fix) | 198 | 0.981 |

A guard classifier could not see exactly the markers it was asked about.

## Investigation

1. **Tokenize caller segments with `parse_special = false`** — insufficient:
   on Qwen-family vocabularies `<think>`, `<tool_call>` are
   `LLAMA_TOKEN_ATTR_USER_DEFINED`, and `tokenizer_st_partition` only skips
   CONTROL/UNKNOWN when `parse_special` is false; user-defined tokens still
   match (verified: `<think>` 1 token both ways on tinyopenjev).
2. **Re-split the joined prompt by template pieces** — the clef and
   lfm2-d1-omni builders split on `<<clef:sep>>` / `<<d1omni:sep>>` *after*
   rendering; caller text containing a separator would shift pieces. Pieces
   must be computed from spans, ignoring separators inside caller spans.
3. **Template input marking** — states that are maps are rendered with
   `| tojson`, which drops jinja's `is_input`. Placeholders are required.
4. **Root cause found**: joined-prompt tokenization with special parsing,
   plus user-defined tokens matching regardless.

## Root Cause

Same as the chat issue: once the prompt is a single string the
template/caller boundary is gone. The decision port inherited upstream's
`common_tokenize(vocab, prompt, false, true)` on every path (generic, d1-omni
pieces, clef pieces, `h2o_cut_state`).

## Solution

- Shared `caller_text_guard` + `tokenize_pieces` (see the chat solution) with
  `input_policy::strict`: control *and* non-whitespace user-defined token
  texts in caller spans are cut after their first code point.
- `protect_request` guards every caller string (state, instructions, option
  keys/descriptions, nested questions), leaving `type`, `label`, and the
  port's own marker fields untouched.
- `rendered_prompt` keeps caller spans; `split(sep)` ignores separators inside
  caller spans; `tokenize_prompt(prompt, begin, end)` replaces
  `common_tokenize(..., true)` on all paths; `tokenize_caller` for
  `h2o_cut_state`.

### Files Changed

- `c_src/llama_cpp_ex/decision.cpp` — guard, `rendered_prompt`, tokenization
- `c_src/llama_cpp_ex/input_tokenize.{h,cpp}` — strict policy, `blank` flag
- `test/decision_test.exs`, `test/decision_clef_test.exs`
- `scripts/decision_compare.exs` — "special text" request marked by-design

## Prevention

- Parity check that mattered: run every decision model before/after with
  states *without* special text — answers must be bit-identical (they were,
  18/18 lines), and `scripts/decision_compare.exs` vs llama-server must stay
  ALL MATCH on everything except the by-design "special text" request.
- Test pattern: (markup tokens − plain tokens) == (literal markup − literal
  plain), per field (state, instructions, description), on tinylaya
  (`[CLS]`/`[SEP]` control) and tinyopenjev (`<|im_start|>` control,
  `<think>` user-defined).
- On a llama.cpp bump, any upstream hunk touching
  `common_tokenize(vocab, prompt, false, true)` in the port must be ported
  through `tokenize_prompt` (docs/release-guide.md).

## Related

- `.claude/solutions/security-issues/special-tokens-in-chat-content-forge-turns-chat-20261031.md`
