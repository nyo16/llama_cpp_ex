#!/bin/bash
# Run each edge case in its own BEAM. Native aborts crash the whole BEAM, so
# we need fresh processes to recover. Each script must exit 0 on clean error,
# nonzero on crash.

cd /Users/nmaroulis/Source/llama_cpp_ex

CASES=(
  "n_ctx_zero"
  "n_ctx_neg"
  "n_ctx_huge"
  "max_tokens_neg"
  "top_k_zero"
  "top_k_one"
  "temp_neg"
  "temp_extreme"
  "top_p_zero"
  "tokenize_empty"
  "tokenize_100kb"
  "decode_empty"
  "decode_negative_token"
  "decode_oov_token"
  "mtp_n_draft_zero"
  "mtp_n_draft_neg"
  "side_effect_demo"
)

for c in "${CASES[@]}"; do
  echo "=== $c ==="
  MIX_ENV=dev mix run --no-start /tmp/llama_review/case_${c}.exs 2>&1 \
    | grep -vE 'graph_reserve|sched_reserve|compute buffer|matches expectation|deallocating|^ggml_|^load_tensors|^llama_model|^print_info|^load:|^common:|^llama_context:|^llama_kv|^llama_memory|^max_buffer|^Model:|^model:|^common_init|^GPU device|fused Gated|^create_tensor|^done_getting|^\.\.\.|^read_tensor|^set_abort_callback|^tensor blk' \
    | tail -20
  EXIT=${PIPESTATUS[0]}
  echo "exit=$EXIT"
  echo
done
