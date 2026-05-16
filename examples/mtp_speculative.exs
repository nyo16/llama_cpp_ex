# Demonstrates Multi-Token Prediction (MTP) speculative decoding.
#
# Requires a GGUF that ships MTP head layers, e.g.
#   https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF (Q4_K_M ≈ 21 GB)
#
# Usage:
#   LLAMA_MTP_MODEL_PATH=~/Downloads/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf \
#     mix run examples/mtp_speculative.exs
#
# Optional flags via env vars:
#   N_DRAFT=3        # drafts proposed per iteration (2–4 typical)
#   N_CTX=8192       # context size
#   MAX_TOKENS=512   # max tokens to generate

model_path =
  System.get_env("LLAMA_MTP_MODEL_PATH") ||
    raise """
    Set LLAMA_MTP_MODEL_PATH to an MTP-enabled .gguf file
    (e.g. from ggml-org/Qwen3.6-35B-A3B-MTP-GGUF on HuggingFace).
    """

n_draft = String.to_integer(System.get_env("N_DRAFT") || "3")
n_ctx = String.to_integer(System.get_env("N_CTX") || "8192")
max_tokens = String.to_integer(System.get_env("MAX_TOKENS") || "512")

prompt =
  System.get_env("PROMPT") ||
    "Write a small but complete implementation of a hash table in C99, with " <>
      "comments explaining the design choices:"

:ok = LlamaCppEx.init()

IO.puts("Loading model: #{Path.basename(model_path)}")
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: 999)

IO.puts(
  "Building MTP session (n_draft=#{n_draft}, n_ctx=#{n_ctx}). " <>
    "This builds two contexts on the same GGUF — main target + MTP draft."
)

{:ok, mtp} =
  LlamaCppEx.MTP.init(model,
    n_draft: n_draft,
    n_ctx: n_ctx,
    flash_attn: :enabled
  )

IO.puts("MTP rollback capacity: #{LlamaCppEx.Context.n_rs_seq(mtp.mtp_ctx)} tokens")
IO.puts("\n--- prompt ---\n#{prompt}\n--- output ---")

mtp
|> LlamaCppEx.MTP.stream(prompt, max_tokens: max_tokens, temp: 0.7, top_p: 0.95)
|> Stream.each(&IO.write/1)
|> Stream.run()

IO.puts("\n--- stats ---")

stats = LlamaCppEx.MTP.stats(mtp)

%{
  iters: iters,
  drafts_generated: dgen,
  drafts_accepted: dacc,
  tokens_emitted: emitted,
  acceptance_rate: acc,
  tokens_per_sec: tps,
  timing_us: timings
} = stats

IO.puts("speculative iterations: #{iters}")
IO.puts("drafts generated:       #{dgen}")
IO.puts("drafts accepted:        #{dacc}")
IO.puts("tokens emitted:         #{emitted}")
IO.puts("acceptance rate:        #{Float.round(acc * 100, 1)}%")
IO.puts("throughput:             #{Float.round(tps, 1)} tok/s")

IO.puts(
  "timings (ms):           " <>
    "draft=#{Float.round(timings.draft / 1000, 1)}  " <>
    "verify=#{Float.round(timings.verify / 1000, 1)}  " <>
    "sample=#{Float.round(timings.sample / 1000, 1)}  " <>
    "total=#{Float.round(timings.total / 1000, 1)}"
)
