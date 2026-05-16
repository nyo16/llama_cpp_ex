# MTP vs non-MTP throughput benchmark across different max_tokens.
#
# Usage:
#   LLAMA_MTP_MODEL_PATH=~/Downloads/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf \
#     mix run examples/mtp_benchmark.exs
#
# Reports tokens/sec for MTP and non-MTP generation across several lengths,
# with the MTP draft acceptance rate at each length. Same prompt, same
# sampler params, model loaded once.

model_path =
  System.get_env("LLAMA_MTP_MODEL_PATH") ||
    raise "Set LLAMA_MTP_MODEL_PATH"

token_counts =
  (System.get_env("TOKEN_COUNTS") || "32,64,128,256")
  |> String.split(",")
  |> Enum.map(&String.to_integer/1)

prompt =
  System.get_env("PROMPT") ||
    "Write a short paragraph explaining why functional programming makes" <>
      " concurrent systems easier to reason about. Be concrete:"

n_draft = String.to_integer(System.get_env("N_DRAFT") || "3")
n_ctx = String.to_integer(System.get_env("N_CTX") || "8192")
seed = String.to_integer(System.get_env("SEED") || "42")

:ok = LlamaCppEx.init()

IO.puts("loading model: #{Path.basename(model_path)}")
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: 999)

run_mtp = fn max_tokens ->
  {:ok, mtp} =
    LlamaCppEx.MTP.init(model, n_draft: n_draft, n_ctx: n_ctx, flash_attn: :enabled)

  t0 = System.monotonic_time(:millisecond)

  text =
    mtp
    |> LlamaCppEx.MTP.stream(prompt,
      max_tokens: max_tokens,
      temp: 0.7,
      top_p: 0.95,
      seed: seed
    )
    |> Enum.to_list()
    |> IO.iodata_to_binary()

  wall_ms = System.monotonic_time(:millisecond) - t0
  stats = LlamaCppEx.MTP.stats(mtp)

  %{
    mode: :mtp,
    max_tokens: max_tokens,
    emitted: stats.tokens_emitted,
    text_len: byte_size(text),
    wall_ms: wall_ms,
    tok_per_sec: stats.tokens_emitted * 1000 / wall_ms,
    drafts_generated: stats.drafts_generated,
    drafts_accepted: stats.drafts_accepted,
    acceptance_rate: stats.acceptance_rate,
    iters: stats.iters,
    text: text
  }
end

run_plain = fn max_tokens ->
  t0 = System.monotonic_time(:millisecond)

  text =
    model
    |> LlamaCppEx.stream(prompt,
      max_tokens: max_tokens,
      temp: 0.7,
      top_p: 0.95,
      seed: seed,
      n_ctx: n_ctx,
      flash_attn: :enabled
    )
    |> Enum.to_list()
    |> IO.iodata_to_binary()

  wall_ms = System.monotonic_time(:millisecond) - t0

  # Token count isn't returned directly; approximate via re-tokenizing the output.
  {:ok, out_tokens} =
    LlamaCppEx.Tokenizer.encode(model, text, add_special: false)

  emitted = length(out_tokens)

  %{
    mode: :plain,
    max_tokens: max_tokens,
    emitted: emitted,
    text_len: byte_size(text),
    wall_ms: wall_ms,
    tok_per_sec: emitted * 1000 / wall_ms,
    text: text
  }
end

# Header
IO.puts("\nprompt: #{String.slice(prompt, 0..60)}...")
IO.puts("n_draft=#{n_draft} n_ctx=#{n_ctx} seed=#{seed}\n")

IO.puts(
  String.pad_trailing("mode", 7) <>
    String.pad_leading("max_tok", 9) <>
    String.pad_leading("emitted", 9) <>
    String.pad_leading("wall_ms", 10) <>
    String.pad_leading("tok/s", 8) <>
    String.pad_leading("accept", 8) <>
    "  (#acc / #gen)"
)

IO.puts(String.duplicate("─", 80))

results =
  Enum.flat_map(token_counts, fn n ->
    plain = run_plain.(n)
    mtp = run_mtp.(n)

    for r <- [plain, mtp] do
      mode_str =
        case r.mode do
          :mtp -> "mtp"
          :plain -> "plain"
        end

      accept_str =
        case Map.get(r, :acceptance_rate) do
          nil -> "       —"
          rate -> String.pad_leading("#{Float.round(rate * 100, 1)}%", 8)
        end

      acc_gen =
        case r do
          %{drafts_accepted: a, drafts_generated: g} -> "  (#{a} / #{g})"
          _ -> ""
        end

      IO.puts(
        String.pad_trailing(mode_str, 7) <>
          String.pad_leading("#{r.max_tokens}", 9) <>
          String.pad_leading("#{r.emitted}", 9) <>
          String.pad_leading("#{r.wall_ms}", 10) <>
          String.pad_leading("#{Float.round(r.tok_per_sec, 1)}", 8) <>
          accept_str <>
          acc_gen
      )

      r
    end
  end)

# Compute pairwise speedup.
IO.puts("\nspeedup (mtp / plain):")

results
|> Enum.chunk_every(2)
|> Enum.each(fn [plain, mtp] ->
  speedup = mtp.tok_per_sec / plain.tok_per_sec

  IO.puts(
    "  max_tokens=#{plain.max_tokens}: " <>
      "#{Float.round(speedup, 2)}x  " <>
      "(plain #{Float.round(plain.tok_per_sec, 1)} tok/s → " <>
      "mtp #{Float.round(mtp.tok_per_sec, 1)} tok/s)"
  )
end)
