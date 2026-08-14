Code.require_file("helpers.exs", __DIR__)

# What does Multi-Token Prediction buy on a DGX Spark?
#
#   scripts/spark/remote.sh --env MIX_ENV=bench --big-cores spark-1 \
#     mix run bench/spark_mtp.exs \
#       ~/models/unsloth/Qwen3.6-27B-GGUF/main/Qwen3.6-27B-Q4_K_M.gguf \
#       ~/models/unsloth/Qwen3.6-27B-MTP-GGUF/main/Qwen3.6-27B-Q4_K_M.gguf
#
# The two arguments are the same weights, one built without the MTP head and one
# with, which makes this a clean A/B rather than a comparison across quantizations.
#
# MTP should matter *more* here than on a bandwidth-rich part, and the reason is
# the shape of this chip: decode is memory-bandwidth-bound and prefill is
# compute-rich (measured at 3500-4300 t/s prefill against ~40 t/s decode on 8B).
# Speculative decoding converts decode steps into a batched verification pass —
# it spends the abundant resource to save the scarce one. Whether the draft
# acceptance rate is high enough to cash that in is the measurement.
#
# LlamaCppEx.MTP's own docs claim ~2x at ~75% acceptance with n_draft: 3 on
# Qwen 3.6. This is that claim, on this hardware.

alias LlamaCppEx.{Model, MTP}

{plain_path, mtp_path} =
  case System.argv() do
    [a, b | _] -> {a, b}
    _ -> raise "usage: mix run bench/spark_mtp.exs <plain.gguf> <mtp.gguf>"
  end

for p <- [plain_path, mtp_path] do
  File.exists?(p) || raise "missing #{p}"
end

raw_prompt = """
Write a detailed technical explanation of how a memory-bandwidth-bound decode \
loop differs from a compute-bound prefill pass in a transformer, and why that \
distinction changes which optimisations are worth applying.
"""

max_tokens = 256

# The chat template is not optional here. Qwen3.6-35B-A3B emits end-of-generation
# immediately when handed a bare completion prompt — measured: zero tokens, from
# both the plain and the MTP path — while the same prompt inside the template
# generates normally. The 27B tolerates raw completion, which is exactly the kind
# of difference that turns into a mystery if the harness does not template.
templated = fn model ->
  case LlamaCppEx.Chat.apply_template(model, [%{role: "user", content: raw_prompt}]) do
    {:ok, text} -> text
    {:error, _} -> raw_prompt
  end
end

time_ms = fn fun ->
  t0 = System.monotonic_time(:microsecond)
  result = fun.()
  {(System.monotonic_time(:microsecond) - t0) / 1000, result}
end

median = fn values ->
  s = Enum.sort(values)
  n = length(s)

  if rem(n, 2) == 1,
    do: Enum.at(s, div(n, 2)),
    else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
end

# MTP is by far the noisier arm — acceptance varies run to run, so throughput
# does too. A single sample per setting is not enough to separate a real 1.4x
# from a lucky draw, which is exactly the trap the README's GB10 note avoids by
# interleaving n=11. Median of `reps`, and the range is reported alongside so a
# wide spread is visible rather than hidden behind the median.
reps = String.to_integer(System.get_env("SPARK_MTP_REPS") || "5")

# --- Baseline: no MTP head, ordinary decode ----------------------------------

IO.puts("\n=== baseline (no MTP head)\n#{Path.basename(plain_path)}")

# Scoped in a function so the plain model's NIF resource has no live reference
# once the baseline is done. Both models resident at once would put ~20 GB of
# unrelated weights in the way of the MTP arm on a unified-memory part, which is
# exactly the sort of confound that turns a 0.9x into a 1.1x. The explicit
# collection is what actually releases it: `fine` resources are freed by the GC,
# not at end of scope.
measure_baseline = fn ->
  {:ok, plain} = Model.load(plain_path, n_gpu_layers: 99)
  prompt = templated.(plain)

  # Warm the kernels.
  {_, _} = time_ms.(fn -> LlamaCppEx.generate(plain, prompt, max_tokens: 8, temp: 0.0) end)

  samples =
    for _ <- 1..reps do
      {ms, {:ok, text}} =
        time_ms.(fn -> LlamaCppEx.generate(plain, prompt, max_tokens: max_tokens, temp: 0.0) end)

      tokens = LlamaCppEx.Tokenizer.encode(plain, text) |> elem(1) |> length()

      if tokens == 0 do
        raise """
        the baseline generated zero tokens. The model ended the sequence immediately,
        which on a Qwen3.6 instruct checkpoint means the chat template did not apply —
        check LlamaCppEx.Chat.apply_template/3 against #{Path.basename(plain_path)}.
        """
      end

      tokens * 1000 / ms
    end

  samples
end

base_samples = measure_baseline.()
base_tps = median.(base_samples)

:erlang.garbage_collect()
Process.sleep(500)

IO.puts(
  "  #{Float.round(base_tps, 2)} t/s median of #{reps} " <>
    "(#{Float.round(Enum.min(base_samples), 1)}-#{Float.round(Enum.max(base_samples), 1)})"
)

# --- MTP: the same weights plus the head -------------------------------------

IO.puts("\n=== MTP (load_mtp: true)\n#{Path.basename(mtp_path)}")

{:ok, mtp_model} = Model.load(mtp_path, n_gpu_layers: 99, load_mtp: true)
mtp_prompt = templated.(mtp_model)

rows =
  for n_draft <- [1, 2, 3, 4] do
    # A fresh session per setting: MTP sessions hold two long-lived contexts and
    # reusing one across configurations would carry KV state between them.
    case MTP.init(mtp_model, n_draft: n_draft, n_ctx: 4096) do
      {:ok, session} ->
        {_, _} = time_ms.(fn -> MTP.generate(session, mtp_prompt, max_tokens: 8, temp: 0.0) end)

        samples =
          for _ <- 1..reps do
            {ms, result} =
              time_ms.(fn ->
                MTP.generate(session, mtp_prompt, max_tokens: max_tokens, temp: 0.0)
              end)

            case result do
              {:ok, text} ->
                tokens = LlamaCppEx.Tokenizer.encode(mtp_model, text) |> elem(1) |> length()
                {tokens * 1000 / ms, MTP.stats(session)}

              other ->
                {:error, other}
            end
          end

        case Enum.reject(samples, &match?({:error, _}, &1)) do
          [] ->
            IO.puts("  n_draft=#{n_draft}: FAILED")
            %{n_draft: n_draft, error: "generation failed"}

          ok ->
            tps_values = Enum.map(ok, &elem(&1, 0))
            tps = median.(tps_values)
            stats = ok |> List.last() |> elem(1)

            acceptance =
              case stats do
                %{acceptance_rate: r} when is_number(r) -> Float.round(r * 100, 1)
                _ -> nil
              end

            IO.puts(
              "  n_draft=#{n_draft}: #{Float.round(tps, 2)} t/s median of #{length(ok)} " <>
                "(#{Float.round(Enum.min(tps_values), 1)}-#{Float.round(Enum.max(tps_values), 1)})" <>
                if(acceptance, do: ", #{acceptance}% accepted", else: "")
            )

            %{
              n_draft: n_draft,
              tps: tps,
              lo: Enum.min(tps_values),
              hi: Enum.max(tps_values),
              acceptance: acceptance,
              stats: stats
            }
        end

      {:error, reason} ->
        IO.puts("  n_draft=#{n_draft}: init refused: #{inspect(reason)}")
        %{n_draft: n_draft, error: inspect(reason)}
    end
  end

IO.puts(
  "\n| configuration | decode t/s (median of #{reps}) | range | vs baseline | draft acceptance |"
)

IO.puts("|---|---|---|---|---|")

IO.puts(
  "| no MTP head | #{Float.round(base_tps, 2)} | " <>
    "#{Float.round(Enum.min(base_samples), 1)}–#{Float.round(Enum.max(base_samples), 1)} | — | — |"
)

for r <- rows do
  if Map.has_key?(r, :error) do
    IO.puts("| MTP n_draft=#{r.n_draft} | FAILED: #{r.error} | | | |")
  else
    speedup = Float.round(r.tps / base_tps, 2)

    IO.puts(
      "| MTP n_draft=#{r.n_draft} | #{Float.round(r.tps, 2)} | " <>
        "#{Float.round(r.lo, 1)}–#{Float.round(r.hi, 1)} | **#{speedup}x** | " <>
        "#{if r.acceptance, do: "#{r.acceptance}%", else: "n/a"} |"
    )
  end
end

IO.puts("\nfull stats from the best run:")

rows
|> Enum.reject(&Map.has_key?(&1, :error))
|> Enum.max_by(& &1.tps, fn -> nil end)
|> IO.inspect()
