Code.require_file("helpers.exs", __DIR__)

# Single-node baseline for a DGX Spark (GB10).
#
#   scripts/spark/remote.sh --env MIX_ENV=bench spark-1 \
#     mix run bench/spark_baseline.exs
#
# Prefill and decode are reported **separately**, because on this chip they are
# two different machines. Prefill is compute-bound and excellent; decode is
# bandwidth-bound and the number people are disappointed by. A single
# tokens-per-second figure averages the two into something that describes
# neither, and the published Spark comparisons quote them apart.
#
# The separation uses only the public API: two generations from the same prompt
# with prompt caching off differ by exactly K decode steps, so
#
#   decode  = K / (t(1+K) - t(1))
#   prefill = n_prompt / (t(1) - one_decode_step)
#
# That is llama-bench's pp/tg split, derived rather than instrumented.

n_ctx = 40_960
decode_steps = 64

server = Bench.Helpers.start_server(n_parallel: 1, n_ctx: n_ctx, cache_prompt: false)
model = Bench.Helpers.await_model(server)

defmodule SparkBaseline do
  def time_ms(fun) do
    t0 = System.monotonic_time(:microsecond)
    fun.()
    (System.monotonic_time(:microsecond) - t0) / 1000
  end

  # Median of `n` runs. Decode on a bandwidth-bound part is noisy enough that a
  # single sample is not worth printing.
  def median(values) do
    sorted = Enum.sort(values)
    len = length(sorted)

    case rem(len, 2) do
      1 -> Enum.at(sorted, div(len, 2))
      0 -> (Enum.at(sorted, div(len, 2) - 1) + Enum.at(sorted, div(len, 2))) / 2
    end
  end

  def split(server, prompt, n_prompt, decode_steps, samples) do
    gen = fn max_tokens ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: max_tokens)
    end

    # Warm the kernels and any autotuning before the first timed run.
    gen.(4)

    t_one = median(for _ <- 1..samples, do: time_ms(fn -> gen.(1) end))
    t_many = median(for _ <- 1..samples, do: time_ms(fn -> gen.(1 + decode_steps) end))

    per_decode = (t_many - t_one) / decode_steps
    prefill_ms = t_one - per_decode

    %{
      n_prompt: n_prompt,
      prefill_ms: prefill_ms,
      ttft_ms: t_one,
      prefill_tps: n_prompt * 1000 / prefill_ms,
      decode_tps: 1000 / per_decode
    }
  end
end

# The short suite plus the >1k regime, same inputs as bench/server_generate.exs
# so the numbers stay comparable with everything under bench/results/.
inputs =
  Bench.Helpers.prompts()
  |> Map.merge(Bench.Helpers.long_prompts(model))
  |> Enum.map(fn {name, prompt} ->
    {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)
    {name, prompt, length(tokens)}
  end)
  |> Enum.sort_by(fn {_, _, n} -> n end)

IO.puts("\nprefill / decode split (median of 3, #{decode_steps} decode steps)\n")
IO.puts("| prompt | tokens | TTFT ms | prefill t/s | decode t/s |")
IO.puts("|---|---|---|---|---|")

for {name, prompt, n_prompt} <- inputs do
  r = SparkBaseline.split(server, prompt, n_prompt, decode_steps, 3)

  IO.puts(
    "| #{name} | #{r.n_prompt} | #{Float.round(r.ttft_ms, 1)} | " <>
      "#{Float.round(r.prefill_tps, 1)} | #{Float.round(r.decode_tps, 2)} |"
  )
end

IO.puts("""

External single-Spark references for comparison (llama.cpp, Q4_K_M):
  Qwen3-8B       pp512 3167 t/s   tg 43.7 t/s
  Qwen3-30B-A3B  pp512 2541 t/s   tg 89.3 t/s
A gap over 20% is a finding, not noise — explain it before tuning anything.
""")

# The established Benchee suite, unchanged in shape from
# bench/server_generate.exs so the wall-clock numbers stay comparable.
benchee_inputs = Map.new(inputs, fn {name, prompt, _n} -> {name, {name, prompt}} end)

Benchee.run(
  %{
    "server generate 32 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 32)
    end,
    "server generate 128 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 128)
    end
  },
  inputs: benchee_inputs,
  warmup: 1,
  time: 10,
  formatters: [{Benchee.Formatters.Console, extended_statistics: true}]
)

GenServer.stop(server)
