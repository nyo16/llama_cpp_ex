# Binding overhead against upstream's own tools, on one model.
#
#   scripts/spark/remote.sh --env MIX_ENV=bench --env LLAMA_MODEL_PATH=<gguf> \
#     --big-cores spark-1 mix run bench/spark_overhead.exs
#
# Compare with, from the same llama.cpp commit:
#
#   llama-bench -m <gguf> -fa 1 -p 512 -n 128 -r 3
#   llama-batched-bench -m <gguf> -fa 1 -c 16384 -npp 128 -ntg 128 -npl 1,4,8
#
# Three decode paths, each timed from the first streamed token to the last so
# prompt processing and context creation stay out of the number:
#
#   stream   LlamaCppEx.stream/3 -- the C++ generate_tokens loop, one sequence
#   server1  LlamaCppEx.Server, n_parallel 1 -- the batch_eval_sample tick loop
#   serverN  LlamaCppEx.Server, n_parallel N, N concurrent streams -- aggregate
#
# llama-bench's tg does no sampling and sends nothing anywhere, so the gap to
# `stream` is the whole binding cost per token: sampler chain, token_to_piece,
# enif_send and the BEAM receive.

model_path = System.get_env("LLAMA_MODEL_PATH") || raise "Set LLAMA_MODEL_PATH"
n_gen = String.to_integer(System.get_env("SPARK_N_GEN") || "128")
samples = String.to_integer(System.get_env("SPARK_SAMPLES") || "3")

parallel =
  (System.get_env("SPARK_PARALLEL") || "4,8")
  |> String.split(",", trim: true)
  |> Enum.map(&String.to_integer/1)

# Base-model friendly and long-running under greedy decoding; the chat template
# is deliberately not applied so every path sees the same tokens.
prompt =
  "Write a detailed, multi-chapter history of the Roman Empire, starting with " <>
    "the founding of Rome and continuing through the fall of Constantinople.\n\n"

defmodule Overhead do
  def median([_ | _] = xs) do
    s = Enum.sort(xs)
    n = length(s)
    if rem(n, 2) == 1, do: Enum.at(s, div(n, 2)), else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
  end

  # Consumes an enumerable of chunks and returns {tokens_after_first, seconds}.
  def time_decode(enum) do
    {count, first, last} =
      Enum.reduce(enum, {0, nil, nil}, fn
        {:error, reason}, _ ->
          raise "generation failed: #{inspect(reason)}"

        chunk, {n, first, _} when is_binary(chunk) or is_map(chunk) ->
          now = System.monotonic_time(:microsecond)
          {n + 1, first || now, now}

        _other, acc ->
          acc
      end)

    {count - 1, (last - first) / 1_000_000}
  end

  def tps({n, s}) when s > 0, do: n / s
end

:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

IO.puts("model: #{Path.basename(model_path)}  n_gen: #{n_gen}  samples: #{samples}\n")
IO.puts("| path | streams | tokens/stream | per-stream t/s | aggregate t/s |")
IO.puts("|---|---|---|---|---|")

gen_opts = [max_tokens: n_gen, temp: 0.0, n_ctx: 4096]

# Warm-up: kernel selection and CUDA graph capture happen on the first run.
model |> LlamaCppEx.stream(prompt, gen_opts) |> Stream.run()

runs = for _ <- 1..samples, do: Overhead.time_decode(LlamaCppEx.stream(model, prompt, gen_opts))
tps = Overhead.median(Enum.map(runs, &Overhead.tps/1))
{n, _} = hd(runs)
IO.puts("| stream | 1 | #{n} | #{Float.round(tps, 2)} | #{Float.round(tps, 2)} |")

for n_par <- [1 | parallel] do
  {:ok, server} =
    LlamaCppEx.Server.start_link(
      model_path: model_path,
      n_gpu_layers: -1,
      n_parallel: n_par,
      n_ctx: 4096 * n_par,
      temp: 0.0,
      cache_prompt: false
    )

  # start_link returns before the load; a generate call queues behind it.
  {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 8)

  one_round = fn ->
    t0 = System.monotonic_time(:microsecond)

    results =
      1..n_par
      |> Task.async_stream(
        fn _ ->
          Overhead.time_decode(LlamaCppEx.Server.stream(server, prompt, max_tokens: n_gen))
        end,
        max_concurrency: n_par,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, r} -> r end)

    wall = (System.monotonic_time(:microsecond) - t0) / 1_000_000
    per = Overhead.median(Enum.map(results, &Overhead.tps/1))
    total = results |> Enum.map(&elem(&1, 0)) |> Enum.sum()
    {per, total / wall, results |> hd() |> elem(0)}
  end

  rounds = for _ <- 1..samples, do: one_round.()
  per = Overhead.median(Enum.map(rounds, &elem(&1, 0)))
  agg = Overhead.median(Enum.map(rounds, &elem(&1, 1)))
  toks = rounds |> hd() |> elem(2)

  IO.puts(
    "| server#{n_par} | #{n_par} | #{toks} | #{Float.round(per, 2)} | #{Float.round(agg, 2)} |"
  )

  GenServer.stop(server)
end
