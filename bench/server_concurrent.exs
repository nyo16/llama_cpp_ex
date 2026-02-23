Code.require_file("helpers.exs", __DIR__)

defmodule Bench.Concurrent do
  @moduledoc false

  @iterations 5
  @concurrency_levels [1, 2, 4]

  def run(opts \\ []) do
    max_tokens = Keyword.get(opts, :max_tokens, 32)
    prompt = Keyword.get(opts, :prompt, "short")
    n_parallel = Enum.max(@concurrency_levels)

    server =
      Bench.Helpers.start_server(n_parallel: n_parallel, n_ctx: Keyword.get(opts, :n_ctx, 4096))

    prompt_text = Bench.Helpers.prompts()[prompt]

    # Attach telemetry collector
    tick_agent = start_tick_collector()

    IO.puts("")

    IO.puts(
      "Continuous Batching Benchmark (max_tokens: #{max_tokens}, prompt: #{inspect(prompt)})"
    )

    IO.puts(String.duplicate("\u2500", 78))

    header =
      pad("Concurrency", 13) <>
        pad("Wall time", 12) <>
        pad("Total tok/s", 14) <>
        pad("Per-req tok/s", 16) <>
        pad("Speedup", 10) <>
        pad("Avg batch", 10)

    IO.puts(header)

    _baseline_wall =
      Enum.reduce(@concurrency_levels, nil, fn n, baseline_wall ->
        {timings, batch_sizes} = run_level(server, prompt_text, max_tokens, n, tick_agent)
        wall_median = median(timings)
        wall_p99 = percentile(timings, 99)

        total_tokens = n * max_tokens
        total_tok_s = total_tokens / (wall_median / 1_000_000)
        per_req_tok_s = total_tok_s / n

        avg_batch =
          if batch_sizes == [], do: 0.0, else: Enum.sum(batch_sizes) / length(batch_sizes)

        baseline_wall = baseline_wall || wall_median

        speedup =
          total_tok_s / (baseline_wall |> then(fn bw -> 1 * max_tokens / (bw / 1_000_000) end))

        row =
          pad(to_string(n), 13) <>
            pad(format_time(wall_median) <> " (p99: #{format_time(wall_p99)})", 12 + 18) <>
            pad(:erlang.float_to_binary(total_tok_s, decimals: 1), 14) <>
            pad(:erlang.float_to_binary(per_req_tok_s, decimals: 1), 16) <>
            pad(:erlang.float_to_binary(speedup, decimals: 2) <> "x", 10) <>
            pad(:erlang.float_to_binary(avg_batch, decimals: 1), 10)

        IO.puts(row)
        baseline_wall
      end)

    IO.puts("")

    # Clean up
    Agent.stop(tick_agent)
    GenServer.stop(server)

    :ok
  end

  defp run_level(server, prompt, max_tokens, concurrency, tick_agent) do
    # Reset tick collector
    Agent.update(tick_agent, fn _ -> [] end)

    timings =
      for _ <- 1..@iterations do
        # Warm up: single request to stabilize
        if concurrency == 1 do
          {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: max_tokens)
        end

        Agent.update(tick_agent, fn _ -> [] end)

        start = System.monotonic_time(:microsecond)

        tasks =
          for _ <- 1..concurrency do
            Task.async(fn ->
              {:ok, _text} = LlamaCppEx.Server.generate(server, prompt, max_tokens: max_tokens)
            end)
          end

        Task.await_many(tasks, 120_000)
        elapsed = System.monotonic_time(:microsecond) - start
        elapsed
      end

    batch_sizes = Agent.get(tick_agent, & &1)
    {timings, batch_sizes}
  end

  defp start_tick_collector do
    {:ok, agent} = Agent.start_link(fn -> [] end)

    :telemetry.attach(
      "bench-tick-collector",
      [:llama_cpp_ex, :server, :tick],
      fn _event, measurements, _meta, _config ->
        Agent.update(agent, fn sizes -> [measurements.batch_size | sizes] end)
      end,
      nil
    )

    agent
  end

  defp median(list) do
    sorted = Enum.sort(list)
    len = length(sorted)
    mid = div(len, 2)

    if rem(len, 2) == 0 do
      (Enum.at(sorted, mid - 1) + Enum.at(sorted, mid)) / 2
    else
      Enum.at(sorted, mid)
    end
  end

  defp percentile(list, p) do
    sorted = Enum.sort(list)
    k = p / 100 * (length(sorted) - 1)
    f = floor(k)
    c = ceil(k)

    if f == c do
      Enum.at(sorted, f)
    else
      Enum.at(sorted, f) * (c - k) + Enum.at(sorted, c) * (k - f)
    end
  end

  defp format_time(us) when us < 1_000, do: "#{round(us)}us"

  defp format_time(us) when us < 1_000_000,
    do: :erlang.float_to_binary(us / 1_000, decimals: 1) <> "ms"

  defp format_time(us), do: :erlang.float_to_binary(us / 1_000_000, decimals: 2) <> "s"

  defp pad(str, width) do
    len = String.length(str)
    if len >= width, do: str <> " ", else: str <> String.duplicate(" ", width - len)
  end
end

Bench.Concurrent.run()
