Code.require_file("helpers.exs", __DIR__)

# Prefix caching under concurrency — the headline benchmark for the caching
# work: K interleaved multi-turn conversations sharing one system prompt,
# served by n_parallel slots. Reports per-turn TTFT and the prefix-cache hit
# ratio for cache-off vs cache-on (cross-slot sharing + session affinity).
#
# Run: LLAMA_MODEL_PATH=... mix run bench/prefix_cache_concurrent.exs
# (Use a dense-attention model for cross-slot sharing; hybrid GDN models
# degrade to exact-prefix per-slot hits.)

defmodule Bench.PrefixCacheConcurrent do
  @moduledoc false

  @n_parallel 4
  @n_conversations 8
  @n_turns 4
  @max_tokens 24

  def run do
    system_prompt =
      "You are a helpful assistant. Answer concisely. " <>
        String.duplicate("Follow the house style: short sentences, plain words. ", 24)

    IO.puts("Prefix Cache Under Concurrency")

    IO.puts(
      "  #{@n_conversations} conversations x #{@n_turns} turns, " <>
        "n_parallel #{@n_parallel}, shared system prompt, #{@max_tokens} tok/turn"
    )

    IO.puts("")

    modes = [
      {"cache OFF            ", cache_prompt: false},
      {"per-slot cache       ", cache_prompt: true, kv_unified: false},
      {"cross-slot + affinity", cache_prompt: true, kv_unified: true}
    ]

    for {label, opts} <- modes do
      {ttfts, cache_ratios} = run_workload(opts, system_prompt)
      avg_ratio = Enum.sum(cache_ratios) / length(cache_ratios)

      IO.puts(
        "  #{label}: TTFT median #{fmt(median(ttfts))} ms, p95 #{fmt(percentile(ttfts, 95))} ms" <>
          ", mean hit ratio #{Float.round(avg_ratio * 100, 1)}%"
      )
    end
  end

  defp run_workload(opts, system_prompt) do
    server =
      Bench.Helpers.start_server([n_parallel: @n_parallel, n_ctx: 8192] ++ opts)

    collector = start_collector()

    # Each conversation runs its turns sequentially; conversations run
    # concurrently (2x the slot count, so the queue and eviction paths are
    # exercised too).
    tasks =
      for conv <- 1..@n_conversations do
        Task.async(fn ->
          opener = "\nUser: Question #{conv}.1 — name #{conv} colors.\nAssistant:"

          Enum.reduce(1..@n_turns, system_prompt <> opener, fn turn, prompt ->
            {:ok, reply} =
              LlamaCppEx.Server.generate(server, prompt,
                max_tokens: @max_tokens,
                timeout: 120_000,
                session: conv
              )

            prompt <> reply <> "\nUser: Question #{conv}.#{turn + 1} — one more.\nAssistant:"
          end)
        end)
      end

    Task.await_many(tasks, 600_000)

    measurements = stop_collector(collector)
    GenServer.stop(server)

    ttfts = Enum.map(measurements, & &1.ttft_ms)
    ratios = Enum.map(measurements, & &1.prefix_cache_ratio)
    {ttfts, ratios}
  end

  # ETS-based collector — no extra process needed for a bench script.
  defp start_collector do
    table = :ets.new(:prefix_cache_bench, [:public, :duplicate_bag])

    :telemetry.attach(
      {__MODULE__, table},
      [:llama_cpp_ex, :server, :request, :done],
      fn _event, m, _meta, _ ->
        :ets.insert(table, {:m, m.ttft_ms, m.prefix_cache_ratio})
      end,
      nil
    )

    table
  end

  defp stop_collector(table) do
    :telemetry.detach({__MODULE__, table})

    measurements =
      for {:m, ttft, ratio} <- :ets.tab2list(table) do
        %{ttft_ms: ttft, prefix_cache_ratio: ratio}
      end

    :ets.delete(table)
    measurements
  end

  defp median(values) do
    sorted = Enum.sort(values)
    Enum.at(sorted, div(length(sorted), 2))
  end

  defp percentile(values, p) do
    sorted = Enum.sort(values)
    idx = min(length(sorted) - 1, round(p / 100 * (length(sorted) - 1)))
    Enum.at(sorted, idx)
  end

  defp fmt(ms), do: :erlang.float_to_binary(ms / 1, decimals: 1)
end

Bench.PrefixCacheConcurrent.run()
