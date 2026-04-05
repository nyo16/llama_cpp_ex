Code.require_file("helpers.exs", __DIR__)

alias LlamaCppEx.Server.Strategy.{DecodeMaximal, PrefillPriority, Balanced}

IO.puts("Batching Strategy Benchmark")
IO.puts("  Comparing strategies under concurrent load")
IO.puts("")

short_prompt = "The capital of France is"
long_prompt = String.duplicate("The quick brown fox jumps over the lazy dog. ", 30)
concurrency = 4

# Start a server per strategy
servers = %{
  "decode_maximal" =>
    Bench.Helpers.start_server(
      n_parallel: concurrency,
      n_ctx: 4096,
      batch_strategy: DecodeMaximal
    ),
  "prefill_priority" =>
    Bench.Helpers.start_server(
      n_parallel: concurrency,
      n_ctx: 4096,
      batch_strategy: PrefillPriority
    ),
  "balanced" =>
    Bench.Helpers.start_server(
      n_parallel: concurrency,
      n_ctx: 4096,
      batch_strategy: Balanced
    )
}

run_concurrent = fn server, prompt, n ->
  tasks =
    for _ <- 1..n do
      Task.async(fn ->
        {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 32)
      end)
    end

  Task.await_many(tasks, 120_000)
end

Benchee.run(
  %{
    "decode_maximal / short prompt" => fn ->
      run_concurrent.(servers["decode_maximal"], short_prompt, concurrency)
    end,
    "prefill_priority / short prompt" => fn ->
      run_concurrent.(servers["prefill_priority"], short_prompt, concurrency)
    end,
    "balanced / short prompt" => fn ->
      run_concurrent.(servers["balanced"], short_prompt, concurrency)
    end,
    "decode_maximal / long prompt" => fn ->
      run_concurrent.(servers["decode_maximal"], long_prompt, concurrency)
    end,
    "prefill_priority / long prompt" => fn ->
      run_concurrent.(servers["prefill_priority"], long_prompt, concurrency)
    end,
    "balanced / long prompt" => fn ->
      run_concurrent.(servers["balanced"], long_prompt, concurrency)
    end
  },
  warmup: 1,
  time: 15,
  formatters: [
    {Benchee.Formatters.Console, extended_statistics: true}
  ]
)

Enum.each(servers, fn {_, s} -> GenServer.stop(s) end)
