Code.require_file("helpers.exs", __DIR__)

alias LlamaCppEx.Server.Strategy.{DecodeMaximal, PrefillPriority, Balanced}

IO.puts("Batching Strategy Benchmark")
IO.puts("  Comparing strategies: single request throughput")
IO.puts("")

prompt = "The capital of France is"

strategies = [
  {"decode_maximal", DecodeMaximal},
  {"prefill_priority", PrefillPriority},
  {"balanced", Balanced}
]

benchmarks =
  Map.new(strategies, fn {name, strategy} ->
    server =
      Bench.Helpers.start_server(
        n_parallel: 1,
        n_ctx: 8192,
        batch_strategy: strategy,
        cache_prompt: false
      )

    {name,
     {fn -> {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 32) end,
      after_scenario: fn _ -> GenServer.stop(server) end}}
  end)

Benchee.run(
  benchmarks,
  warmup: 1,
  time: 10,
  formatters: [{Benchee.Formatters.Console, extended_statistics: true}]
)
