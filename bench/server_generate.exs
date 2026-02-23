Code.require_file("helpers.exs", __DIR__)

server = Bench.Helpers.start_server(n_parallel: 1)

Benchee.run(
  %{
    "server generate 32 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 32)
    end,
    "server generate 128 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 128)
    end
  },
  inputs: Bench.Helpers.prompts() |> Enum.map(fn {k, v} -> {k, {k, v}} end) |> Map.new(),
  warmup: 1,
  time: 10,
  formatters: [{Benchee.Formatters.Console, extended_statistics: true}]
)
