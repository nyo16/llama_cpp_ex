Code.require_file("helpers.exs", __DIR__)

IO.puts("Tokenization Overhead Benchmark")
IO.puts("  Comparing text API vs pre-tokenized API")
IO.puts("")

server = Bench.Helpers.start_server(n_parallel: 1, n_ctx: 4096)
model = LlamaCppEx.Server.get_model(server)

prompts = Bench.Helpers.prompts()

# Pre-tokenize all prompts
tokenized =
  Map.new(prompts, fn {name, text} ->
    {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, text)
    IO.puts("  #{name}: #{length(tokens)} tokens")
    {name, tokens}
  end)

IO.puts("")

Benchee.run(
  %{
    "generate (text API)" => fn {name, _tokens} ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompts[name], max_tokens: 32)
    end,
    "generate_tokens (pre-tokenized)" => fn {_name, tokens} ->
      {:ok, _} = LlamaCppEx.Server.generate_tokens(server, tokens, max_tokens: 32)
    end
  },
  inputs: tokenized |> Enum.map(fn {k, v} -> {k, {k, v}} end) |> Map.new(),
  warmup: 1,
  time: 10,
  formatters: [
    {Benchee.Formatters.Console, extended_statistics: true}
  ]
)

GenServer.stop(server)
