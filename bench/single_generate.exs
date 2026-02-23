Code.require_file("helpers.exs", __DIR__)

model = Bench.Helpers.setup()

Benchee.run(
  %{
    "generate 32 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.generate(model, prompt, max_tokens: 32, temp: 0.0)
    end,
    "generate 128 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.generate(model, prompt, max_tokens: 128, temp: 0.0)
    end
  },
  inputs: Bench.Helpers.prompts() |> Enum.map(fn {k, v} -> {k, {k, v}} end) |> Map.new(),
  warmup: 1,
  time: 10,
  formatters: [{Benchee.Formatters.Console, extended_statistics: true}]
)
