Code.require_file("helpers.exs", __DIR__)

# The short suite (~6 / ~110 / ~220 tokens) plus the >1k regime (exact 1k / 8k /
# 32k prompts). Keeping both in one run is the point: the long inputs are where
# the server's per-token costs live, and the short ones are the only axis
# comparable with the numbers under bench/results/.

# 32768 prompt + 128 generated, with headroom.
n_ctx = 40_960

server = Bench.Helpers.start_server(n_parallel: 1, n_ctx: n_ctx)
model = LlamaCppEx.Server.get_model(server)

inputs =
  Bench.Helpers.prompts()
  |> Map.merge(Bench.Helpers.long_prompts(model))
  |> Map.new(fn {name, prompt} -> {name, {name, prompt}} end)

for {name, {_, prompt}} <- Enum.sort_by(inputs, fn {_, {_, p}} -> byte_size(p) end) do
  {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, prompt)
  IO.puts("  #{name}: #{length(tokens)} tokens")
end

IO.puts("")

Benchee.run(
  %{
    "server generate 32 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 32)
    end,
    "server generate 128 tokens" => fn {_name, prompt} ->
      {:ok, _} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 128)
    end
  },
  inputs: inputs,
  warmup: 1,
  time: 10,
  formatters: [{Benchee.Formatters.Console, extended_statistics: true}]
)

GenServer.stop(server)
