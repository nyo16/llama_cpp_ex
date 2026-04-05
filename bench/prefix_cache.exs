Code.require_file("helpers.exs", __DIR__)

# Benchee comparison: with vs without prefix caching
# Simulates multi-turn chat where each request extends the previous

system_prompt = "You are a helpful assistant. Answer concisely."

turns = [
  "What is Elixir?",
  "How does it compare to Ruby?",
  "Show me a GenServer example.",
  "What about supervision trees?"
]

# Build progressively longer prompts (simulating multi-turn chat)
prompts =
  Enum.scan(turns, system_prompt, fn turn, acc ->
    acc <> "\nUser: " <> turn <> "\nAssistant:"
  end)

IO.puts("Prefix Cache Benchmark")

IO.puts(
  "  Prompts: #{length(prompts)} turns, final length: #{String.length(List.last(prompts))} chars"
)

IO.puts("")

# Server with caching enabled
server_cached = Bench.Helpers.start_server(n_parallel: 1, n_ctx: 4096, cache_prompt: true)

# Server with caching disabled
server_no_cache = Bench.Helpers.start_server(n_parallel: 1, n_ctx: 4096, cache_prompt: false)

Benchee.run(
  %{
    "multi-turn WITH prefix cache" => fn ->
      for prompt <- prompts do
        {:ok, _} = LlamaCppEx.Server.generate(server_cached, prompt, max_tokens: 16)
      end
    end,
    "multi-turn WITHOUT prefix cache" => fn ->
      for prompt <- prompts do
        {:ok, _} = LlamaCppEx.Server.generate(server_no_cache, prompt, max_tokens: 16)
      end
    end
  },
  warmup: 1,
  time: 15,
  memory_time: 2,
  formatters: [
    {Benchee.Formatters.Console, extended_statistics: true}
  ]
)

GenServer.stop(server_cached)
GenServer.stop(server_no_cache)
