# Usage: LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/server.exs
#
# Demonstrates the continuous batching server with concurrent requests.

model_path =
  System.get_env("LLAMA_MODEL_PATH") ||
    raise "Set LLAMA_MODEL_PATH to a .gguf model file"

:ok = LlamaCppEx.init()

{:ok, server} =
  LlamaCppEx.Server.start_link(
    model_path: model_path,
    n_gpu_layers: -1,
    n_parallel: 4,
    n_ctx: 4096
  )

IO.puts("Server started with 4 parallel slots")
IO.puts("---")

# --- Synchronous generation ---

IO.puts("\n=== Synchronous generation ===")
{:ok, text} = LlamaCppEx.Server.generate(server, "What is Elixir?", max_tokens: 64)
IO.puts(text)

# --- Streaming ---

IO.puts("\n=== Streaming ===")

LlamaCppEx.Server.stream(server, "Count from 1 to 5:", max_tokens: 64)
|> Enum.each(&IO.write/1)

IO.puts("")

# --- Concurrent requests ---

IO.puts("\n=== Concurrent requests (4 tasks) ===")

prompts = [
  "Name a programming language:",
  "Name a color:",
  "Name a planet:",
  "Name an animal:"
]

tasks =
  Enum.map(prompts, fn prompt ->
    Task.async(fn ->
      {:ok, text} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 32)
      {prompt, text}
    end)
  end)

results = Task.await_many(tasks, 60_000)

for {prompt, text} <- results do
  IO.puts("  Prompt: #{prompt}")
  IO.puts("  Reply:  #{String.trim(text)}")
  IO.puts("")
end

# --- Server stats ---

stats = LlamaCppEx.Server.get_stats(server)
IO.puts("Server stats: #{inspect(stats)}")
