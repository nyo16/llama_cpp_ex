# Multi-Model Manager Example
#
# Demonstrates keeping two models resident at once and routing requests to them
# by id with LlamaCppEx.ModelManager — a generation model (server-backed, with
# batching/streaming) and an embedding model (direct mode).
#
# Usage:
#   LLAMA_GEN_MODEL_PATH=/path/to/chat-model.gguf \
#   LLAMA_EMB_MODEL_PATH=/path/to/embedding-model.gguf \
#     mix run examples/model_manager.exs
#
# Either path may also be a HuggingFace repo+file via {:hub, repo, file}.

gen_path = System.get_env("LLAMA_GEN_MODEL_PATH") || raise "set LLAMA_GEN_MODEL_PATH"
emb_path = System.get_env("LLAMA_EMB_MODEL_PATH") || raise "set LLAMA_EMB_MODEL_PATH"

# Start the supervisor (Registry + DynamicSupervisor + ModelManager). In an app,
# add {LlamaCppEx.ModelSupervisor, ...} to your supervision tree instead.
{:ok, _sup} = LlamaCppEx.ModelSupervisor.start_link(memory_budget: :auto)

IO.puts("=== Loading two models ===\n")

# Server-backed generation model, marked as the default route.
{:ok, "chat"} =
  LlamaCppEx.ModelManager.load("chat", {:path, gen_path},
    n_gpu_layers: -1,
    n_ctx: 2048,
    default: true
  )

# Embedding model — :embed capability auto-selects :direct mode.
{:ok, "embed"} =
  LlamaCppEx.ModelManager.load("embed", {:path, emb_path},
    n_gpu_layers: -1,
    capabilities: [:embed]
  )

for m <- LlamaCppEx.ModelManager.list() do
  IO.puts("  #{m.id}: #{m.status} (#{m.mode})")
end

IO.puts("\n=== Routing ===\n")

# Route a generation request to the chat model by id.
{:ok, text} =
  LlamaCppEx.ModelManager.generate("chat", "The capital of France is", max_tokens: 12, temp: 0.0)

IO.puts("generate(\"chat\"): #{inspect(text)}")

# Stream from the default model.
IO.write("stream(:default): ")

LlamaCppEx.ModelManager.stream(:default, "Count from 1 to 5:", max_tokens: 32)
|> Enum.each(&IO.write/1)

IO.puts("")

# Chat with templating, routed to the chat model.
{:ok, reply} =
  LlamaCppEx.ModelManager.chat("chat", [%{role: "user", content: "Say hello in one word."}],
    max_tokens: 16
  )

IO.puts("chat(\"chat\"): #{inspect(reply)}")

# Route an embedding request to the embedding model.
{:ok, vector} = LlamaCppEx.ModelManager.embed("embed", "hello world")
IO.puts("embed(\"embed\"): #{length(vector)}-dim vector")

# Embeddings are refused on a generation model.
{:error, :not_embedding_model} = LlamaCppEx.ModelManager.embed("chat", "x")
IO.puts("embed(\"chat\"): correctly refused")

IO.puts("\n=== Unload / reload ===\n")

:ok = LlamaCppEx.ModelManager.unload("chat")
IO.puts("after unload, loaded?(\"chat\"): #{LlamaCppEx.ModelManager.loaded?("chat")}")

{:ok, "chat"} =
  LlamaCppEx.ModelManager.load("chat", {:path, gen_path}, n_gpu_layers: -1, n_ctx: 2048)

IO.puts("reloaded; resident: #{inspect(Enum.map(LlamaCppEx.ModelManager.list(), & &1.id))}")
