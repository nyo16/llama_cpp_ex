# Usage: LLAMA_EMBEDDING_MODEL_PATH=/path/to/embedding-model.gguf mix run examples/embeddings.exs
#
# Demonstrates embedding generation and cosine similarity.

model_path =
  System.get_env("LLAMA_EMBEDDING_MODEL_PATH") ||
    raise "Set LLAMA_EMBEDDING_MODEL_PATH to an embedding .gguf model file"

:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

IO.puts("Model: #{LlamaCppEx.Model.desc(model)}")
IO.puts("Embedding dimensions: #{LlamaCppEx.Model.n_embd(model)}")
IO.puts("---")

# --- Single embedding ---

{:ok, embedding} = LlamaCppEx.embed(model, "Elixir is a functional programming language.")
IO.puts("Single embedding (first 5 dims): #{inspect(Enum.take(embedding, 5))}")

# --- Batch embeddings & similarity ---

texts = [
  "Elixir is a functional programming language.",
  "Erlang runs on the BEAM virtual machine.",
  "The weather today is sunny and warm."
]

{:ok, embeddings} = LlamaCppEx.embed_batch(model, texts)

cosine_similarity = fn a, b ->
  dot = Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  norm_a = :math.sqrt(Enum.reduce(a, 0.0, fn x, acc -> acc + x * x end))
  norm_b = :math.sqrt(Enum.reduce(b, 0.0, fn x, acc -> acc + x * x end))
  dot / (norm_a * norm_b)
end

IO.puts("\nPairwise cosine similarities:")

for i <- 0..(length(texts) - 2), j <- (i + 1)..(length(texts) - 1) do
  sim = cosine_similarity.(Enum.at(embeddings, i), Enum.at(embeddings, j))
  IO.puts("  [#{i}] vs [#{j}]: #{Float.round(sim, 4)}")
  IO.puts("    \"#{Enum.at(texts, i)}\"")
  IO.puts("    \"#{Enum.at(texts, j)}\"")
end
