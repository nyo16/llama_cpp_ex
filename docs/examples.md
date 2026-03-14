# Examples

Runnable example scripts are in the `examples/` directory. Each script can be run with `mix run`.

## Basic Generation

```bash
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/basic_generation.exs
```

```elixir
:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

{:ok, text} =
  LlamaCppEx.generate(model, "Explain what Elixir is in one paragraph:",
    max_tokens: 256,
    temp: 0.7,
    seed: 42
  )

IO.puts(text)
```

## Streaming

Stream tokens to the terminal as they are generated.

```bash
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/streaming.exs
```

```elixir
model
|> LlamaCppEx.stream("Once upon a time in a land of functional programming,",
  max_tokens: 256,
  temp: 0.8
)
|> Enum.each(&IO.write/1)
```

## Interactive Chat

Multi-turn chat loop using `stream_chat/3`. Type "exit" or "quit" to stop.

```bash
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/chat.exs
```

```elixir
messages = [
  %{role: "system", content: "You are a helpful assistant. Be concise."},
  %{role: "user", content: "What is pattern matching?"}
]

chunks =
  model
  |> LlamaCppEx.stream_chat(messages, max_tokens: 512, temp: 0.7)
  |> Enum.map(fn chunk ->
    IO.write(chunk)
    chunk
  end)
```

## Structured Output

JSON Schema constrained generation with optional Ecto schema integration.

```bash
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/structured_output.exs
```

```elixir
schema = %{
  "type" => "object",
  "properties" => %{
    "name" => %{"type" => "string"},
    "age" => %{"type" => "integer"},
    "hobbies" => %{"type" => "array", "items" => %{"type" => "string"}}
  },
  "required" => ["name", "age", "hobbies"],
  "additionalProperties" => false
}

{:ok, json} =
  LlamaCppEx.chat(
    model,
    [%{role: "user", content: "Generate a profile for a fictional software developer."}],
    json_schema: schema,
    max_tokens: 256,
    temp: 0.7
  )
```

With Ecto schemas:

```elixir
defmodule Book do
  use Ecto.Schema

  @primary_key false
  embedded_schema do
    field(:title, :string)
    field(:author, :string)
    field(:year, :integer)
    field(:genre, :string)
  end
end

ecto_schema = LlamaCppEx.Schema.to_json_schema(Book)

{:ok, book_json} =
  LlamaCppEx.chat(
    model,
    [%{role: "user", content: "Generate a JSON object for a classic science fiction book."}],
    json_schema: ecto_schema,
    max_tokens: 256,
    temp: 0.3
  )
```

## Embeddings

Embedding generation and cosine similarity.

```bash
LLAMA_EMBEDDING_MODEL_PATH=/path/to/embedding-model.gguf mix run examples/embeddings.exs
```

```elixir
{:ok, model} = LlamaCppEx.load_model(embedding_model_path, n_gpu_layers: -1)

texts = [
  "Elixir is a functional programming language.",
  "Erlang runs on the BEAM virtual machine.",
  "The weather today is sunny and warm."
]

{:ok, embeddings} = LlamaCppEx.embed_batch(model, texts)

# Compute cosine similarity between pairs
cosine_similarity = fn a, b ->
  dot = Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  norm_a = :math.sqrt(Enum.reduce(a, 0.0, fn x, acc -> acc + x * x end))
  norm_b = :math.sqrt(Enum.reduce(b, 0.0, fn x, acc -> acc + x * x end))
  dot / (norm_a * norm_b)
end
```

## Continuous Batching Server

Server with concurrent requests using `LlamaCppEx.Server`.

```bash
LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/server.exs
```

```elixir
{:ok, server} =
  LlamaCppEx.Server.start_link(
    model_path: model_path,
    n_gpu_layers: -1,
    n_parallel: 4,
    n_ctx: 4096
  )

# Synchronous
{:ok, text} = LlamaCppEx.Server.generate(server, "What is Elixir?", max_tokens: 64)

# Streaming
LlamaCppEx.Server.stream(server, "Count from 1 to 5:", max_tokens: 64)
|> Enum.each(&IO.write/1)

# Concurrent requests
prompts = ["Name a language:", "Name a color:", "Name a planet:", "Name an animal:"]

tasks = Enum.map(prompts, fn prompt ->
  Task.async(fn ->
    {:ok, text} = LlamaCppEx.Server.generate(server, prompt, max_tokens: 32)
    {prompt, text}
  end)
end)

results = Task.await_many(tasks, 60_000)
```
