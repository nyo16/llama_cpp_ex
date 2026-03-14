# Usage: LLAMA_MODEL_PATH=/path/to/model.gguf mix run examples/structured_output.exs
#
# Demonstrates JSON Schema constrained generation and Ecto schema integration.

model_path =
  System.get_env("LLAMA_MODEL_PATH") ||
    raise "Set LLAMA_MODEL_PATH to a .gguf model file"

:ok = LlamaCppEx.init()
{:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)

# --- Example 1: Raw JSON Schema ---

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

IO.puts("=== Raw JSON Schema ===")

{:ok, json} =
  LlamaCppEx.chat(
    model,
    [%{role: "user", content: "Generate a profile for a fictional software developer."}],
    json_schema: schema,
    max_tokens: 256,
    temp: 0.7
  )

IO.puts(json)

# --- Example 2: Ecto Schema integration ---

if Code.ensure_loaded?(Ecto.Schema) do
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

  IO.puts("\n=== Ecto Schema (Book) ===")

  ecto_schema = LlamaCppEx.Schema.to_json_schema(Book)
  IO.puts("JSON Schema: #{inspect(ecto_schema)}")

  {:ok, book_json} =
    LlamaCppEx.chat(
      model,
      [%{role: "user", content: "Generate a JSON object for a classic science fiction book."}],
      json_schema: ecto_schema,
      max_tokens: 256,
      temp: 0.3
    )

  IO.puts("Generated: #{book_json}")
end
