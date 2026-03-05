defmodule LlamaCppEx.Grammar do
  @moduledoc """
  Converts JSON Schema to GBNF grammar for constrained generation.

  Uses llama.cpp's built-in `json_schema_to_grammar()` to convert a JSON Schema
  into a GBNF grammar string that can be used with the `:grammar` option.

  In most cases you don't need to call this module directly — pass `:json_schema`
  to `LlamaCppEx.generate/3`, `LlamaCppEx.chat/3`, or any other generate function
  and the conversion happens automatically.

  ## Examples

      schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        },
        "required" => ["name", "age"],
        "additionalProperties" => false
      }

      {:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(schema)
      # Use with the :grammar option
      {:ok, sampler} = LlamaCppEx.Sampler.create(model, grammar: gbnf, temp: 0.0)

  Supports all JSON Schema types: `object`, `array`, `string`, `number`,
  `integer`, `boolean`, `null`, `enum`, `oneOf`, `anyOf`, `allOf`, `$ref`, etc.
  """

  @doc """
  Converts a JSON Schema map to a GBNF grammar string.

  Returns `{:ok, gbnf_string}` on success or `{:error, reason}` on failure.
  """
  @spec from_json_schema(map()) :: {:ok, String.t()} | {:error, String.t()}
  def from_json_schema(schema) when is_map(schema) do
    json_str = JSON.encode!(schema)
    LlamaCppEx.NIF.json_schema_to_grammar_nif(json_str)
  end

  @doc """
  Converts a JSON Schema map to a GBNF grammar string.

  Returns the GBNF string on success or raises on failure.
  """
  @spec from_json_schema!(map()) :: String.t()
  def from_json_schema!(schema) when is_map(schema) do
    case from_json_schema(schema) do
      {:ok, gbnf} ->
        gbnf

      {:error, reason} ->
        raise ArgumentError, "failed to convert JSON schema to grammar: #{reason}"
    end
  end
end
