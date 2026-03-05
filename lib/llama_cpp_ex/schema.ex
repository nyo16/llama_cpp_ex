defmodule LlamaCppEx.Schema do
  @moduledoc """
  Converts Ecto schema modules to JSON Schema maps for structured output.

  Requires `{:ecto, "~> 3.0"}` as an optional dependency. If Ecto is not
  available, calling these functions will raise at runtime.

  ## Type Mapping

  | Ecto type | JSON Schema |
  |-----------|-------------|
  | `:string`, `:binary` | `{"type": "string"}` |
  | `:integer`, `:id` | `{"type": "integer"}` |
  | `:float`, `:decimal` | `{"type": "number"}` |
  | `:boolean` | `{"type": "boolean"}` |
  | `:map` | `{"type": "object"}` |
  | `{:array, inner}` | `{"type": "array", "items": ...}` |
  | `:date` | `{"type": "string", "format": "date"}` |
  | `:utc_datetime`, etc. | `{"type": "string", "format": "date-time"}` |
  | `embeds_one` | nested object |
  | `embeds_many` | array of nested objects |

  ## Excluded Fields

  The following fields are automatically excluded: `:id`, `:inserted_at`,
  `:updated_at`, and virtual fields.

  ## Examples

      defmodule MyApp.Person do
        use Ecto.Schema

        schema "people" do
          field :name, :string
          field :age, :integer
          field :email, :string
          timestamps()
        end
      end

      schema = LlamaCppEx.Schema.to_json_schema(MyApp.Person)
      # => %{"type" => "object", "properties" => %{"name" => ..., "age" => ..., "email" => ...}, ...}

      # Use directly with generate/chat
      {:ok, json} = LlamaCppEx.chat(model, messages, json_schema: schema, temp: 0.0)

  """

  @doc """
  Converts an Ecto schema module to a JSON Schema map.

  Extracts fields from the schema, maps Ecto types to JSON Schema types,
  and marks all non-virtual fields as required. Excludes `:id`, timestamp
  fields (`:inserted_at`, `:updated_at`), and virtual fields.
  """
  @spec to_json_schema(module()) :: map()
  def to_json_schema(module) do
    unless Code.ensure_loaded?(Ecto.Schema) do
      raise "Ecto is required for LlamaCppEx.Schema. Add {:ecto, \"~> 3.0\"} to your deps."
    end

    unless function_exported?(module, :__schema__, 1) do
      raise ArgumentError, "#{inspect(module)} is not an Ecto schema module"
    end

    fields = module.__schema__(:fields)
    types = for field <- fields, into: %{}, do: {field, module.__schema__(:type, field)}

    # Exclude :id and timestamp fields
    excluded = MapSet.new([:id, :inserted_at, :updated_at])
    fields = Enum.reject(fields, &MapSet.member?(excluded, &1))

    # Exclude virtual fields (they won't appear in __schema__(:fields) anyway,
    # but filter by checking type is not nil)
    fields = Enum.filter(fields, fn f -> types[f] != nil end)

    # Build properties and handle embeds
    embeds_one = module.__schema__(:embeds) |> MapSet.new()

    {properties, required} =
      Enum.reduce(fields, {%{}, []}, fn field, {props, req} ->
        if MapSet.member?(embeds_one, field) do
          embed_schema = module.__schema__(:embed, field)
          embed_mod = embed_schema.related

          case embed_schema.cardinality do
            :one ->
              nested = to_json_schema(embed_mod)
              {Map.put(props, to_string(field), nested), [to_string(field) | req]}

            :many ->
              nested = to_json_schema(embed_mod)
              item = %{"type" => "array", "items" => nested}
              {Map.put(props, to_string(field), item), [to_string(field) | req]}
          end
        else
          json_type = ecto_type_to_json(types[field])
          {Map.put(props, to_string(field), json_type), [to_string(field) | req]}
        end
      end)

    %{
      "type" => "object",
      "properties" => properties,
      "required" => Enum.reverse(required)
    }
  end

  defp ecto_type_to_json(:string), do: %{"type" => "string"}
  defp ecto_type_to_json(:binary), do: %{"type" => "string"}
  defp ecto_type_to_json(:integer), do: %{"type" => "integer"}
  defp ecto_type_to_json(:float), do: %{"type" => "number"}
  defp ecto_type_to_json(:decimal), do: %{"type" => "number"}
  defp ecto_type_to_json(:boolean), do: %{"type" => "boolean"}
  defp ecto_type_to_json(:map), do: %{"type" => "object"}
  defp ecto_type_to_json({:map, _}), do: %{"type" => "object"}

  defp ecto_type_to_json({:array, inner}),
    do: %{"type" => "array", "items" => ecto_type_to_json(inner)}

  defp ecto_type_to_json(:date), do: %{"type" => "string", "format" => "date"}
  defp ecto_type_to_json(:time), do: %{"type" => "string", "format" => "time"}
  defp ecto_type_to_json(:utc_datetime), do: %{"type" => "string", "format" => "date-time"}
  defp ecto_type_to_json(:utc_datetime_usec), do: %{"type" => "string", "format" => "date-time"}
  defp ecto_type_to_json(:naive_datetime), do: %{"type" => "string", "format" => "date-time"}
  defp ecto_type_to_json(:naive_datetime_usec), do: %{"type" => "string", "format" => "date-time"}
  defp ecto_type_to_json(:id), do: %{"type" => "integer"}
  defp ecto_type_to_json(:binary_id), do: %{"type" => "string"}
  defp ecto_type_to_json(_other), do: %{"type" => "string"}
end
