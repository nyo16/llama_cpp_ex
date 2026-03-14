if Code.ensure_loaded?(Ecto.Schema) do
  defmodule LlamaCppEx.SchemaTest do
    use ExUnit.Case, async: true

    alias LlamaCppEx.Schema

    # -- Test schemas --

    defmodule Address do
      use Ecto.Schema

      @primary_key false
      embedded_schema do
        field(:street, :string)
        field(:city, :string)
        field(:zip, :string)
      end
    end

    defmodule Tag do
      use Ecto.Schema

      @primary_key false
      embedded_schema do
        field(:label, :string)
        field(:weight, :float)
      end
    end

    defmodule PersonWithAddress do
      use Ecto.Schema

      @primary_key false
      embedded_schema do
        field(:name, :string)
        embeds_one(:address, Address)
      end
    end

    defmodule PersonWithTags do
      use Ecto.Schema

      @primary_key false
      embedded_schema do
        field(:name, :string)
        embeds_many(:tags, Tag)
      end
    end

    defmodule FullTypesSchema do
      use Ecto.Schema

      @primary_key false
      embedded_schema do
        field(:a_date, :date)
        field(:a_datetime, :utc_datetime)
        field(:a_decimal, :decimal)
        field(:a_map, :map)
        field(:a_boolean, :boolean)
        field(:an_integer, :integer)
      end
    end

    defmodule EmptySchema do
      use Ecto.Schema

      @primary_key false
      embedded_schema do
      end
    end

    # -- Tests --

    describe "embeds_one" do
      test "nested embedded schema produces nested object" do
        schema = Schema.to_json_schema(PersonWithAddress)

        assert schema["type"] == "object"
        assert schema["properties"]["name"] == %{"type" => "string"}

        address = schema["properties"]["address"]
        assert address["type"] == "object"
        assert address["properties"]["street"] == %{"type" => "string"}
        assert address["properties"]["city"] == %{"type" => "string"}
        assert address["properties"]["zip"] == %{"type" => "string"}
        assert "street" in address["required"]
        assert "city" in address["required"]

        assert "address" in schema["required"]
      end
    end

    describe "embeds_many" do
      test "nested embedded schema produces array of objects" do
        schema = Schema.to_json_schema(PersonWithTags)

        assert schema["type"] == "object"
        tags = schema["properties"]["tags"]
        assert tags["type"] == "array"
        assert tags["items"]["type"] == "object"
        assert tags["items"]["properties"]["label"] == %{"type" => "string"}
        assert tags["items"]["properties"]["weight"] == %{"type" => "number"}
        assert "tags" in schema["required"]
      end
    end

    describe "additional Ecto types" do
      test "date, utc_datetime, decimal, map, boolean, integer" do
        schema = Schema.to_json_schema(FullTypesSchema)

        assert schema["properties"]["a_date"] == %{"type" => "string", "format" => "date"}

        assert schema["properties"]["a_datetime"] == %{
                 "type" => "string",
                 "format" => "date-time"
               }

        assert schema["properties"]["a_decimal"] == %{"type" => "number"}
        assert schema["properties"]["a_map"] == %{"type" => "object"}
        assert schema["properties"]["a_boolean"] == %{"type" => "boolean"}
        assert schema["properties"]["an_integer"] == %{"type" => "integer"}
      end
    end

    describe "empty schema" do
      test "schema with no fields produces empty object" do
        schema = Schema.to_json_schema(EmptySchema)

        assert schema["type"] == "object"
        assert schema["properties"] == %{}
        assert schema["required"] == []
      end
    end

    describe "end-to-end: nested schema -> JSON Schema -> GBNF" do
      test "nested embeds_one schema converts to valid GBNF" do
        :ok = LlamaCppEx.init()

        json_schema = Schema.to_json_schema(PersonWithAddress)
        assert {:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(json_schema)
        assert is_binary(gbnf)
        assert byte_size(gbnf) > 0
        assert gbnf =~ "root"
      end

      test "nested embeds_many schema converts to valid GBNF" do
        :ok = LlamaCppEx.init()

        json_schema = Schema.to_json_schema(PersonWithTags)
        assert {:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(json_schema)
        assert is_binary(gbnf)
        assert byte_size(gbnf) > 0
      end
    end
  end
end
