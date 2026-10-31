defmodule LlamaCppEx.InvalidJsonSchemaTest do
  # A `:json_schema` llama.cpp's converter rejects used to raise ArgumentError
  # out of functions specced to return `{:ok, _} | {:error, _}`. A schema is
  # data, often from user config, so a bad one is an error tuple now.
  #
  # The schema is converted before the model or the server is touched, so
  # neither has to exist: these run in the default suite.
  use ExUnit.Case, async: true

  alias LlamaCppEx.Model

  @bad %{
    "type" => "object",
    "properties" => %{"a" => %{"$ref" => "#/$defs/Missing"}},
    "required" => ["a"]
  }

  @unowned %Model{ref: nil}
  @messages [%{role: "user", content: "hi"}]

  test "tuple-returning functions return {:error, {:invalid_json_schema, message}}" do
    for {label, call} <- [
          {"generate/3", fn -> LlamaCppEx.generate(@unowned, "x", json_schema: @bad) end},
          {"chat/3", fn -> LlamaCppEx.chat(@unowned, @messages, json_schema: @bad) end},
          {"chat_completion/3",
           fn -> LlamaCppEx.chat_completion(@unowned, @messages, json_schema: @bad) end},
          {"chat_completion/3 (server)",
           fn -> LlamaCppEx.chat_completion(:no_such_server, @messages, json_schema: @bad) end}
        ] do
      assert {:error, {:invalid_json_schema, message}} = call.(), label
      assert message =~ "$defs", "#{label} lost the converter's message: #{message}"
    end
  end

  test "streams make the error their only element" do
    for {label, stream} <- [
          {"stream/3", LlamaCppEx.stream(@unowned, "x", json_schema: @bad)},
          {"stream_chat/3", LlamaCppEx.stream_chat(@unowned, @messages, json_schema: @bad)},
          {"stream_chat_completion/3",
           LlamaCppEx.stream_chat_completion(@unowned, @messages, json_schema: @bad)},
          {"stream_chat_completion/3 (server)",
           LlamaCppEx.stream_chat_completion(:no_such_server, @messages, json_schema: @bad)}
        ] do
      assert [{:error, {:invalid_json_schema, _message}}] = Enum.to_list(stream), label
    end
  end

  test "passing both :grammar and :json_schema is still a programmer error" do
    assert_raise ArgumentError, ~r/both :grammar and :json_schema/, fn ->
      LlamaCppEx.chat_completion(@unowned, @messages,
        grammar: "root ::= \"x\"",
        json_schema: %{"type" => "string"}
      )
    end
  end
end
