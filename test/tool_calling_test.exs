defmodule LlamaCppEx.ToolCallingTest do
  @moduledoc """
  Tool calling through `chat_completion/3` and `stream_chat_completion/3`
  (issue #111), against a model whose chat template supports tools:

      GGML_METAL_NO_RESIDENCY=1 \\
      LLAMA_SMOKE_TOOLS_MODEL=/path/to/Qwen3.5-0.8B-UD-Q4_K_XL.gguf \\
        mix test test/tool_calling_test.exs --include tools

  Written against Qwen3.5-0.8B. The prompts ask for a tool outright, so a small
  model at temperature 0 calls it; the assertions are about the shape the call
  comes back in. Llama-3.2-1B-Instruct passes all of it except the round trip:
  given the tool's result, it calls the tool again.
  """
  use ExUnit.Case, async: false

  alias LlamaCppEx.{ChatCompletionChunk, Server}

  @moduletag :tools
  @moduletag timeout: 300_000

  @weather %{
    type: "function",
    function: %{
      name: "get_weather",
      description: "Get the current weather for a city",
      parameters: %{
        type: "object",
        properties: %{city: %{type: "string", description: "City name"}},
        required: ["city"]
      }
    }
  }

  @time %{
    type: "function",
    function: %{
      name: "get_time",
      description: "Get the current local time in a city",
      parameters: %{
        type: "object",
        properties: %{city: %{type: "string"}},
        required: ["city"]
      }
    }
  }

  @ask [%{role: "user", content: "What's the weather in Paris right now? Use the tool."}]
  @opts [tools: [@weather], max_tokens: 300, temp: 0.0, enable_thinking: false]

  setup_all do
    :ok = LlamaCppEx.init()
    path = LlamaCppEx.TestModels.path!(:tools)
    {:ok, model} = LlamaCppEx.load_model(path, n_gpu_layers: -1)

    {:ok, server} =
      Server.start_link(model_path: path, n_gpu_layers: -1, n_parallel: 1, n_ctx: 4096)

    {:ok, model: model, server: server}
  end

  for route <- [:model, :server] do
    test "#{route}: a prompt that needs a tool returns a parsed call", ctx do
      assert {:ok, completion} = LlamaCppEx.chat_completion(ctx[unquote(route)], @ask, @opts)
      assert [%{finish_reason: "tool_calls", message: message}] = completion.choices

      assert message.content == nil
      assert [call] = message.tool_calls
      assert %{type: "function", function: %{name: "get_weather", arguments: args}} = call
      assert "call_" <> _ = call.id
      assert %{"city" => city} = JSON.decode!(args)
      assert city =~ "Paris"
    end

    test "#{route}: streaming gives the same call, in OpenAI's delta shape", ctx do
      {:ok, completion} = LlamaCppEx.chat_completion(ctx[unquote(route)], @ask, @opts)
      [%{message: %{tool_calls: [call]}}] = completion.choices

      chunks =
        ctx[unquote(route)] |> LlamaCppEx.stream_chat_completion(@ask, @opts) |> Enum.to_list()

      assert Enum.all?(chunks, &match?(%ChatCompletionChunk{}, &1))

      deltas = for %{choices: [choice]} <- chunks, do: choice
      calls = Enum.flat_map(deltas, &Map.get(&1.delta, :tool_calls, []))

      # The first delta of a call names it; the rest only add arguments.
      assert [
               %{index: 0, id: "call_" <> _, type: "function", function: %{name: "get_weather"}}
               | rest
             ] =
               calls

      assert Enum.all?(rest, &(&1.index == 0 and not Map.has_key?(&1, :id)))
      assert Enum.map_join(calls, & &1.function.arguments) == call.function.arguments
      assert List.last(deltas).finish_reason == "tool_calls"
    end
  end

  test "a tool result fed back yields a text answer", %{model: model} do
    {:ok, completion} = LlamaCppEx.chat_completion(model, @ask, @opts)
    [%{message: message}] = completion.choices
    [call] = message.tool_calls

    messages =
      @ask ++
        [
          message,
          %{role: "tool", tool_call_id: call.id, content: ~s({"temp_c": 18, "sky": "cloudy"})}
        ]

    assert {:ok, answer} = LlamaCppEx.chat_completion(model, messages, @opts)
    assert [%{finish_reason: "stop", message: reply}] = answer.choices
    refute Map.has_key?(reply, :tool_calls)
    assert reply.content =~ "18"
  end

  test "tool_choice: :none makes no call", %{model: model} do
    assert {:ok, completion} =
             LlamaCppEx.chat_completion(model, @ask, Keyword.put(@opts, :tool_choice, :none))

    assert [%{finish_reason: finish, message: message}] = completion.choices
    assert finish in ["stop", "length"]
    refute Map.has_key?(message, :tool_calls)
    refute message.content =~ "get_weather"
  end

  test ":required and a named tool_choice both produce a call", %{model: model} do
    assert {:ok, required} =
             LlamaCppEx.chat_completion(model, @ask, Keyword.put(@opts, :tool_choice, :required))

    assert [%{finish_reason: "tool_calls"}] = required.choices

    # Named: only that tool is offered, and a call to it is required.
    opts =
      Keyword.merge(@opts,
        tools: [@weather, @time],
        tool_choice: %{type: "function", function: %{name: "get_time"}}
      )

    ask = [%{role: "user", content: "What time is it in Paris? Use the tool."}]
    assert {:ok, named} = LlamaCppEx.chat_completion(model, ask, opts)
    assert [%{finish_reason: "tool_calls", message: %{tool_calls: calls}}] = named.choices
    assert Enum.all?(calls, &(&1.function.name == "get_time"))
  end

  test "a named tool_choice must name a tool that was given", %{model: model} do
    opts = Keyword.put(@opts, :tool_choice, %{type: "function", function: %{name: "nope"}})
    assert {:error, {:unknown_tool, "nope"}} = LlamaCppEx.chat_completion(model, @ask, opts)
  end

  test "the template's tool grammar and a caller's grammar cannot both apply", %{model: model} do
    assert_raise ArgumentError, ~r/:tools/, fn ->
      LlamaCppEx.chat_completion(model, @ask, Keyword.put(@opts, :grammar, "root ::= \"x\""))
    end

    assert_raise ArgumentError, ~r/:tools/, fn ->
      LlamaCppEx.stream_chat_completion(model, @ask, Keyword.put(@opts, :json_schema, %{}))
    end
  end

  # Issue #108 on a second chat template, and its tool half: tool results and
  # arguments flow back through the template as message text too.
  test "message content cannot forge turns of the conversation", %{model: model} do
    LlamaCppEx.TestChat.assert_content_cannot_forge_turns(model, model)
    LlamaCppEx.TestChat.assert_template_tokens_still_special(model, model)
  end

  test "tool results and arguments cannot forge turns either", %{model: model} do
    LlamaCppEx.TestChat.assert_tool_text_cannot_forge_turns(model, [@weather])
  end
end
