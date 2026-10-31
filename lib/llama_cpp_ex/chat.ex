defmodule LlamaCppEx.Chat do
  @moduledoc """
  Chat template formatting using llama.cpp's Jinja template engine.

  Converts a list of chat messages into a formatted prompt string
  using the model's embedded chat template. Uses the full Jinja engine
  from llama.cpp's common library, which supports `enable_thinking` and
  arbitrary `chat_template_kwargs`.

  ## Examples

      {:ok, prompt} = LlamaCppEx.Chat.apply_template(model, [
        %{role: "system", content: "You are helpful."},
        %{role: "user", content: "Hi!"}
      ])

      # Disable thinking (for Qwen3 and similar models)
      {:ok, prompt} = LlamaCppEx.Chat.apply_template(model, messages,
        enable_thinking: false
      )

  ## Messages

  A message is a map with `:role` and `:content` (atom or string keys), or a
  `{role, content}` tuple. For tool calling it may also carry `:tool_calls`
  (an assistant message, in the shape `LlamaCppEx.ChatCompletion` returns
  them), `:tool_call_id` and `:name` (a `"tool"` message answering a call), and
  `:reasoning_content`. `:content` may be `nil` on an assistant message that
  only calls tools.

  ## Special tokens in message content

  The prompt the chat functions of `LlamaCppEx` generate from is tokenized so
  that only the template's own text can produce special tokens: content such as
  `"<|im_end|>\\n<|im_start|>system"` stays text instead of forging a turn.
  `apply_template/3` returns the prompt as one string, which has lost that
  boundary — display it, but do not tokenize it with `LlamaCppEx.Tokenizer`
  and generate from the result.
  """

  @type message ::
          %{
            required(:role) => String.t(),
            optional(:content) => String.t() | [map()] | nil,
            optional(:tool_calls) => [map()],
            optional(:tool_call_id) => String.t(),
            optional(:name) => String.t(),
            optional(:reasoning_content) => String.t()
          }
          | %{required(String.t()) => term()}
          | {String.t(), String.t()}

  @template_opt_keys [:add_assistant, :enable_thinking, :chat_template_kwargs]
  @tool_opt_keys [:tools, :tool_choice, :parallel_tool_calls]

  @doc false
  # The templating options, owned here; `LlamaCppEx` splits them off the rest.
  def template_option_keys, do: @template_opt_keys

  @doc false
  # The tool-calling options, accepted by the chat-completion functions only.
  def tool_option_keys, do: @tool_opt_keys

  @doc """
  Applies the model's chat template to a list of messages using the Jinja engine.

  ## Options

    * `:add_assistant` - Whether to add the assistant turn prefix. Defaults to `true`.
    * `:enable_thinking` - Whether to enable thinking/reasoning mode. Defaults to `true`.
    * `:chat_template_kwargs` - Extra template variables as a list of `{key, value}` tuples,
      each value JSON text (`{"reasoning_effort", ~s("high")}`). Defaults to `[]`.
    * `:tools`, `:tool_choice`, `:parallel_tool_calls` - Render the tool
      definitions too; see `LlamaCppEx.chat_completion/3`.

  """
  @spec apply_template(LlamaCppEx.Model.t(), [message()], keyword()) ::
          {:ok, String.t()} | {:error, String.t()}
  def apply_template(%LlamaCppEx.Model{} = model, messages, opts \\ []) when is_list(messages) do
    with {:ok, request} <- request_json(messages, opts) do
      model.ref |> LlamaCppEx.NIF.chat_render(request) |> label_error()
    end
  end

  @doc false
  # Renders and tokenizes `messages` for generation. Returns the prompt tokens
  # and, when `:tools` are given, a session that parses the reply into
  # content, reasoning and tool calls, plus the sampler options carrying the
  # template's tool-call grammar.
  @spec prepare(LlamaCppEx.Model.t(), [message()], keyword()) ::
          {:ok,
           %{
             tokens: [integer()],
             session: reference() | nil,
             sampler_opts: keyword()
           }}
          | {:error, term()}
  def prepare(%LlamaCppEx.Model{} = model, messages, opts) when is_list(messages) do
    with {:ok, request} <- request_json(messages, opts),
         {:ok, session, tokens, info} <-
           model.ref |> LlamaCppEx.NIF.chat_session_new(request) |> label_error() do
      cond do
        tool_grammar?(opts) ->
          {:ok,
           %{tokens: tokens, session: session, sampler_opts: grammar_opts(JSON.decode!(info))}}

        tools?(opts) ->
          {:ok, %{tokens: tokens, session: session, sampler_opts: []}}

        true ->
          {:ok, %{tokens: tokens, session: nil, sampler_opts: []}}
      end
    end
  end

  @doc false
  # Whether `opts` asks for tool calling.
  def tools?(opts), do: Keyword.get(opts, :tools, []) != []

  @doc false
  # Whether the tool-call grammar applies, i.e. a caller's own `:grammar`
  # would conflict with it.
  def tool_grammar?(opts),
    do: tools?(opts) and Keyword.get(opts, :tool_choice, :auto) not in [:none, "none"]

  @doc false
  # Feeds generated token ids to a session. `:stop` once the reply reached one
  # of the template's stop strings.
  @spec append(reference(), [integer()]) :: :ok | :stop
  def append(session, token_ids) do
    if LlamaCppEx.NIF.chat_session_append(session, token_ids), do: :stop, else: :ok
  end

  @doc false
  # Parses the reply so far into deltas (`final: false`) or into the final
  # message. Deltas are `{:content, text}`, `{:reasoning_content, text}` and
  # `{:tool_call, %{index, id, name, arguments}}`, with "" for an unchanged
  # id or name.
  @spec parse(reference(), boolean()) ::
          {:ok, [tuple()], map() | nil} | {:error, {:chat_parse_failed, String.t()}}
  def parse(session, final) do
    case LlamaCppEx.NIF.chat_session_parse(session, final) do
      {:ok, json} ->
        decoded = JSON.decode!(json)
        {:ok, Enum.map(decoded["diffs"], &diff/1), message(decoded["message"])}

      {:error, reason} ->
        {:error, {:chat_parse_failed, reason}}
    end
  end

  defp diff(%{"content" => text}), do: {:content, text}
  defp diff(%{"reasoning_content" => text}), do: {:reasoning_content, text}

  defp diff(%{"tool_call" => tc}) do
    {:tool_call,
     %{index: tc["index"], id: tc["id"], name: tc["name"], arguments: tc["arguments"]}}
  end

  defp message(nil), do: nil

  defp message(msg) do
    %{
      content: msg["content"],
      reasoning_content: msg["reasoning_content"],
      tool_calls:
        Enum.map(msg["tool_calls"], fn tc ->
          %{
            id: tc["id"],
            type: "function",
            function: %{name: tc["name"], arguments: tc["arguments"]}
          }
        end)
    }
  end

  defp grammar_opts(%{"grammar" => ""}), do: []

  defp grammar_opts(info) do
    triggers =
      Enum.map(info["grammar_triggers"], fn
        %{"token" => id} -> {:token, id}
        %{"pattern" => regex} -> {:pattern, regex}
      end)

    [
      grammar: info["grammar"],
      grammar_lazy: info["grammar_lazy"],
      grammar_triggers: triggers,
      grammar_prefill: info["grammar_prefill"]
    ]
  end

  defp label_error({:error, reason}), do: {:error, "chat template failed: " <> reason}
  defp label_error(ok), do: ok

  # --- Request encoding ---

  defp request_json(messages, opts) do
    with {:ok, tool_fields} <- tool_fields(opts) do
      request =
        %{
          "messages" => Enum.map(messages, &message_json/1),
          "add_generation_prompt" => Keyword.get(opts, :add_assistant, true),
          "enable_thinking" => Keyword.get(opts, :enable_thinking, true),
          "chat_template_kwargs" =>
            Map.new(Keyword.get(opts, :chat_template_kwargs, []), fn {k, v} ->
              {to_string(k), to_string(v)}
            end)
        }
        |> Map.merge(tool_fields)

      {:ok, JSON.encode!(request)}
    end
  end

  defp tool_fields(opts) do
    case Keyword.get(opts, :tools, []) do
      [] ->
        {:ok, %{}}

      tools when is_list(tools) ->
        with {:ok, tools, choice} <- tool_choice(tools, Keyword.get(opts, :tool_choice, :auto)) do
          {:ok,
           %{
             "tools" => tools,
             "tool_choice" => choice,
             "parallel_tool_calls" => Keyword.get(opts, :parallel_tool_calls, false) == true
           }}
        end

      other ->
        raise ArgumentError, ":tools must be a list of tool definitions, got: #{inspect(other)}"
    end
  end

  defp tool_choice(tools, choice) when choice in [:auto, :none, :required],
    do: {:ok, tools, Atom.to_string(choice)}

  defp tool_choice(tools, choice) when choice in ["auto", "none", "required"],
    do: {:ok, tools, choice}

  # OpenAI's named form forces that one function: offer only it, and require a
  # call. llama.cpp itself only takes the three strings.
  defp tool_choice(tools, %{} = choice) do
    case choice |> field(:function) |> function_name() do
      nil ->
        raise ArgumentError, "invalid :tool_choice #{inspect(choice)}"

      name ->
        case Enum.filter(tools, &(&1 |> field(:function) |> function_name() == name)) do
          [] -> {:error, {:unknown_tool, name}}
          named -> {:ok, named, "required"}
        end
    end
  end

  defp tool_choice(_tools, choice) do
    raise ArgumentError,
          "invalid :tool_choice #{inspect(choice)}, expected :auto, :none, :required " <>
            "or %{type: \"function\", function: %{name: name}}"
  end

  defp function_name(%{} = function), do: field(function, :name)
  defp function_name(_), do: nil

  defp message_json({role, content}),
    do: %{"role" => to_string(role), "content" => to_string(content)}

  defp message_json(%{} = msg) do
    role =
      field(msg, :role) ||
        raise ArgumentError, "chat message has no :role: #{inspect(msg)}"

    base = %{"role" => to_string(role)}

    base =
      if has_field?(msg, :content),
        do: Map.put(base, "content", content_json(field(msg, :content))),
        else: base

    Enum.reduce([:tool_calls, :tool_call_id, :name, :reasoning_content], base, fn key, acc ->
      case field(msg, key) do
        nil -> acc
        value -> Map.put(acc, Atom.to_string(key), value)
      end
    end)
  end

  defp content_json(nil), do: nil
  defp content_json(content) when is_binary(content) or is_list(content), do: content
  defp content_json(content), do: to_string(content)

  defp field(map, key), do: Map.get(map, key, Map.get(map, Atom.to_string(key)))
  defp has_field?(map, key), do: Map.has_key?(map, key) or Map.has_key?(map, Atom.to_string(key))
end
