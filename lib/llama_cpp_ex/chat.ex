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

  """

  @type message :: %{role: String.t(), content: String.t()} | {String.t(), String.t()}

  @doc """
  Applies the model's chat template to a list of messages using the Jinja engine.

  ## Options

    * `:add_assistant` - Whether to add the assistant turn prefix. Defaults to `true`.
    * `:enable_thinking` - Whether to enable thinking/reasoning mode. Defaults to `true`.
    * `:chat_template_kwargs` - Extra template variables as a list of `{key, value}` string tuples.
      Defaults to `[]`.

  """
  @spec apply_template(LlamaCppEx.Model.t(), [message()], keyword()) ::
          {:ok, String.t()} | {:error, String.t()}
  def apply_template(%LlamaCppEx.Model{} = model, messages, opts \\ []) when is_list(messages) do
    add_assistant = Keyword.get(opts, :add_assistant, true)
    enable_thinking = Keyword.get(opts, :enable_thinking, true)
    extra_kwargs = Keyword.get(opts, :chat_template_kwargs, [])

    msg_tuples =
      Enum.map(messages, fn
        %{role: role, content: content} -> {to_string(role), to_string(content)}
        {role, content} -> {to_string(role), to_string(content)}
      end)

    kwargs_tuples =
      Enum.map(extra_kwargs, fn {k, v} -> {to_string(k), to_string(v)} end)

    try do
      result =
        LlamaCppEx.NIF.chat_apply_template_jinja(
          model.ref,
          msg_tuples,
          add_assistant,
          enable_thinking,
          kwargs_tuples
        )

      {:ok, result}
    rescue
      e in ErlangError -> {:error, "chat template failed: #{inspect(e.original)}"}
    end
  end
end
