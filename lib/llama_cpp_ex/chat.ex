defmodule LlamaCppEx.Chat do
  @moduledoc """
  Chat template formatting using llama.cpp's built-in template engine.

  Converts a list of chat messages into a formatted prompt string
  using the model's embedded chat template (or a custom one).

  ## Examples

      {:ok, prompt} = LlamaCppEx.Chat.apply_template(model, [
        %{role: "system", content: "You are helpful."},
        %{role: "user", content: "Hi!"}
      ])

  """

  @type message :: %{role: String.t(), content: String.t()} | {String.t(), String.t()}

  @doc """
  Applies a chat template to a list of messages, producing a formatted prompt string.

  ## Options

    * `:template` - Custom template string. Defaults to the model's embedded template.
    * `:add_assistant` - Whether to add the assistant turn prefix. Defaults to `true`.

  """
  @spec apply_template(LlamaCppEx.Model.t(), [message()], keyword()) ::
          {:ok, String.t()} | {:error, String.t()}
  def apply_template(%LlamaCppEx.Model{} = model, messages, opts \\ []) when is_list(messages) do
    add_assistant = Keyword.get(opts, :add_assistant, true)

    template =
      case Keyword.get(opts, :template) do
        nil ->
          case LlamaCppEx.Model.chat_template(model) do
            nil ->
              raise ArgumentError, "model has no embedded chat template, pass :template option"

            tmpl ->
              tmpl
          end

        tmpl ->
          tmpl
      end

    msg_tuples =
      Enum.map(messages, fn
        %{role: role, content: content} -> {to_string(role), to_string(content)}
        {role, content} -> {to_string(role), to_string(content)}
      end)

    {:ok, LlamaCppEx.NIF.chat_apply_template(template, msg_tuples, add_assistant)}
  end
end
