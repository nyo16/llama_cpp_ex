defmodule LlamaCppEx.ChatCompletion do
  @moduledoc """
  OpenAI-compatible chat completion response struct.

  Mirrors the shape of `POST /v1/chat/completions` responses.
  """

  @type t :: %__MODULE__{
          id: String.t(),
          object: String.t(),
          created: integer(),
          model: String.t(),
          choices: [choice()],
          usage: usage()
        }

  @type choice :: %{
          index: integer(),
          message: %{
            required(:role) => String.t(),
            required(:content) => String.t() | nil,
            required(:reasoning_content) => String.t() | nil,
            optional(:tool_calls) => [tool_call()]
          },
          finish_reason: String.t()
        }

  @typedoc """
  A tool call, present on `message` only when the reply called a tool (see the
  `:tools` option of `LlamaCppEx.chat_completion/3`). `arguments` is JSON text,
  as in the OpenAI API.
  """
  @type tool_call :: %{
          id: String.t(),
          type: String.t(),
          function: %{name: String.t(), arguments: String.t()}
        }

  @type usage :: %{
          prompt_tokens: integer(),
          completion_tokens: integer(),
          total_tokens: integer()
        }

  defstruct [:id, :object, :created, :model, :choices, :usage]
end
