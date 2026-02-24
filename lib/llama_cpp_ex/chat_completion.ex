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
          message: %{role: String.t(), content: String.t()},
          finish_reason: String.t()
        }

  @type usage :: %{
          prompt_tokens: integer(),
          completion_tokens: integer(),
          total_tokens: integer()
        }

  defstruct [:id, :object, :created, :model, :choices, :usage]
end
