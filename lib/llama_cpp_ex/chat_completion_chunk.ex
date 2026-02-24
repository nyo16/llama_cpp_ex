defmodule LlamaCppEx.ChatCompletionChunk do
  @moduledoc """
  OpenAI-compatible streaming chat completion chunk struct.

  Mirrors the shape of server-sent events from `POST /v1/chat/completions` with `stream: true`.
  """

  @type t :: %__MODULE__{
          id: String.t(),
          object: String.t(),
          created: integer(),
          model: String.t(),
          choices: [chunk_choice()]
        }

  @type chunk_choice :: %{
          index: integer(),
          delta: %{optional(:role) => String.t(), optional(:content) => String.t()},
          finish_reason: String.t() | nil
        }

  defstruct [:id, :object, :created, :model, :choices]
end
