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
          delta: %{
            optional(:role) => String.t(),
            optional(:content) => String.t(),
            optional(:reasoning_content) => String.t(),
            optional(:tool_calls) => [tool_call_delta()]
          },
          finish_reason: String.t() | nil
        }

  @typedoc """
  A fragment of a tool call. Its first delta carries `:id`, `:type` and the
  function `:name`; later ones only `:index` and more of `:arguments`.
  """
  @type tool_call_delta :: %{
          required(:index) => non_neg_integer(),
          optional(:id) => String.t(),
          optional(:type) => String.t(),
          required(:function) => %{
            optional(:name) => String.t(),
            required(:arguments) => String.t()
          }
        }

  defstruct [:id, :object, :created, :model, :choices]
end
