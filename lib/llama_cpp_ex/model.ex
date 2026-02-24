defmodule LlamaCppEx.Model do
  @moduledoc """
  Model loading and introspection.
  """

  @enforce_keys [:ref]
  defstruct [:ref]

  @type t :: %__MODULE__{ref: reference()}

  @doc """
  Loads a GGUF model from the given file path.

  ## Options

    * `:n_gpu_layers` - Number of layers to offload to GPU. Use `-1` for all layers.
      Defaults to `99` (offload all layers).
    * `:use_mmap` - Whether to memory-map the model file. Defaults to `true`.

  ## Examples

      {:ok, model} = LlamaCppEx.Model.load("path/to/model.gguf", n_gpu_layers: -1)

  """
  @spec load(String.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def load(path, opts \\ []) do
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, 99)
    use_mmap = Keyword.get(opts, :use_mmap, true)

    case LlamaCppEx.NIF.model_load(path, n_gpu_layers, use_mmap) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref}}
      {:error, _} = error -> error
    end
  end

  @doc "Returns the training context size of the model."
  @spec n_ctx_train(t()) :: integer()
  def n_ctx_train(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_ctx_train(ref)

  @doc "Returns the embedding dimension of the model."
  @spec n_embd(t()) :: integer()
  def n_embd(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_embd(ref)

  @doc "Returns a human-readable description of the model."
  @spec desc(t()) :: String.t()
  def desc(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_desc(ref)

  @doc "Returns the model file size in bytes."
  @spec size(t()) :: integer()
  def size(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_size(ref)

  @doc "Returns the number of model parameters."
  @spec n_params(t()) :: integer()
  def n_params(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.model_n_params(ref)

  @doc """
  Returns the chat template string embedded in the model, or `nil` if none.
  """
  @spec chat_template(t()) :: String.t() | nil
  def chat_template(%__MODULE__{ref: ref}) do
    case LlamaCppEx.NIF.model_chat_template(ref) do
      "" -> nil
      template -> template
    end
  end
end
