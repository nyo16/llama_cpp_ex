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
    * `:main_gpu` - GPU device index for single-GPU mode. Defaults to `0`.
    * `:split_mode` - How to split the model across GPUs: `:none`, `:layer`, or `:row`.
      Defaults to `:none`.
    * `:tensor_split` - List of floats specifying the proportion of work per GPU
      (e.g. `[0.5, 0.5]` for two GPUs). Defaults to `[]`.
    * `:use_mlock` - Pin model memory in RAM to prevent swapping. Defaults to `false`.
    * `:use_direct_io` - Bypass page cache when loading (takes precedence over mmap).
      Defaults to `false`.
    * `:vocab_only` - Load vocabulary and metadata only, skip weights. Defaults to `false`.

  ## Examples

      {:ok, model} = LlamaCppEx.Model.load("path/to/model.gguf", n_gpu_layers: -1)
      {:ok, model} = LlamaCppEx.Model.load("path/to/model.gguf", split_mode: :layer, tensor_split: [0.5, 0.5])
      {:ok, model} = LlamaCppEx.Model.load("path/to/model.gguf", vocab_only: true)

  """
  @spec load(String.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def load(path, opts \\ []) do
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, 99)
    use_mmap = Keyword.get(opts, :use_mmap, true)
    main_gpu = Keyword.get(opts, :main_gpu, 0)
    split_mode = Keyword.get(opts, :split_mode, :none) |> encode_split_mode()
    tensor_split = Keyword.get(opts, :tensor_split, [])
    use_mlock = Keyword.get(opts, :use_mlock, false)
    use_direct_io = Keyword.get(opts, :use_direct_io, false)
    vocab_only = Keyword.get(opts, :vocab_only, false)

    case LlamaCppEx.NIF.model_load(
           path,
           n_gpu_layers,
           use_mmap,
           main_gpu,
           split_mode,
           tensor_split,
           use_mlock,
           use_direct_io,
           vocab_only
         ) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref}}
      {:error, _} = error -> error
    end
  end

  defp encode_split_mode(:none), do: 0
  defp encode_split_mode(:layer), do: 1
  defp encode_split_mode(:row), do: 2

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
