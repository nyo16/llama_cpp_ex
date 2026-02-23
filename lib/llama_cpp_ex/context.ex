defmodule LlamaCppEx.Context do
  @moduledoc """
  Inference context with KV cache.
  """

  @enforce_keys [:ref, :model]
  defstruct [:ref, :model]

  @type t :: %__MODULE__{ref: reference(), model: LlamaCppEx.Model.t()}

  @doc """
  Creates a new inference context for the given model.

  ## Options

    * `:n_ctx` - Context size (max tokens). Defaults to `2048`.
    * `:n_batch` - Max tokens per decode batch. Defaults to `2048`.
    * `:n_ubatch` - Max tokens per micro-batch. Defaults to `512`.
    * `:n_threads` - Number of threads for generation. Defaults to system CPU count.
    * `:n_threads_batch` - Number of threads for prompt processing. Defaults to `:n_threads`.

  """
  @spec create(LlamaCppEx.Model.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def create(%LlamaCppEx.Model{ref: model_ref} = model, opts \\ []) do
    n_threads = Keyword.get(opts, :n_threads, System.schedulers_online())
    n_ctx = Keyword.get(opts, :n_ctx, 2048)
    n_batch = Keyword.get(opts, :n_batch, n_ctx)
    n_ubatch = Keyword.get(opts, :n_ubatch, 512)
    n_threads_batch = Keyword.get(opts, :n_threads_batch, n_threads)

    case LlamaCppEx.NIF.context_create(
           model_ref,
           n_ctx,
           n_batch,
           n_ubatch,
           n_threads,
           n_threads_batch
         ) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref, model: model}}
      {:error, _} = error -> error
    end
  end

  @doc "Returns the context size."
  @spec n_ctx(t()) :: integer()
  def n_ctx(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.context_n_ctx(ref)

  @doc "Clears the KV cache."
  @spec clear(t()) :: :ok
  def clear(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.memory_clear(ref)

  @doc """
  Decodes a list of tokens through the model.
  """
  @spec decode(t(), [integer()]) :: :ok | {:error, String.t()}
  def decode(%__MODULE__{ref: ref}, tokens) when is_list(tokens) do
    LlamaCppEx.NIF.decode(ref, tokens)
  end

  @doc """
  Runs the generation loop: decodes prompt tokens and generates up to `max_tokens` new tokens.

  Returns the generated text (not including the prompt).

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.

  """
  @spec generate(t(), LlamaCppEx.Sampler.t(), [integer()], keyword()) ::
          {:ok, String.t()} | {:error, String.t()}
  def generate(
        %__MODULE__{ref: ctx_ref},
        %LlamaCppEx.Sampler{ref: sampler_ref},
        tokens,
        opts \\ []
      ) do
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    LlamaCppEx.NIF.generate(ctx_ref, sampler_ref, tokens, max_tokens)
  end
end
