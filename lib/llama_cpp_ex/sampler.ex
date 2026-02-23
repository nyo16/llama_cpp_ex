defmodule LlamaCppEx.Sampler do
  @moduledoc """
  Token sampling configuration.

  Builds a sampler chain with the common sampling parameters.
  The samplers are applied in order: penalties -> top_k -> top_p -> min_p -> temp -> dist/greedy.
  """

  @enforce_keys [:ref]
  defstruct [:ref]

  @type t :: %__MODULE__{ref: reference()}

  @doc """
  Creates a new sampler chain.

  ## Options

    * `:seed` - Random seed for sampling. Defaults to a random value.
    * `:temp` - Temperature. `0.0` for greedy sampling. Defaults to `0.8`.
    * `:top_k` - Top-K filtering. `0` to disable. Defaults to `40`.
    * `:top_p` - Top-P (nucleus) filtering. `1.0` to disable. Defaults to `0.95`.
    * `:min_p` - Min-P filtering. `0.0` to disable. Defaults to `0.05`.
    * `:penalty_repeat` - Repetition penalty. `1.0` to disable. Defaults to `1.0`.

  """
  @spec create(keyword()) :: {:ok, t()}
  def create(opts \\ []) do
    seed = Keyword.get(opts, :seed, :rand.uniform(1_000_000_000))
    temp = Keyword.get(opts, :temp, 0.8)
    top_k = Keyword.get(opts, :top_k, 40)
    top_p = Keyword.get(opts, :top_p, 0.95)
    min_p = Keyword.get(opts, :min_p, 0.05)
    penalty_repeat = Keyword.get(opts, :penalty_repeat, 1.0)

    ref =
      LlamaCppEx.NIF.sampler_init(seed, temp / 1, top_k, top_p / 1, min_p / 1, penalty_repeat / 1)

    {:ok, %__MODULE__{ref: ref}}
  end

  @doc "Resets the sampler state."
  @spec reset(t()) :: :ok
  def reset(%__MODULE__{ref: ref}), do: LlamaCppEx.NIF.sampler_reset(ref)

  @doc "Accepts a token (updates sampler internal state)."
  @spec accept(t(), integer()) :: :ok
  def accept(%__MODULE__{ref: ref}, token), do: LlamaCppEx.NIF.sampler_accept(ref, token)

  @doc "Samples the next token from the context's logits."
  @spec sample(t(), LlamaCppEx.Context.t()) :: integer()
  def sample(%__MODULE__{ref: ref}, %LlamaCppEx.Context{ref: ctx_ref}) do
    LlamaCppEx.NIF.sampler_sample(ref, ctx_ref)
  end
end
