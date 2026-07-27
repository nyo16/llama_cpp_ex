defmodule LlamaCppEx.Sampler do
  @moduledoc """
  Token sampling configuration.

  Builds a sampler chain with the common sampling parameters.
  The samplers are applied in order: grammar -> penalties -> top_k -> top_p -> min_p -> temp -> dist/greedy.
  """

  @enforce_keys [:ref]
  defstruct [:ref, :model]

  @typedoc """
  A sampler chain.

  `:model` holds the `t:LlamaCppEx.Model.t/0` the chain was built from. It is
  carried so the model term cannot be garbage-collected while the chain is alive:
  a grammar stage keeps a raw pointer to the model's vocabulary for its whole
  lifetime, and dropping the model out from under it made the next `reset/1` or
  `accept/2` dereference freed memory. `nil` only for hand-constructed structs.
  """
  @type t :: %__MODULE__{ref: reference(), model: LlamaCppEx.Model.t() | nil}

  @option_keys [
    :seed,
    :temp,
    :top_k,
    :top_p,
    :min_p,
    :penalty_repeat,
    :penalty_freq,
    :penalty_present,
    :grammar,
    :grammar_root
  ]

  @doc """
  The options `create/2` accepts.

  This module is the single source of truth: callers that forward user sampling
  options (`LlamaCppEx`, `LlamaCppEx.Server`) select them with this function
  instead of keeping their own copy of the list. Every sampling option is safe
  to forward, so there is no tuning/structural split here.
  """
  @spec option_keys() :: [atom()]
  def option_keys, do: @option_keys

  @doc """
  Creates a new sampler chain.

  Requires a model reference (needed for grammar-constrained sampling).

  ## Options

    * `:seed` - Random seed for sampling. Defaults to a random value.
    * `:temp` - Temperature. `0.0` for greedy sampling. Defaults to `0.8`.
    * `:top_k` - Top-K filtering. `0` to disable. Defaults to `40`.
    * `:top_p` - Top-P (nucleus) filtering. `1.0` to disable. Defaults to `0.95`.
    * `:min_p` - Min-P filtering. `0.0` to disable. Defaults to `0.05`.
    * `:penalty_repeat` - Repetition penalty. `1.0` to disable. Defaults to `1.0`.
    * `:penalty_freq` - Frequency penalty (0.0–2.0). `0.0` to disable. Defaults to `0.0`.
    * `:penalty_present` - Presence penalty (0.0–2.0). `0.0` to disable. Defaults to `0.0`.
    * `:grammar` - GBNF grammar string for constrained generation. Defaults to `""` (none).
    * `:grammar_root` - Root rule name for grammar. Defaults to `"root"`.

  ## Errors

  Returns `{:error, :invalid_grammar}` when `:grammar` does not compile, exceeds
  1 MiB, or nests `(` groups more than 64 deep. A grammar that fails to compile
  is **not** silently dropped: a caller who asked for JSON-constrained output
  would otherwise get unconstrained output and no indication of it.

  """
  @spec create(LlamaCppEx.Model.t(), keyword()) ::
          {:ok, t()} | {:error, :invalid_grammar}
  def create(%LlamaCppEx.Model{ref: model_ref} = model, opts \\ []) do
    seed = Keyword.get(opts, :seed, :rand.uniform(1_000_000_000))
    temp = Keyword.get(opts, :temp, 0.8)
    top_k = Keyword.get(opts, :top_k, 40)
    top_p = Keyword.get(opts, :top_p, 0.95)
    min_p = Keyword.get(opts, :min_p, 0.05)
    penalty_repeat = Keyword.get(opts, :penalty_repeat, 1.0)
    penalty_freq = Keyword.get(opts, :penalty_freq, 0.0)
    penalty_present = Keyword.get(opts, :penalty_present, 0.0)
    grammar = Keyword.get(opts, :grammar, "")
    grammar_root = Keyword.get(opts, :grammar_root, "root")

    result =
      LlamaCppEx.NIF.sampler_init(
        model_ref,
        seed,
        temp / 1,
        top_k,
        top_p / 1,
        min_p / 1,
        penalty_repeat / 1,
        penalty_freq / 1,
        penalty_present / 1,
        grammar,
        grammar_root
      )

    case result do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref, model: model}}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Checks that the `:grammar` in `opts` compiles, without building a chain.

  `create/2` only discovers a bad grammar as a side effect of building the
  sampler, which is too late for `LlamaCppEx.Server`: the request is already
  admitted to a slot inside the process that owns the model, so the failure
  crashed the server rather than the request. This is the same check
  `create/2`'s NIF runs, callable before a request is queued.

  Returns `:ok` when `opts` carries no `:grammar`, or an empty one.
  """
  @spec validate_grammar(LlamaCppEx.Model.t(), keyword()) :: :ok | {:error, :invalid_grammar}
  def validate_grammar(%LlamaCppEx.Model{ref: model_ref}, opts) do
    case Keyword.get(opts, :grammar, "") do
      grammar when grammar in [nil, ""] ->
        :ok

      grammar ->
        LlamaCppEx.NIF.grammar_validate(
          model_ref,
          grammar,
          Keyword.get(opts, :grammar_root, "root")
        )
    end
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
