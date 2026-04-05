defmodule LlamaCppEx.Server.BatchStrategy do
  @moduledoc """
  Behavior for batch building strategies.

  A strategy decides how to allocate the token budget between
  decode tokens (generation) and prefill chunks each tick.

  ## Built-in Strategies

    * `LlamaCppEx.Server.Strategy.DecodeMaximal` - Decode tokens first, prefill fills
      remaining budget. Best for interactive use (lowest generation latency). **Default.**
    * `LlamaCppEx.Server.Strategy.PrefillPriority` - Prefill chunks first, decode fills
      remaining budget. Best for batch processing (highest throughput).
    * `LlamaCppEx.Server.Strategy.Balanced` - Equal budget split between decode and
      prefill. Fair under mixed workloads.

  ## Custom Strategies

  Implement the `c:build_batch/4` callback:

      defmodule MyStrategy do
        @behaviour LlamaCppEx.Server.BatchStrategy

        @impl true
        def build_batch(slots, budget, chunk_size, opts) do
          # Return {entries, updated_slots}
        end
      end

  """

  @type entry ::
          {token_id :: integer(), pos :: integer(), seq_id :: integer(), logits :: boolean()}

  @doc """
  Build a batch of entries from the current slot state.

  Returns `{entries, updated_slots}` where entries is a list of
  `{token_id, pos, seq_id, logits}` tuples in forward order (will be
  reversed by the caller).

  ## Parameters

    * `slots` - Map of seq_id to slot state maps.
    * `budget` - Maximum tokens allowed in this batch (`n_batch`).
    * `chunk_size` - Maximum prefill tokens per slot per tick.
    * `opts` - Additional context:
      * `:queue_depth` - Number of requests waiting for a slot.
      * `:model_ref` - Model reference for detokenization.

  """
  @callback build_batch(
              slots :: %{non_neg_integer() => map()},
              budget :: pos_integer(),
              chunk_size :: pos_integer(),
              opts :: keyword()
            ) :: {entries :: [entry()], updated_slots :: %{non_neg_integer() => map()}}
end
