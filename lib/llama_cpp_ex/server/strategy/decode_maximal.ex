defmodule LlamaCppEx.Server.Strategy.DecodeMaximal do
  @moduledoc """
  Decode-maximal batching strategy.

  Decode tokens (one per generating slot) are always added to the batch first.
  They represent active generation that users are waiting on, so they get priority.
  Remaining budget is filled with prefill chunks.

  This is the default strategy and optimal for interactive use where low
  generation latency matters most.
  """

  @behaviour LlamaCppEx.Server.BatchStrategy

  alias LlamaCppEx.Server.Strategy.Batch

  @impl true
  def build_batch(slots, budget, chunk_size, opts) do
    model_ref = Keyword.fetch!(opts, :model_ref)

    # Decode tokens first (priority), then prefill chunks fill remaining budget.
    {entries, n_entries, slots, budget} =
      Batch.add_decode_tokens(slots, [], 0, budget, model_ref)

    {entries, _n_entries, slots, _budget} =
      Batch.add_prefill_chunks(slots, entries, n_entries, budget, chunk_size)

    {Enum.reverse(entries), slots}
  end
end
