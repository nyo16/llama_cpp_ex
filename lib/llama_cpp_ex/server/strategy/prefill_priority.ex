defmodule LlamaCppEx.Server.Strategy.PrefillPriority do
  @moduledoc """
  Prefill-priority batching strategy.

  Prefill chunks are added to the batch first, decode tokens fill the
  remaining budget. This prioritizes getting new requests through prefill
  quickly, which is optimal for batch processing workloads where overall
  throughput matters more than per-request generation latency.
  """

  @behaviour LlamaCppEx.Server.BatchStrategy

  alias LlamaCppEx.Server.Strategy.Batch

  @impl true
  def build_batch(slots, budget, chunk_size, opts) do
    model_ref = Keyword.fetch!(opts, :model_ref)

    # Prefill chunks first (priority), then decode tokens fill remaining budget.
    {entries, n_entries, slots, budget} =
      Batch.add_prefill_chunks(slots, [], 0, budget, chunk_size)

    {entries, _n_entries, slots, _budget} =
      Batch.add_decode_tokens(slots, entries, n_entries, budget, model_ref)

    {Enum.reverse(entries), slots}
  end
end
