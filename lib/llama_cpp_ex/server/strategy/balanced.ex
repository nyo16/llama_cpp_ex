defmodule LlamaCppEx.Server.Strategy.Balanced do
  @moduledoc """
  Balanced batching strategy.

  Splits the token budget equally between decode and prefill operations.
  Decode tokens always use 1 token per slot, so the decode half is capped
  at the number of generating slots. The prefill half gets the remainder.

  Fair under mixed workloads where both generation latency and prefill
  throughput matter equally.
  """

  @behaviour LlamaCppEx.Server.BatchStrategy

  alias LlamaCppEx.Server.Strategy.Batch

  @impl true
  def build_batch(slots, budget, chunk_size, opts) do
    model_ref = Keyword.fetch!(opts, :model_ref)

    n_generating =
      Enum.count(slots, fn {_id, slot} ->
        slot.state == :generating and slot.pending_token != nil
      end)

    # Split budget: decode gets half (capped at generating count), prefill the rest.
    decode_budget = min(div(budget, 2), n_generating)
    prefill_budget = budget - decode_budget

    {entries, n_entries, slots, decode_remaining} =
      Batch.add_decode_tokens(slots, [], 0, decode_budget, model_ref)

    # Prefill gets its half plus any unused decode budget.
    effective_prefill_budget = prefill_budget + decode_remaining

    {entries, _n_entries, slots, _budget} =
      Batch.add_prefill_chunks(slots, entries, n_entries, effective_prefill_budget, chunk_size)

    {Enum.reverse(entries), slots}
  end
end
