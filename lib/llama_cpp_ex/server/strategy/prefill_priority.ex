defmodule LlamaCppEx.Server.Strategy.PrefillPriority do
  @moduledoc """
  Prefill-priority batching strategy.

  Prefill chunks are added to the batch first, decode tokens fill the
  remaining budget. This prioritizes getting new requests through prefill
  quickly, which is optimal for batch processing workloads where overall
  throughput matters more than per-request generation latency.
  """

  @behaviour LlamaCppEx.Server.BatchStrategy

  @impl true
  def build_batch(slots, budget, chunk_size, opts) do
    model_ref = Keyword.fetch!(opts, :model_ref)

    # Prefill chunks first (priority)
    {entries, slots, budget} = add_prefill_chunks(slots, [], budget, chunk_size)

    # Decode tokens fill remaining budget
    {entries, slots, _budget} = add_decode_tokens(slots, entries, budget, model_ref)

    {Enum.reverse(entries), slots}
  end

  # Reuse the shared helpers — same logic, just different order
  defp add_decode_tokens(slots, entries, budget, model_ref) do
    generating_slots =
      slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :generating and slot.pending_token != nil
      end)
      |> Enum.sort_by(&elem(&1, 0))

    Enum.reduce(generating_slots, {entries, slots, budget}, fn {seq_id, _slot},
                                                               {entries, slots, budget} ->
      if budget <= 0 do
        {entries, slots, budget}
      else
        slot = slots[seq_id]
        token = slot.pending_token

        piece = LlamaCppEx.NIF.token_to_piece(model_ref, token)

        if slot.stream_pid && slot.stream_ref do
          send(slot.stream_pid, {slot.stream_ref, {:token, piece}})
        end

        batch_idx = length(entries)

        slot = %{
          slot
          | accumulated_text: slot.accumulated_text <> piece,
            batch_idx: batch_idx,
            tokens_generated: slot.tokens_generated + 1,
            generated_token_ids: [token | slot.generated_token_ids]
        }

        entry = {token, slot.pos, seq_id, true}
        slots = Map.put(slots, seq_id, slot)

        {[entry | entries], slots, budget - 1}
      end
    end)
  end

  defp add_prefill_chunks(slots, entries, budget, chunk_size) do
    prefilling_slots =
      slots
      |> Enum.filter(fn {_id, slot} -> slot.state == :prefilling end)
      |> Enum.sort_by(&elem(&1, 0))

    Enum.reduce(prefilling_slots, {entries, slots, budget}, fn {seq_id, _slot},
                                                               {entries, slots, budget} ->
      if budget <= 0 do
        {entries, slots, budget}
      else
        slot = slots[seq_id]
        remaining = length(slot.prompt_tokens) - slot.prefill_pos
        chunk_len = min(budget, min(chunk_size, remaining))
        is_last_chunk = slot.prefill_pos + chunk_len >= length(slot.prompt_tokens)

        chunk_tokens = Enum.slice(slot.prompt_tokens, slot.prefill_pos, chunk_len)

        {new_entries, last_batch_idx} =
          chunk_tokens
          |> Enum.with_index()
          |> Enum.reduce({entries, -1}, fn {token, i}, {entries, _last_idx} ->
            pos = slot.prefill_pos + i
            batch_idx = length(entries)
            is_last_token_of_last_chunk = is_last_chunk and i == chunk_len - 1
            logits = is_last_token_of_last_chunk
            entry = {token, pos, seq_id, logits}
            {[entry | entries], batch_idx}
          end)

        slot =
          if is_last_chunk do
            %{slot | batch_idx: last_batch_idx, prefill_pos: slot.prefill_pos + chunk_len}
          else
            %{slot | batch_idx: -1, prefill_pos: slot.prefill_pos + chunk_len}
          end

        slots = Map.put(slots, seq_id, slot)
        {new_entries, slots, budget - chunk_len}
      end
    end)
  end
end
