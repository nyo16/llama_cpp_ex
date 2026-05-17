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

  @impl true
  def build_batch(slots, budget, chunk_size, opts) do
    model_ref = Keyword.fetch!(opts, :model_ref)

    n_generating =
      Enum.count(slots, fn {_id, slot} ->
        slot.state == :generating and slot.pending_token != nil
      end)

    # Split budget: decode gets half (capped at generating count), prefill gets remainder
    decode_budget = min(div(budget, 2), n_generating)
    prefill_budget = budget - decode_budget

    # Decode tokens first (they only need 1 each)
    {entries, slots, decode_remaining} = add_decode_tokens(slots, [], decode_budget, model_ref)

    # Prefill gets its half plus any unused decode budget
    effective_prefill_budget = prefill_budget + decode_remaining

    {entries, slots, _budget} =
      add_prefill_chunks(slots, entries, effective_prefill_budget, chunk_size)

    {Enum.reverse(entries), slots}
  end

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
