defmodule LlamaCppEx.Server.Strategy.Batch do
  @moduledoc """
  Shared batch-assembly helpers used by the batching strategies.

  The strategies (`DecodeMaximal`, `PrefillPriority`, `Balanced`) only differ in
  the order and budget split between decode and prefill — the per-slot assembly
  of decode tokens and prefill chunks is identical, so it lives here.

  ## Performance notes

  These helpers run on every tick of the server loop, once per active slot, and
  the prefill helper runs once per token of every prompt. To keep the loop linear
  in the number of entries:

    * A running `n_entries` count is threaded through the accumulators instead of
      calling `length/1` on the growing `entries` list. Entries are prepended and
      reversed once by the caller, so a freshly appended entry's final batch index
      is exactly the current `n_entries`.
    * Prompt length comes from the cached `slot.n_prompt_tokens` rather than
      `length(slot.prompt_tokens)`.
    * Prefill chunks are sliced from `slot.prompt_tokens_tuple` (O(1) random
      access) rather than `Enum.slice/3` on a list (O(prefill_pos)).

  All helpers take and return the 4-tuple `{entries, n_entries, slots, budget}`.
  """

  @doc """
  Adds one decode token for each generating slot (lowest seq_id first) until the
  budget is exhausted. Streams each token piece to the slot's subscriber.
  """
  def add_decode_tokens(slots, entries, n_entries, budget, model_ref) do
    generating_slots =
      slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :generating and slot.pending_token != nil
      end)
      |> Enum.sort_by(&elem(&1, 0))

    Enum.reduce(generating_slots, {entries, n_entries, slots, budget}, fn
      {_seq_id, _slot}, {entries, n_entries, slots, budget} when budget <= 0 ->
        {entries, n_entries, slots, budget}

      {seq_id, _slot}, {entries, n_entries, slots, budget} ->
        slot = slots[seq_id]
        token = slot.pending_token

        piece = LlamaCppEx.NIF.token_to_piece(model_ref, token)

        if slot.stream_pid && slot.stream_ref do
          send(slot.stream_pid, {slot.stream_ref, {:token, piece}})
        end

        slot = %{
          slot
          | accumulated_pieces: [piece | slot.accumulated_pieces],
            batch_idx: n_entries,
            tokens_generated: slot.tokens_generated + 1,
            generated_token_ids: [token | slot.generated_token_ids]
        }

        entry = {token, slot.pos, seq_id, true}
        slots = Map.put(slots, seq_id, slot)

        {[entry | entries], n_entries + 1, slots, budget - 1}
    end)
  end

  @doc """
  Fills the remaining budget with prefill chunks for each prefilling slot
  (lowest seq_id first). Only the last token of a slot's final chunk requests
  logits.
  """
  def add_prefill_chunks(slots, entries, n_entries, budget, chunk_size) do
    prefilling_slots =
      slots
      |> Enum.filter(fn {_id, slot} -> slot.state == :prefilling end)
      |> Enum.sort_by(&elem(&1, 0))

    Enum.reduce(prefilling_slots, {entries, n_entries, slots, budget}, fn
      {_seq_id, _slot}, {entries, n_entries, slots, budget} when budget <= 0 ->
        {entries, n_entries, slots, budget}

      {seq_id, _slot}, {entries, n_entries, slots, budget} ->
        slot = slots[seq_id]
        remaining = slot.n_prompt_tokens - slot.prefill_pos
        chunk_len = min(budget, min(chunk_size, remaining))
        is_last_chunk = slot.prefill_pos + chunk_len >= slot.n_prompt_tokens

        tuple = slot.prompt_tokens_tuple

        # Build this chunk's entries in O(chunk_len): elem/2 is O(1) on a tuple,
        # and n_entries gives each entry's final batch index without length/1.
        {entries, n_entries, last_batch_idx} =
          Enum.reduce(0..(chunk_len - 1)//1, {entries, n_entries, -1}, fn i,
                                                                          {entries, n_entries,
                                                                           _last} ->
            pos = slot.prefill_pos + i
            token = elem(tuple, pos)
            logits = is_last_chunk and i == chunk_len - 1
            entry = {token, pos, seq_id, logits}
            {[entry | entries], n_entries + 1, n_entries}
          end)

        batch_idx = if is_last_chunk, do: last_batch_idx, else: -1
        slot = %{slot | batch_idx: batch_idx, prefill_pos: slot.prefill_pos + chunk_len}

        slots = Map.put(slots, seq_id, slot)
        {entries, n_entries, slots, budget - chunk_len}
    end)
  end
end
