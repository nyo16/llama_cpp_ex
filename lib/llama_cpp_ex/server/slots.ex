defmodule LlamaCppEx.Server.Slots do
  @moduledoc false
  # Slot selection and prefix matching for `LlamaCppEx.Server`.
  #
  # Every function here is state-free: they take slot maps, token lists or the
  # session map, never the server's state struct. That is what makes them safe
  # to unit-test directly, and it is why the RAM prompt-cache helpers stayed in
  # `LlamaCppEx.Server` — those take and return the state struct and emit the
  # server's telemetry, so moving them would relocate code without decoupling
  # anything.
  #
  # `idle_slots` is always a list of `{seq_id, slot}` tuples.

  @doc """
  Length of the longest common prefix of two token lists.
  """
  def common_prefix_length(a, b), do: common_prefix_length(a, b, 0)

  defp common_prefix_length([x | a], [x | b], n), do: common_prefix_length(a, b, n + 1)
  defp common_prefix_length(_, _, n), do: n

  @doc """
  Session affinity: the slot that served this session last, if it is idle.

  Overrides the similarity rule — the session's cache lives there. Takes the
  server's `sessions` map (session => seq_id) rather than the whole state.
  """
  def session_slot_if_idle(_sessions, nil, _idle_slots), do: nil

  def session_slot_if_idle(sessions, session, idle_slots) do
    with seq_id when seq_id != nil <- Map.get(sessions, session),
         true <- Enum.any?(idle_slots, fn {id, _} -> id == seq_id end) do
      seq_id
    else
      _ -> nil
    end
  end

  @doc """
  llama-server's slot-pick rule (server-context.cpp).

  Reuses the slot with the best cached-prefix similarity only when it clears a
  threshold (LCP/prompt_len > 0.1); otherwise takes the least-recently-used idle
  slot, so a tiny unrelated request doesn't evict a valuable long cache.
  """
  def pick_cached_slot(idle_slots, tokens) do
    prompt_len = length(tokens)

    {best_id, best_lcp} =
      idle_slots
      |> Enum.map(fn {id, slot} -> {id, common_prefix_length(tokens, slot.cached_tokens)} end)
      |> Enum.max_by(fn {_id, lcp} -> lcp end)

    if best_lcp / prompt_len > 0.1 do
      best_id
    else
      pick_lru_slot(idle_slots)
    end
  end

  @doc """
  The least-recently-used idle slot.
  """
  def pick_lru_slot(idle_slots) do
    {seq_id, _} = Enum.min_by(idle_slots, fn {_id, slot} -> slot.t_last_used end)
    seq_id
  end

  @doc """
  How many leading tokens of `tokens` this slot already has in its KV cache.

  Only tokens actually fed to the model count, which is why each slot state is
  bounded differently.
  """
  def donor_prefix_match(%{state: :idle} = slot, tokens) do
    common_prefix_length(tokens, slot.cached_tokens)
  end

  def donor_prefix_match(%{state: :prefilling} = slot, tokens) do
    # Only positions 0..prefill_pos-1 are in the KV so far.
    min(common_prefix_length(tokens, slot.prompt_tokens), slot.prefill_pos)
  end

  def donor_prefix_match(%{state: :generating} = slot, tokens) do
    fed = slot.prompt_tokens ++ Enum.reverse(slot.generated_token_ids)
    min(common_prefix_length(tokens, fed), slot.pos)
  end
end
