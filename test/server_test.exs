defmodule LlamaCppEx.ServerTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Server

  # --- Pure slot/cache logic (no model files needed) ---

  defp idle_slot(seq_id, cached_tokens, t_last_used) do
    {seq_id,
     %{
       state: :idle,
       cached_tokens: cached_tokens,
       cached_pos: length(cached_tokens),
       t_last_used: t_last_used
     }}
  end

  describe "pick_cached_slot/2 (similarity threshold + LRU fallback)" do
    test "picks the slot with the best match when it clears the 10% threshold" do
      slots = [
        idle_slot(0, [1, 2, 3, 4, 5, 6, 7, 8], 100),
        idle_slot(1, [1, 2, 9, 9], 200)
      ]

      # 8/10 of the prompt matches slot 0
      assert Server.pick_cached_slot(slots, [1, 2, 3, 4, 5, 6, 7, 8, 20, 21]) == 0
    end

    test "falls back to LRU when the best match is below the threshold" do
      # Slot 1 holds a long valuable cache; the new prompt barely matches it.
      slots = [
        idle_slot(0, [], 100),
        idle_slot(1, Enum.to_list(1..200), 200)
      ]

      prompt = [1] ++ Enum.to_list(900..950)
      # 1/52 match < 0.1 → LRU (slot 0, oldest) — protects slot 1's cache.
      assert Server.pick_cached_slot(slots, prompt) == 0
    end

    test "pick_lru_slot returns the least recently used" do
      slots = [idle_slot(0, [], 300), idle_slot(1, [], 100), idle_slot(2, [], 200)]
      assert Server.pick_lru_slot(slots) == 1
    end
  end

  describe "session_slot_if_idle/3" do
    test "returns the session's slot when idle" do
      state = %{sessions: %{"conv-a" => 1}}
      idle = [idle_slot(0, [], 0), idle_slot(1, [], 0)]
      assert Server.session_slot_if_idle(state, "conv-a", idle) == 1
    end

    test "returns nil when the session's slot is busy" do
      state = %{sessions: %{"conv-a" => 1}}
      idle = [idle_slot(0, [], 0)]
      assert Server.session_slot_if_idle(state, "conv-a", idle) == nil
    end

    test "returns nil for unknown or nil sessions" do
      state = %{sessions: %{}}
      idle = [idle_slot(0, [], 0)]
      assert Server.session_slot_if_idle(state, "ghost", idle) == nil
      assert Server.session_slot_if_idle(state, nil, idle) == nil
    end
  end

  describe "donor_prefix_match/2 (only fed tokens count)" do
    test "idle donor matches against its cached tokens" do
      slot = %{state: :idle, cached_tokens: [1, 2, 3, 4]}
      assert Server.donor_prefix_match(slot, [1, 2, 3, 9]) == 3
    end

    test "prefilling donor is capped at prefill_pos" do
      slot = %{state: :prefilling, prompt_tokens: [1, 2, 3, 4, 5, 6], prefill_pos: 2}
      # Prompt matches 6 tokens, but only 2 are in the KV so far.
      assert Server.donor_prefix_match(slot, [1, 2, 3, 4, 5, 6]) == 2
    end

    test "generating donor is capped at fed position" do
      slot = %{
        state: :generating,
        prompt_tokens: [1, 2, 3],
        generated_token_ids: [5, 4],
        pos: 4
      }

      # Fed history is [1, 2, 3, 4, 5] but only pos=4 tokens are in KV.
      assert Server.donor_prefix_match(slot, [1, 2, 3, 4, 5, 6]) == 4
    end
  end

  describe "RAM cache bookkeeping" do
    defp entry(tokens, bytes), do: %{tokens: tokens, len: length(tokens), bytes: bytes, bin: ""}

    test "ram_cache_covers?/3 detects a covering entry" do
      entries = [entry([1, 2, 3, 4, 5], 100)]
      assert Server.ram_cache_covers?(entries, [1, 2, 3], 3)
      refute Server.ram_cache_covers?(entries, [1, 9, 3], 3)
      refute Server.ram_cache_covers?(entries, [1, 2, 3, 4, 5, 6, 7], 7)
    end

    test "evict_ram_cache_to_budget/2 drops oldest entries first" do
      state = %{
        ram_cache: [entry([1], 400), entry([2], 300), entry([3], 200)],
        ram_cache_bytes: 900
      }

      result = Server.evict_ram_cache_to_budget(state, 500)

      assert [%{tokens: [2]}, %{tokens: [3]}] = result.ram_cache
      assert result.ram_cache_bytes == 500
    end

    test "evict_ram_cache_to_budget/2 is a no-op within budget" do
      state = %{ram_cache: [entry([1], 100)], ram_cache_bytes: 100}
      assert Server.evict_ram_cache_to_budget(state, 100) == state
    end

    test "evict_ram_cache_to_budget/2 survives an inconsistent empty cache" do
      state = %{ram_cache: [], ram_cache_bytes: 999}
      assert %{ram_cache: [], ram_cache_bytes: 0} = Server.evict_ram_cache_to_budget(state, 10)
    end
  end
end
