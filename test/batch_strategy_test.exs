defmodule LlamaCppEx.Server.BatchStrategyTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Server.Slots
  alias LlamaCppEx.Server.Strategy.{Balanced, Batch, DecodeMaximal, PrefillPriority}

  import LlamaCppEx.TestSlots,
    only: [idle_slot: 1, prefilling_slot: 3, generating_slot: 3, generating_slot: 4]

  # Slot fixtures are shared with server_test.exs — see LlamaCppEx.TestSlots,
  # which mirrors `idle_slot_fields/2` in lib/llama_cpp_ex/server.ex.

  # The strategies ignore their opts; they are passed to pin that contract.
  defp opts, do: [model_ref: nil, queue_depth: 0]

  defp seq_ids(entries), do: Enum.map(entries, &elem(&1, 2))

  # --- DecodeMaximal ---

  describe "DecodeMaximal" do
    test "respects budget limit" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 50, 512, opts())
      assert length(entries) == 50
    end

    test "respects chunk_size" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 2048, 128, opts())
      assert length(entries) == 128
    end

    test "empty when all slots idle" do
      slots = Map.new([idle_slot(0), idle_slot(1)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 512, 512, opts())
      assert entries == []
    end

    test "logits flag true only on last prefill token of final chunk" do
      tokens = Enum.to_list(1..10)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 100, 100, opts())

      logits_flags = Enum.map(entries, &elem(&1, 3))
      assert List.last(logits_flags) == true
      assert Enum.count(logits_flags, & &1) == 1
    end

    test "logits flag false for incomplete chunk" do
      tokens = Enum.to_list(1..100)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      # chunk_size=30 and budget=30: first chunk of 100, not last
      {entries, _slots} = DecodeMaximal.build_batch(slots, 30, 30, opts())

      logits_flags = Enum.map(entries, &elem(&1, 3))
      assert Enum.all?(logits_flags, fn f -> f == false end)
    end

    test "handles multiple prefilling slots" do
      slots =
        Map.new([
          prefilling_slot(0, Enum.to_list(1..20), 0),
          prefilling_slot(1, Enum.to_list(101..120), 0)
        ])

      {entries, _slots} = DecodeMaximal.build_batch(slots, 100, 50, opts())

      assert entries |> seq_ids() |> Enum.uniq() |> Enum.sort() == [0, 1]
      assert length(entries) == 40
    end

    test "updates prefill_pos in returned slots" do
      tokens = Enum.to_list(1..100)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {_entries, updated_slots} = DecodeMaximal.build_batch(slots, 30, 30, opts())

      assert updated_slots[0].prefill_pos == 30
    end

    test "sets batch_idx on last chunk completion" do
      tokens = Enum.to_list(1..10)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {_entries, updated_slots} = DecodeMaximal.build_batch(slots, 100, 100, opts())

      # Last chunk completed — batch_idx should be set
      assert updated_slots[0].batch_idx >= 0
    end

    test "sets batch_idx to -1 on incomplete chunk" do
      tokens = Enum.to_list(1..100)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {_entries, updated_slots} = DecodeMaximal.build_batch(slots, 30, 30, opts())

      # Incomplete — batch_idx stays -1
      assert updated_slots[0].batch_idx == -1
    end

    test "preserves positions from prefill_pos offset" do
      tokens = Enum.to_list(1..20)
      slots = Map.new([prefilling_slot(0, tokens, 5)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 100, 100, opts())

      positions = Enum.map(entries, &elem(&1, 1))
      # Should start from position 5
      assert hd(positions) == 5
      assert List.last(positions) == 19
      assert length(entries) == 15
    end

    test "decode tokens come before prefill chunks" do
      slots =
        Map.new([
          prefilling_slot(0, Enum.to_list(1..5), 0),
          generating_slot(1, [10, 11], 12)
        ])

      {entries, _slots} = DecodeMaximal.build_batch(slots, 20, 20, opts())

      assert seq_ids(entries) == [1, 0, 0, 0, 0, 0]
      assert hd(entries) == {12, 2, 1, true}
    end

    test "decode takes the whole budget, starving prefill" do
      slots =
        Map.new(
          [prefilling_slot(0, Enum.to_list(1..100), 0)] ++
            for(id <- 1..4, do: generating_slot(id, [10], 20 + id))
        )

      {entries, updated} = DecodeMaximal.build_batch(slots, 4, 512, opts())

      assert seq_ids(entries) == [1, 2, 3, 4]
      assert updated[0].prefill_pos == 0
    end
  end

  # --- PrefillPriority ---

  describe "PrefillPriority" do
    test "respects budget limit" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = PrefillPriority.build_batch(slots, 50, 512, opts())
      assert length(entries) == 50
    end

    test "respects chunk_size" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = PrefillPriority.build_batch(slots, 2048, 128, opts())
      assert length(entries) == 128
    end

    test "empty when all idle" do
      slots = Map.new([idle_slot(0)])
      {entries, _slots} = PrefillPriority.build_batch(slots, 512, 512, opts())
      assert entries == []
    end

    test "handles multiple prefilling slots" do
      slots =
        Map.new([
          prefilling_slot(0, Enum.to_list(1..10), 0),
          prefilling_slot(1, Enum.to_list(1..10), 0)
        ])

      {entries, _slots} = PrefillPriority.build_batch(slots, 100, 50, opts())
      assert length(entries) == 20
    end

    test "prefill chunks come before decode tokens" do
      slots =
        Map.new([
          prefilling_slot(0, Enum.to_list(1..5), 0),
          generating_slot(1, [10, 11], 12)
        ])

      {entries, _slots} = PrefillPriority.build_batch(slots, 20, 20, opts())

      assert seq_ids(entries) == [0, 0, 0, 0, 0, 1]
      assert List.last(entries) == {12, 2, 1, true}
    end

    test "prefill takes the whole budget, starving decode" do
      slots =
        Map.new(
          [prefilling_slot(0, Enum.to_list(1..100), 0)] ++
            for(id <- 1..4, do: generating_slot(id, [10], 20 + id))
        )

      {entries, updated} = PrefillPriority.build_batch(slots, 4, 512, opts())

      assert seq_ids(entries) == [0, 0, 0, 0]
      assert updated[0].prefill_pos == 4
      # Nothing was fed for the generating slots, so their history is untouched.
      assert Enum.all?(1..4, &(updated[&1].generated_token_ids == []))
    end
  end

  # --- Balanced ---

  describe "Balanced" do
    test "respects budget limit" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = Balanced.build_batch(slots, 50, 512, opts())
      assert length(entries) == 50
    end

    test "empty when all idle" do
      slots = Map.new([idle_slot(0)])
      {entries, _slots} = Balanced.build_batch(slots, 512, 512, opts())
      assert entries == []
    end

    test "prefill-only uses full budget when no decode" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..20), 0)])
      {entries, _slots} = Balanced.build_batch(slots, 100, 100, opts())
      # No generating slots, so all budget goes to prefill
      assert length(entries) == 20
    end

    test "caps decode at half the budget and gives the rest to prefill" do
      slots =
        Map.new(
          [prefilling_slot(0, Enum.to_list(1..100), 0)] ++
            for(id <- 1..4, do: generating_slot(id, [10], 20 + id))
        )

      {entries, updated} = Balanced.build_batch(slots, 4, 512, opts())

      # div(4, 2) = 2 decode slots (lowest seq_ids), 2 tokens of prefill.
      assert seq_ids(entries) == [1, 2, 0, 0]
      assert updated[0].prefill_pos == 2
      # Slots 1-2 were fed; slots 3-4 were not, so their history is untouched.
      assert updated[1].generated_token_ids == [21]
      assert updated[2].generated_token_ids == [22]
      assert updated[3].generated_token_ids == []
      assert updated[4].generated_token_ids == []
    end

    test "unused decode budget rolls into prefill" do
      # One generating slot: decode_budget = min(div(10, 2), 1) = 1, so prefill
      # gets 9 and the single unused decode token is not lost.
      slots =
        Map.new([
          prefilling_slot(0, Enum.to_list(1..100), 0),
          generating_slot(1, [10], 11)
        ])

      {entries, updated} = Balanced.build_batch(slots, 10, 512, opts())

      assert length(entries) == 10
      assert updated[0].prefill_pos == 9
    end
  end

  # The three strategies used to be indistinguishable under test: no fixture
  # produced a :generating slot, so `Batch.add_decode_tokens/4` never ran and
  # every assertion exercised the shared prefill path. This is the scenario that
  # separates them — one prefilling slot with more prompt than budget, four
  # generating slots, and a budget of 4.
  describe "the three strategies are distinguishable" do
    setup do
      slots =
        Map.new(
          [prefilling_slot(0, Enum.to_list(1..100), 0)] ++
            for(id <- 1..4, do: generating_slot(id, [10], 20 + id))
        )

      %{slots: slots}
    end

    test "each splits a budget of 4 differently", %{slots: slots} do
      split = fn strategy ->
        {entries, _} = strategy.build_batch(slots, 4, 512, opts())
        seq_ids(entries)
      end

      assert split.(DecodeMaximal) == [1, 2, 3, 4]
      assert split.(Balanced) == [1, 2, 0, 0]
      assert split.(PrefillPriority) == [0, 0, 0, 0]
    end

    test "no two strategies agree", %{slots: slots} do
      results =
        for strategy <- [DecodeMaximal, Balanced, PrefillPriority] do
          strategy.build_batch(slots, 4, 512, opts())
        end

      assert length(Enum.uniq(results)) == 3
    end
  end

  # --- Batch.add_decode_tokens/4 (was never executed by any test) ---

  describe "add_decode_tokens/4" do
    test "feeds one token per generating slot at its current position" do
      slots = Map.new([generating_slot(0, [1, 2, 3], 7, generated: [4, 5])])

      {entries, n_entries, slots, budget} = Batch.add_decode_tokens(slots, [], 0, 8)

      # pos is n_prompt_tokens + already-generated = 5, and logits are always
      # requested for a decode token.
      assert entries == [{7, 5, 0, true}]
      assert n_entries == 1
      assert budget == 7
      assert slots[0].batch_idx == 0
    end

    test "records the fed token in generated_token_ids, newest first" do
      slots = Map.new([generating_slot(0, [1, 2], 9, generated: [3, 4])])

      {_entries, _n, slots, _budget} = Batch.add_decode_tokens(slots, [], 0, 8)

      # The invariant the prefix cache depends on: generated_token_ids tracks
      # exactly what is in the KV cache, updated at feed time.
      assert slots[0].generated_token_ids == [9, 4, 3]
    end

    test "skips generating slots with no pending token" do
      slots =
        Map.new([
          generating_slot(0, [1], nil, generated: [2]),
          generating_slot(1, [1], 5, generated: [2])
        ])

      {entries, n_entries, slots, _budget} = Batch.add_decode_tokens(slots, [], 0, 8)

      assert seq_ids(entries) == [1]
      assert n_entries == 1
      assert slots[0].generated_token_ids == [2]
      assert slots[0].batch_idx == -1
    end

    test "skips idle and prefilling slots" do
      slots =
        Map.new([
          idle_slot(0),
          prefilling_slot(1, [1, 2, 3], 1),
          generating_slot(2, [1], 9)
        ])

      {entries, _n, _slots, budget} = Batch.add_decode_tokens(slots, [], 0, 8)

      assert seq_ids(entries) == [2]
      assert budget == 7
    end

    test "serves the lowest seq_ids first when the budget is short" do
      slots = Map.new(for id <- 0..3, do: generating_slot(id, [1], 100 + id))

      {entries, n_entries, slots, budget} = Batch.add_decode_tokens(slots, [], 0, 2)

      assert seq_ids(entries) |> Enum.sort() == [0, 1]
      assert n_entries == 2
      assert budget == 0
      assert slots[2].generated_token_ids == []
      assert slots[3].batch_idx == -1
    end

    test "a zero or negative budget feeds nothing" do
      slots = Map.new([generating_slot(0, [1], 9)])

      assert {[], 0, ^slots, 0} = Batch.add_decode_tokens(slots, [], 0, 0)
      assert {[], 0, ^slots, -3} = Batch.add_decode_tokens(slots, [], 0, -3)
    end

    test "batch_idx continues from the entries already accumulated" do
      slots = Map.new([generating_slot(0, [1], 9), generating_slot(1, [1], 8)])
      existing = [{0, 0, 9, false}, {0, 1, 9, false}]

      {entries, n_entries, slots, _budget} =
        Batch.add_decode_tokens(slots, existing, length(existing), 8)

      # Entries are prepended and reversed once by the caller, so the batch index
      # of an appended entry is the n_entries count at the time it was added.
      assert n_entries == 4
      assert slots[0].batch_idx == 2
      assert slots[1].batch_idx == 3
      assert length(entries) == 4
    end
  end

  # --- common_prefix_length ---

  describe "common_prefix_length/2" do
    test "matching prefix" do
      assert Slots.common_prefix_length([1, 2, 3, 4], [1, 2, 3, 5]) == 3
    end

    test "identical lists" do
      assert Slots.common_prefix_length([1, 2, 3], [1, 2, 3]) == 3
    end

    test "no match" do
      assert Slots.common_prefix_length([1, 2, 3], [4, 5, 6]) == 0
    end

    test "first empty" do
      assert Slots.common_prefix_length([], [1, 2]) == 0
    end

    test "second empty" do
      assert Slots.common_prefix_length([1, 2], []) == 0
    end

    test "both empty" do
      assert Slots.common_prefix_length([], []) == 0
    end

    test "different lengths with match" do
      assert Slots.common_prefix_length([1, 2, 3, 4, 5], [1, 2]) == 2
    end
  end
end
