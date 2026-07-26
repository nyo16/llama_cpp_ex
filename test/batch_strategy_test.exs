defmodule LlamaCppEx.Server.BatchStrategyTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.Server.Slots
  alias LlamaCppEx.Server.Strategy.{Balanced, DecodeMaximal, PrefillPriority}

  # We need a mock model_ref for token_to_piece calls.
  # Since strategies call NIF.token_to_piece, we need a real model for integration.
  # For pure unit tests, we test with only prefilling slots (no decode tokens needed).

  # --- Helpers ---

  defp prefilling_slot(seq_id, tokens, prefill_pos) do
    {seq_id,
     %{
       state: :prefilling,
       prompt_tokens: tokens,
       prompt_tokens_tuple: List.to_tuple(tokens),
       n_prompt_tokens: length(tokens),
       prefill_pos: prefill_pos,
       batch_idx: -1
     }}
  end

  defp idle_slot(seq_id) do
    {seq_id, %{state: :idle, batch_idx: -1}}
  end

  # Use opts without model_ref for prefill-only tests (no decode tokens)
  defp prefill_only_opts, do: [model_ref: nil, queue_depth: 0]

  # --- DecodeMaximal ---

  describe "DecodeMaximal" do
    test "respects budget limit" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 50, 512, prefill_only_opts())
      assert length(entries) == 50
    end

    test "respects chunk_size" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 2048, 128, prefill_only_opts())
      assert length(entries) == 128
    end

    test "empty when all slots idle" do
      slots = Map.new([idle_slot(0), idle_slot(1)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 512, 512, prefill_only_opts())
      assert entries == []
    end

    test "logits flag true only on last prefill token of final chunk" do
      tokens = Enum.to_list(1..10)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 100, 100, prefill_only_opts())

      logits_flags = Enum.map(entries, &elem(&1, 3))
      assert List.last(logits_flags) == true
      assert Enum.count(logits_flags, & &1) == 1
    end

    test "logits flag false for incomplete chunk" do
      tokens = Enum.to_list(1..100)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      # chunk_size=30 and budget=30: first chunk of 100, not last
      {entries, _slots} = DecodeMaximal.build_batch(slots, 30, 30, prefill_only_opts())

      logits_flags = Enum.map(entries, &elem(&1, 3))
      assert Enum.all?(logits_flags, fn f -> f == false end)
    end

    test "handles multiple prefilling slots" do
      slots =
        Map.new([
          prefilling_slot(0, Enum.to_list(1..20), 0),
          prefilling_slot(1, Enum.to_list(101..120), 0)
        ])

      {entries, _slots} = DecodeMaximal.build_batch(slots, 100, 50, prefill_only_opts())

      seq_ids = entries |> Enum.map(&elem(&1, 2)) |> Enum.uniq() |> Enum.sort()
      assert seq_ids == [0, 1]
      assert length(entries) == 40
    end

    test "updates prefill_pos in returned slots" do
      tokens = Enum.to_list(1..100)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {_entries, updated_slots} = DecodeMaximal.build_batch(slots, 30, 30, prefill_only_opts())

      assert updated_slots[0].prefill_pos == 30
    end

    test "sets batch_idx on last chunk completion" do
      tokens = Enum.to_list(1..10)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {_entries, updated_slots} = DecodeMaximal.build_batch(slots, 100, 100, prefill_only_opts())

      # Last chunk completed — batch_idx should be set
      assert updated_slots[0].batch_idx >= 0
    end

    test "sets batch_idx to -1 on incomplete chunk" do
      tokens = Enum.to_list(1..100)
      slots = Map.new([prefilling_slot(0, tokens, 0)])
      {_entries, updated_slots} = DecodeMaximal.build_batch(slots, 30, 30, prefill_only_opts())

      # Incomplete — batch_idx stays -1
      assert updated_slots[0].batch_idx == -1
    end

    test "preserves positions from prefill_pos offset" do
      tokens = Enum.to_list(1..20)
      slots = Map.new([prefilling_slot(0, tokens, 5)])
      {entries, _slots} = DecodeMaximal.build_batch(slots, 100, 100, prefill_only_opts())

      positions = Enum.map(entries, &elem(&1, 1))
      # Should start from position 5
      assert hd(positions) == 5
      assert List.last(positions) == 19
      assert length(entries) == 15
    end
  end

  # --- PrefillPriority ---

  describe "PrefillPriority" do
    test "respects budget limit" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = PrefillPriority.build_batch(slots, 50, 512, prefill_only_opts())
      assert length(entries) == 50
    end

    test "respects chunk_size" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = PrefillPriority.build_batch(slots, 2048, 128, prefill_only_opts())
      assert length(entries) == 128
    end

    test "empty when all idle" do
      slots = Map.new([idle_slot(0)])
      {entries, _slots} = PrefillPriority.build_batch(slots, 512, 512, prefill_only_opts())
      assert entries == []
    end

    test "handles multiple prefilling slots" do
      slots =
        Map.new([
          prefilling_slot(0, Enum.to_list(1..10), 0),
          prefilling_slot(1, Enum.to_list(1..10), 0)
        ])

      {entries, _slots} = PrefillPriority.build_batch(slots, 100, 50, prefill_only_opts())
      assert length(entries) == 20
    end
  end

  # --- Balanced ---

  describe "Balanced" do
    test "respects budget limit" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..1000), 0)])
      {entries, _slots} = Balanced.build_batch(slots, 50, 512, prefill_only_opts())
      assert length(entries) == 50
    end

    test "empty when all idle" do
      slots = Map.new([idle_slot(0)])
      {entries, _slots} = Balanced.build_batch(slots, 512, 512, prefill_only_opts())
      assert entries == []
    end

    test "prefill-only uses full budget when no decode" do
      slots = Map.new([prefilling_slot(0, Enum.to_list(1..20), 0)])
      {entries, _slots} = Balanced.build_batch(slots, 100, 100, prefill_only_opts())
      # No generating slots, so all budget goes to prefill
      assert length(entries) == 20
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
