Code.require_file("helpers.exs", __DIR__)

# Slot selection and prefix matching, old vs new — the Phase 7 `[perf]` fixes
# 3.1 (donor_prefix_match no longer concatenates a slot's fed history) and 3.3
# (pick_cached_slot returns the LCP so the caller stops recomputing it).
#
# No model, no NIF, no GPU: `LlamaCppEx.Server.Slots` only ever compares token
# ids. That is deliberate — these costs are microseconds inside the process that
# also runs every forward pass, so measuring them end-to-end would bury them
# under milliseconds of GPU noise. The pre-optimisation implementations are
# copied verbatim into `Bench.SlotsOld` below so both run on the same VM, same
# heap, same inputs.
#
# Run: MIX_ENV=bench mix run bench/slots.exs

alias LlamaCppEx.Server.Slots

defmodule Bench.SlotsOld do
  @moduledoc false
  # Verbatim pre-optimisation `LlamaCppEx.Server.Slots`. `common_prefix_length/2`
  # itself did not change, so it is called through from `Slots` — the only
  # differences under test are the concatenation and the recomputation.

  def pick_cached_slot(idle_slots, tokens) do
    prompt_len = length(tokens)

    {best_id, best_lcp} =
      idle_slots
      |> Enum.map(fn {id, slot} ->
        {id, Slots.common_prefix_length(tokens, slot.cached_tokens)}
      end)
      |> Enum.max_by(fn {_id, lcp} -> lcp end)

    if best_lcp / prompt_len > 0.1 do
      best_id
    else
      pick_lru_slot(idle_slots)
    end
  end

  def pick_lru_slot(idle_slots) do
    {seq_id, _} = Enum.min_by(idle_slots, fn {_id, slot} -> slot.t_last_used end)
    seq_id
  end

  def donor_prefix_match(%{state: :generating} = slot, tokens) do
    fed = slot.prompt_tokens ++ Enum.reverse(slot.generated_token_ids)
    min(Slots.common_prefix_length(tokens, fed), slot.pos)
  end
end

defmodule Bench.SlotsFixtures do
  @moduledoc false

  # A slot that has finished prefill and is 128 tokens into generation — the
  # only slot state whose fed history was being materialised.
  @generated 128

  # n_parallel: 8, the configuration the ~600µs/request figure came from.
  @n_slots 8

  def generating_slot(n) do
    prompt = Bench.Helpers.token_ids(n)

    %{
      state: :generating,
      prompt_tokens: prompt,
      # Newest-first, exactly how the server accumulates it.
      generated_token_ids: Enum.reverse(Bench.Helpers.token_ids(@generated, seed: 7)),
      pos: n + @generated,
      cached_tokens: [],
      t_last_used: 0
    }
  end

  # The common case: an unrelated request, so the walk dies a few tokens in.
  # Old code still copied the whole prompt to get there.
  def early_mismatch_tokens(n), do: List.replace_at(Bench.Helpers.token_ids(n), 8, -1)

  # The worst case: the whole prompt matches, so the generated tail is reached
  # and the new code pays for the reverse it was avoiding.
  def full_match_tokens(n) do
    Bench.Helpers.token_ids(n) ++ Bench.Helpers.token_ids(@generated, seed: 7)
  end

  # Eight idle slots, one holding the request's own prompt (a session returning
  # to its slot: the LCP clears the 0.1 threshold, so that slot wins and its
  # full-length LCP is what the old caller recomputed). The other seven hold
  # unrelated conversations that diverge immediately.
  def idle_slots(n) do
    winner = @n_slots - 3

    for id <- 0..(@n_slots - 1) do
      cached =
        if id == winner,
          do: Bench.Helpers.token_ids(n),
          else: early_mismatch_tokens(n)

      {id, %{state: :idle, cached_tokens: cached, t_last_used: id}}
    end
  end
end

# 220 is the old ceiling of `Bench.Helpers.prompts/0`, kept as the no-regression
# guard: the point of the >1k sizes is moot if the fix costs anything at the size
# everything was previously measured at.
sizes = [220, 4096, 8192, 32768]

donor_inputs =
  for n <- sizes,
      {case_label, tokens} <- [
        {"common (early mismatch)", Bench.SlotsFixtures.early_mismatch_tokens(n)},
        {"worst (full match)", Bench.SlotsFixtures.full_match_tokens(n)}
      ],
      into: %{} do
    {"#{Bench.Helpers.size_label(n)} #{case_label}",
     {Bench.SlotsFixtures.generating_slot(n), tokens}}
  end

IO.puts("Slots.donor_prefix_match/2 on a :generating slot — old vs new")
IO.puts("  cost per candidate slot; the server pays this once per busy slot per request")
IO.puts("")

Benchee.run(
  %{
    "old (prompt ++ reverse(generated))" => fn {slot, tokens} ->
      Bench.SlotsOld.donor_prefix_match(slot, tokens)
    end,
    "new (sequential walk)" => fn {slot, tokens} ->
      Slots.donor_prefix_match(slot, tokens)
    end
  },
  inputs: donor_inputs,
  warmup: 2,
  time: 10,
  memory_time: 2,
  # Reductions are the tie-breaker here: at these magnitudes wall time bounces
  # with GC and scheduler noise between runs, while the reduction count is a
  # deterministic measure of work done.
  reduction_time: 2,
  formatters: [{Benchee.Formatters.Console, extended_statistics: true}]
)

# `max_reuse` in the server is a context-derived cap; a value above every prompt
# length here keeps the `min/2` out of the comparison.
max_reuse = 1_000_000

pick_inputs =
  Map.new(sizes, fn n ->
    {Bench.Helpers.size_label(n), {Bench.SlotsFixtures.idle_slots(n), Bench.Helpers.token_ids(n)}}
  end)

IO.puts("")
IO.puts("Slots.pick_cached_slot/2 + the caller's LCP — old vs new")
IO.puts("  8 idle slots, one holding the request's own prompt; cost per request")
IO.puts("")

Benchee.run(
  %{
    "old (pick, then recompute LCP)" => fn {idle_slots, tokens} ->
      seq_id = Bench.SlotsOld.pick_cached_slot(idle_slots, tokens)
      {^seq_id, slot} = List.keyfind(idle_slots, seq_id, 0)
      min(Slots.common_prefix_length(tokens, slot.cached_tokens), max_reuse)
    end,
    "new (pick returns {seq_id, lcp})" => fn {idle_slots, tokens} ->
      {seq_id, lcp} = Slots.pick_cached_slot(idle_slots, tokens)
      {^seq_id, _slot} = List.keyfind(idle_slots, seq_id, 0)
      min(lcp, max_reuse)
    end
  },
  inputs: pick_inputs,
  warmup: 2,
  time: 10,
  memory_time: 2,
  reduction_time: 2,
  formatters: [{Benchee.Formatters.Console, extended_statistics: true}]
)
