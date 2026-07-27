defmodule LlamaCppEx.TestSlots do
  @moduledoc false

  # The one source of slot fixtures for the tests that exercise
  # `LlamaCppEx.Server.Slots` and the batching strategies.
  #
  # There used to be two incompatible hand-rolled sets — `server_test.exs` built
  # a 4-field `:idle` map, `batch_strategy_test.exs` a 2-field `:idle` and a
  # 6-field `:prefilling` one — and neither matched a real slot. Nothing built a
  # `:generating` slot at all, so `Batch.add_decode_tokens/4` never ran under
  # test and every batch-strategy test passed against strategies whose only
  # differences live on that code path.
  #
  # The field list below is `idle_slot_fields/2` in `lib/llama_cpp_ex/server.ex`
  # plus the three fields that deliberately live outside it (`:sampler`,
  # `:t_last_used`, `:session`). Completeness is the point: a strategy or slot
  # helper that reads a field the fixture omits raises `KeyError` here instead of
  # quietly matching a stub.

  @base %{
    state: :idle,
    from: nil,
    stream_pid: nil,
    stream_ref: nil,
    monitor_ref: nil,
    reply_mode: :text,
    cache_prompt: false,
    prompt_tokens: [],
    prompt_tokens_tuple: {},
    prefill_pos: 0,
    pos: 0,
    pending_token: nil,
    pending_eog: false,
    batch_idx: -1,
    tokens_generated: 0,
    max_tokens: 0,
    accumulated_pieces: [],
    utf8_pending: "",
    t_start: nil,
    t_first_token: nil,
    n_prompt_tokens: 0,
    cached_tokens: [],
    cached_pos: 0,
    cache_scope: nil,
    generated_token_ids: [],
    n_prefix_cache_tokens: 0,
    sampler: nil,
    t_last_used: 0,
    session: nil
  }

  @doc "Every field a real slot carries, in its freshly reset state."
  def base_fields, do: @base

  @doc """
  An idle slot, `{seq_id, slot}`.

  `:cached_tokens` (an override) is the KV cache left behind by the previous
  request; `:cached_pos` is derived from it unless overridden. `:t_last_used`
  drives the LRU pick.
  """
  def idle_slot(seq_id, overrides \\ []) do
    cached = Keyword.get(overrides, :cached_tokens, [])

    build(seq_id, %{state: :idle, cached_tokens: cached, cached_pos: length(cached)}, overrides)
  end

  @doc """
  A slot part-way through prefill, `{seq_id, slot}`.

  `prefill_pos` is how many prompt tokens are already in the KV cache; `:pos`
  tracks it, matching `init_slot/9`, which sets both to the prefix-cache match
  length.
  """
  def prefilling_slot(seq_id, prompt_tokens, prefill_pos, overrides \\ []) do
    build(
      seq_id,
      %{
        state: :prefilling,
        prompt_tokens: prompt_tokens,
        prompt_tokens_tuple: List.to_tuple(prompt_tokens),
        n_prompt_tokens: length(prompt_tokens),
        prefill_pos: prefill_pos,
        pos: prefill_pos
      },
      overrides
    )
  end

  @doc """
  A generating slot, `{seq_id, slot}`: prompt fully fed, `:generated` decoded on
  top of it, and `pending_token` sampled and waiting to be fed by
  `LlamaCppEx.Server.Strategy.Batch.add_decode_tokens/4`.

  The `:generated` override is **chronological** (oldest token first), because
  that is how the model produced it. The slot field stores it reversed, exactly
  as the server builds it by prepending each token at feed time.

  `:pos`, `:prefill_pos` and `:tokens_generated` are derived rather than passed:
  the server's invariant when `add_decode_tokens/4` runs is
  `pos == n_prompt_tokens + length(generated)` — the position `pending_token`
  will occupy, and the cap `donor_prefix_match/2` applies. `pending_token: nil`
  is the one state `add_decode_tokens/4` skips, so it has to be stated
  explicitly rather than defaulted into existence.
  """
  def generating_slot(seq_id, prompt_tokens, pending_token, overrides \\ []) do
    {generated, overrides} = Keyword.pop(overrides, :generated, [])
    n_prompt = length(prompt_tokens)

    build(
      seq_id,
      %{
        state: :generating,
        prompt_tokens: prompt_tokens,
        prompt_tokens_tuple: List.to_tuple(prompt_tokens),
        n_prompt_tokens: n_prompt,
        prefill_pos: n_prompt,
        pos: n_prompt + length(generated),
        generated_token_ids: Enum.reverse(generated),
        tokens_generated: length(generated),
        max_tokens: 256,
        pending_token: pending_token
      },
      overrides
    )
  end

  @doc "The slot map out of a `{seq_id, slot}` fixture."
  def slot({_seq_id, slot}), do: slot

  defp build(seq_id, fields, overrides) do
    overrides = Map.new(overrides)

    case Map.keys(overrides) -- Map.keys(@base) do
      [] ->
        {seq_id, @base |> Map.merge(fields) |> Map.merge(overrides)}

      unknown ->
        raise ArgumentError,
              "LlamaCppEx.TestSlots: no such slot field(s): #{inspect(unknown)}. " <>
                "Real slots carry: #{@base |> Map.keys() |> Enum.sort() |> inspect()}"
    end
  end
end
