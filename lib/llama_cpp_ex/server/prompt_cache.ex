defmodule LlamaCppEx.Server.PromptCache do
  @moduledoc false
  # Level-2 prompt cache for `LlamaCppEx.Server`: KV states of evicted slots kept
  # in RAM under a byte budget, FIFO.
  #
  # ## Why this is a module
  #
  # The `3b62421` decomposition moved only scheduling out of `LlamaCppEx.Server`
  # (slot selection, 88 lines; batch strategies, 206) and its commit message
  # pre-argued that the RAM cache was inseparable because it touches server state.
  # The code said otherwise: the coupling was to three fields (`ram_cache`,
  # `ram_cache_bytes`, `prompt_cache_ram_mb`) plus `ctx.ref`, and the cache's own
  # rules — the minimum-tokens floor, the f_keep restore bar, tenant scoping, FIFO
  # eviction, coverage — never read anything else. Two of its functions had to be
  # `@doc false` public just so tests could reach them, which is the usual sign
  # that a boundary wants to exist.
  #
  # So the state is a struct with the budget inside it, and the two former
  # `@doc false` escape hatches (`covers?/4`, `evict_to_budget/2`) are simply this
  # module's public API.
  #
  # ## What stays in `LlamaCppEx.Server`
  #
  # Everything that drives the NIF against a live context. `save/4` and
  # `restore/4` take `ctx_ref` and `seq_id` because serializing and restoring a
  # sequence's KV is inherently a context operation, and the context is owned by a
  # single process by invariant (see `c_src/llama_cpp_ex/llama_nif.h`) — that is
  # also why this work cannot be moved to a `Task`.
  #
  # ## Entries
  #
  # `%{tokens: [token_id], len: n, bin: binary, bytes: n, scope: term}`. `len` is
  # the number of KV positions the blob holds, which equals `length(tokens)`;
  # it is stored because that is the number every decision here compares against
  # and recomputing it per candidate per request is exactly the kind of waste the
  # perf audit found elsewhere. `scope` is the `:cache_scope` of the request that
  # produced the entry — see `covers?/4`.

  alias LlamaCppEx.Server.Slots

  # Evicted caches below this many tokens are not worth a KV-sized state copy.
  @min_tokens 32

  # Restore an entry only when the usable fraction clears this bar
  # (llama-server's f_keep heuristic) — restoring is a KV-sized memcpy.
  @min_keep 0.25

  defstruct entries: [], bytes: 0, budget_bytes: 0

  @type entry :: %{
          tokens: [integer()],
          len: non_neg_integer(),
          bin: binary(),
          bytes: non_neg_integer(),
          scope: term()
        }

  @type t :: %__MODULE__{
          entries: [entry()],
          bytes: non_neg_integer(),
          budget_bytes: non_neg_integer()
        }

  @doc "A cache with a budget of `mb` megabytes. `0` disables it."
  @spec new(non_neg_integer()) :: t()
  def new(mb) when is_integer(mb) and mb >= 0 do
    %__MODULE__{budget_bytes: mb * 1024 * 1024}
  end

  @doc "Whether the cache is enabled at all."
  @spec enabled?(t()) :: boolean()
  def enabled?(%__MODULE__{budget_bytes: 0}), do: false
  def enabled?(%__MODULE__{}), do: true

  @doc "Number of resident entries."
  @spec size(t()) :: non_neg_integer()
  def size(%__MODULE__{entries: entries}), do: length(entries)

  @doc """
  Whether an existing entry already covers `cached_pos` tokens of `cached_tokens`
  within `scope`.

  Scope is part of the key, not just the token prefix: an entry is a KV blob, so
  serving one caller's blob to another because their prompts happen to share a
  prefix is a cross-request leak. `nil` is the shared pool — the default, and only
  safe when every caller of the server is in one trust domain.
  """
  @spec covers?(t() | [entry()], [integer()], non_neg_integer(), term()) :: boolean()
  def covers?(cache_or_entries, cached_tokens, cached_pos, scope \\ nil)

  def covers?(%__MODULE__{entries: entries}, cached_tokens, cached_pos, scope) do
    covers?(entries, cached_tokens, cached_pos, scope)
  end

  def covers?(entries, cached_tokens, cached_pos, scope) when is_list(entries) do
    Enum.any?(entries, fn entry ->
      entry.scope == scope and entry.len >= cached_pos and
        Slots.common_prefix_length(cached_tokens, entry.tokens) == cached_pos
    end)
  end

  @doc """
  Offers a slot's about-to-be-destroyed KV to the cache.

  Returns the updated cache and the entries it evicted, so the caller can emit
  telemetry. A no-op when the cache is disabled, the slot's cache is too small to
  be worth a KV-sized copy, an existing entry already covers it, or the blob is
  bigger than the whole budget (degrade to disabled, never OOM).

  The checks are ordered cheapest-first on purpose: `state_seq_get_size` alone is
  ~60 µs at `n_ctx=131072`, and `state_seq_get_data` copies the whole blob, so
  neither runs until the free checks have failed to rule the save out.
  """
  @spec save(t(), reference(), non_neg_integer(), map()) ::
          {t(), saved :: entry() | nil, evicted :: [entry()]}
  def save(%__MODULE__{budget_bytes: 0} = cache, _ctx_ref, _seq_id, _slot),
    do: {cache, nil, []}

  def save(%__MODULE__{} = cache, ctx_ref, seq_id, slot) do
    with true <- slot.cached_pos >= @min_tokens,
         false <- covers?(cache, slot.cached_tokens, slot.cached_pos, slot.cache_scope),
         bytes = LlamaCppEx.NIF.state_seq_get_size(ctx_ref, seq_id),
         true <- bytes > 0 and bytes <= cache.budget_bytes,
         {:ok, bin} <- LlamaCppEx.NIF.state_seq_get_data(ctx_ref, seq_id) do
      entry = %{
        tokens: slot.cached_tokens,
        len: slot.cached_pos,
        bin: bin,
        bytes: bytes,
        scope: slot.cache_scope
      }

      cache = %{cache | entries: cache.entries ++ [entry], bytes: cache.bytes + bytes}
      {cache, evicted} = evict_to_budget(cache)
      {cache, entry, evicted}
    else
      _ -> {cache, nil, []}
    end
  end

  @doc """
  FIFO eviction down to the budget. Returns the cache and the entries dropped.

  The empty-cache clause is defensive: entries larger than the whole budget are
  never stored, so over-budget implies non-empty today.
  """
  @spec evict_to_budget(t()) :: {t(), [entry()]}
  def evict_to_budget(%__MODULE__{} = cache), do: evict_to_budget(cache, [])

  defp evict_to_budget(%__MODULE__{bytes: bytes, budget_bytes: budget} = cache, acc)
       when bytes <= budget do
    {cache, Enum.reverse(acc)}
  end

  defp evict_to_budget(%__MODULE__{entries: []} = cache, acc) do
    {%{cache | bytes: 0}, Enum.reverse(acc)}
  end

  defp evict_to_budget(%__MODULE__{entries: [evicted | rest]} = cache, acc) do
    evict_to_budget(%{cache | entries: rest, bytes: cache.bytes - evicted.bytes}, [evicted | acc])
  end

  @doc """
  Best entry for `tokens` within `scope`, or `nil`.

  Returns `{entry, lcp}`. Requires the reusable fraction to clear the f_keep bar
  (restoring is a KV-sized memcpy — not worth it for a sliver) and the unusable
  tail to be trimmable on this model: `seq_rm_kind` `:full` (hybrid GDN) cannot
  trim a partial range, so only an exact-length match is usable there.

  `lcp` is capped at `length(tokens) - 1`: the last prompt token must still be
  decoded to produce logits for the first sampled token.
  """
  @spec best_candidate(t(), [integer()], term(), atom()) :: {entry(), non_neg_integer()} | nil
  def best_candidate(%__MODULE__{entries: []}, _tokens, _scope, _seq_rm_kind), do: nil

  def best_candidate(%__MODULE__{entries: entries}, tokens, scope, seq_rm_kind) do
    max_reuse = length(tokens) - 1

    case Enum.filter(entries, &(&1.scope == scope)) do
      [] -> nil
      candidates -> pick_candidate(candidates, tokens, max_reuse, seq_rm_kind)
    end
  end

  defp pick_candidate(candidates, tokens, max_reuse, seq_rm_kind) do
    {entry, lcp} =
      candidates
      |> Enum.map(fn entry ->
        {entry, min(Slots.common_prefix_length(tokens, entry.tokens), max_reuse)}
      end)
      |> Enum.max_by(fn {_entry, lcp} -> lcp end)

    usable? =
      lcp > 0 and lcp / entry.len >= @min_keep and
        (lcp == entry.len or seq_rm_kind != :full)

    if usable?, do: {entry, lcp}
  end

  @doc """
  Restores `entry` into an (empty) `seq_id` and trims the unusable tail.

  Returns `{:ok, reusable_prefix_len}`, or `{:error, reason}` after clearing the
  sequence — a partial restore would otherwise leave garbage KV behind, which the
  next decode reads as real positions.
  """
  @spec restore(reference(), non_neg_integer(), entry(), non_neg_integer()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def restore(ctx_ref, seq_id, entry, lcp) do
    case LlamaCppEx.NIF.state_seq_set_data(ctx_ref, entry.bin, seq_id) do
      {:ok, _bytes} ->
        if lcp < entry.len do
          true = LlamaCppEx.NIF.memory_seq_rm(ctx_ref, seq_id, lcp, -1)
        end

        {:ok, lcp}

      {:error, reason} ->
        _ = LlamaCppEx.NIF.memory_seq_rm(ctx_ref, seq_id, 0, -1)
        {:error, reason}
    end
  end
end
