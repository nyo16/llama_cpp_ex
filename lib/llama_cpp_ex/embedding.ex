defmodule LlamaCppEx.Embedding do
  @moduledoc "Generate embeddings from text using an embedding model."

  alias LlamaCppEx.{Context, Model, Tokenizer}

  @type t :: [float()]

  # Default cap on sequences packed into a single decode batch, bounding the
  # context's n_seq_max and KV-cache fan-out.
  @default_max_batch_sequences 64

  @doc """
  Computes an embedding for a single text.

  ## Options

    * `:n_ctx` - Context size. Defaults to `2048`.
    * `:pooling_type` - Pooling type. Defaults to `:unspecified` (model's default).
      Values: `:unspecified`, `:none`, `:mean`, `:cls`, `:last`.
    * `:normalize` - Normalization mode. `2` = L2 (default), `0` = max-abs, `-1` = none.

  """
  @spec embed(Model.t(), String.t(), keyword()) :: {:ok, t()} | {:error, String.t()}
  def embed(%Model{} = model, text, opts \\ []) when is_binary(text) do
    n_ctx = Keyword.get(opts, :n_ctx, 2048)
    pooling_type = Keyword.get(opts, :pooling_type, :unspecified)
    normalize = Keyword.get(opts, :normalize, 2)

    {:ok, tokens} = Tokenizer.encode(model, text)
    ctx_size = max(n_ctx, length(tokens) + 8)

    with {:ok, ctx} <-
           Context.create(model,
             n_ctx: ctx_size,
             embeddings: true,
             pooling_type: pooling_type
           ),
         :ok <- embed_decode(ctx, tokens, 0) do
      get_embeddings(ctx, 0, normalize)
    end
  end

  @doc """
  Computes embeddings for multiple texts.

  Packs multiple texts into a single context as distinct sequences and decodes
  them in batches, rather than allocating a fresh context (and KV cache) per
  text. Accepts the same options as `embed/3`, plus:

    * `:max_batch_sequences` - Max texts per decode batch. Defaults to `64`.

  Pooled embeddings only. When `:pooling_type` is `:none` (no per-sequence pooled
  vector exists), falls back to one context per text.
  """
  @spec embed_batch(Model.t(), [String.t()], keyword()) :: {:ok, [t()]} | {:error, String.t()}
  def embed_batch(%Model{} = model, texts, opts \\ []) when is_list(texts) do
    pooling_type = Keyword.get(opts, :pooling_type, :unspecified)

    cond do
      texts == [] -> {:ok, []}
      pooling_type == :none -> embed_batch_sequential(model, texts, opts)
      true -> embed_batch_pooled(model, texts, opts)
    end
  end

  # --- Batched (single context, multiple sequences) ---

  defp embed_batch_pooled(model, texts, opts) do
    n_ctx = Keyword.get(opts, :n_ctx, 2048)
    pooling_type = Keyword.get(opts, :pooling_type, :unspecified)
    normalize = Keyword.get(opts, :normalize, 2)
    max_seqs = Keyword.get(opts, :max_batch_sequences, @default_max_batch_sequences)

    # texts is non-empty here (embed_batch handles the [] case), so tokenized
    # and groups are non-empty and Enum.max/1 is safe.
    with {:ok, tokenized} <- map_while_ok(texts, &Tokenizer.encode(model, &1)) do
      longest = tokenized |> Enum.map(&length/1) |> Enum.max()
      budget = max(n_ctx, longest + 8)
      groups = group_by_budget(tokenized, budget, max_seqs)
      max_group = groups |> Enum.map(&length/1) |> Enum.max()

      with {:ok, ctx} <-
             Context.create(model,
               n_ctx: budget,
               embeddings: true,
               pooling_type: pooling_type,
               n_seq_max: max(max_group, 1)
             ) do
        decode_groups(ctx, groups, normalize)
      end
    end
  end

  # Greedy bin-packing: each group's total tokens stays within `budget` and its
  # size within `max_seqs`. A single text longer than the budget gets its own
  # group (the context is sized to fit the longest text).
  defp group_by_budget(tokenized, budget, max_seqs) do
    {groups, current, _sum} =
      Enum.reduce(tokenized, {[], [], 0}, fn tokens, {groups, current, sum} ->
        len = length(tokens)
        new_sum = sum + len

        cond do
          current == [] ->
            {groups, [tokens], len}

          length(current) >= max_seqs or new_sum > budget ->
            {[Enum.reverse(current) | groups], [tokens], len}

          true ->
            {groups, [tokens | current], new_sum}
        end
      end)

    groups = if current == [], do: groups, else: [Enum.reverse(current) | groups]
    Enum.reverse(groups)
  end

  # Decodes each group in one batch and extracts per-sequence embeddings,
  # preserving the original text order.
  defp decode_groups(ctx, groups, normalize) do
    with {:ok, nested} <- map_while_ok(groups, &decode_group(ctx, &1, normalize)) do
      {:ok, Enum.concat(nested)}
    end
  end

  defp decode_group(ctx, group, normalize) do
    sequences = Enum.with_index(group, fn tokens, i -> {i, tokens} end)

    with :ok <- embed_batch_decode(ctx, sequences) do
      map_while_ok(0..(length(group) - 1)//1, &get_embeddings(ctx, &1, normalize))
    end
  end

  # --- Sequential fallback (one context per text) ---

  defp embed_batch_sequential(model, texts, opts) do
    map_while_ok(texts, &embed(model, &1, opts))
  end

  # Maps `fun` (returning {:ok, value} | {:error, _}) over `enum`, preserving
  # order and short-circuiting on the first error.
  defp map_while_ok(enum, fun) do
    enum
    |> Enum.reduce_while({:ok, []}, fn item, {:ok, acc} ->
      case fun.(item) do
        {:ok, value} -> {:cont, {:ok, [value | acc]}}
        {:error, _} = err -> {:halt, err}
      end
    end)
    |> then(fn
      {:ok, acc} -> {:ok, Enum.reverse(acc)}
      {:error, _} = err -> err
    end)
  end

  # --- NIF wrappers ---

  defp embed_decode(%Context{ref: ref}, tokens, seq_id),
    do: LlamaCppEx.NIF.embed_decode(ref, tokens, seq_id)

  defp embed_batch_decode(%Context{ref: ref}, sequences),
    do: LlamaCppEx.NIF.embed_batch_decode(ref, sequences)

  defp get_embeddings(%Context{ref: ref}, seq_id, normalize),
    do: LlamaCppEx.NIF.get_embeddings(ref, seq_id, normalize)
end
