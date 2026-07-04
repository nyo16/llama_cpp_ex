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
    with {:ok, tokenized} <- tokenize_all(model, texts) do
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

  # Tokenizes every text up front, short-circuiting on the first error.
  defp tokenize_all(model, texts) do
    Enum.reduce_while(texts, {:ok, []}, fn text, {:ok, acc} ->
      case Tokenizer.encode(model, text) do
        {:ok, tokens} -> {:cont, {:ok, [tokens | acc]}}
        {:error, _} = err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, acc} -> {:ok, Enum.reverse(acc)}
      {:error, _} = err -> err
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
    Enum.reduce_while(groups, {:ok, []}, fn group, {:ok, acc} ->
      sequences = group |> Enum.with_index() |> Enum.map(fn {tokens, i} -> {i, tokens} end)

      with :ok <- embed_batch_decode(ctx, sequences),
           {:ok, embs} <- collect_group_embeddings(ctx, length(group), normalize) do
        {:cont, {:ok, acc ++ embs}}
      else
        {:error, _} = err -> {:halt, err}
      end
    end)
  end

  defp collect_group_embeddings(ctx, count, normalize) do
    Enum.reduce_while(0..(count - 1)//1, {:ok, []}, fn seq_id, {:ok, acc} ->
      case get_embeddings(ctx, seq_id, normalize) do
        {:ok, emb} -> {:cont, {:ok, [emb | acc]}}
        {:error, _} = err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, acc} -> {:ok, Enum.reverse(acc)}
      {:error, _} = err -> err
    end
  end

  # --- Sequential fallback (one context per text) ---

  defp embed_batch_sequential(model, texts, opts) do
    Enum.reduce_while(texts, {:ok, []}, fn text, {:ok, acc} ->
      case embed(model, text, opts) do
        {:ok, emb} -> {:cont, {:ok, [emb | acc]}}
        {:error, _} = err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, embeddings} -> {:ok, Enum.reverse(embeddings)}
      {:error, _} = err -> err
    end
  end

  # --- NIF wrappers ---

  defp embed_decode(%Context{ref: ref}, tokens, seq_id) do
    case LlamaCppEx.NIF.embed_decode(ref, tokens, seq_id) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  defp embed_batch_decode(%Context{ref: ref}, sequences) do
    case LlamaCppEx.NIF.embed_batch_decode(ref, sequences) do
      :ok -> :ok
      {:error, _} = err -> err
    end
  end

  defp get_embeddings(%Context{ref: ref}, seq_id, normalize) do
    case LlamaCppEx.NIF.get_embeddings(ref, seq_id, normalize) do
      {:ok, _} = result -> result
      {:error, _} = err -> err
    end
  end
end
