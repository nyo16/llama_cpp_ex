defmodule LlamaCppEx.Embedding do
  @moduledoc "Generate embeddings from text using an embedding model."

  alias LlamaCppEx.{Model, Context, Tokenizer}

  @type t :: [float()]

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

  Uses a fresh context for each text. Accepts the same options as `embed/3`.
  """
  @spec embed_batch(Model.t(), [String.t()], keyword()) :: {:ok, [t()]} | {:error, String.t()}
  def embed_batch(%Model{} = model, texts, opts \\ []) when is_list(texts) do
    results =
      Enum.reduce_while(texts, {:ok, []}, fn text, {:ok, acc} ->
        case embed(model, text, opts) do
          {:ok, emb} -> {:cont, {:ok, [emb | acc]}}
          {:error, _} = err -> {:halt, err}
        end
      end)

    case results do
      {:ok, embeddings} -> {:ok, Enum.reverse(embeddings)}
      {:error, _} = err -> err
    end
  end

  defp embed_decode(%Context{ref: ref}, tokens, seq_id) do
    case LlamaCppEx.NIF.embed_decode(ref, tokens, seq_id) do
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
