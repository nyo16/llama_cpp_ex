defmodule LlamaCppEx.EmbeddingsTest do
  @moduledoc """
  Embedding paths against a real embedding model.

  Its own module, with `:embeddings` as its only gate tag. These two tests used to
  live in `test/smoke_test.exs` under `@moduletag :smoke` with `@tag :embeddings`
  on each — two gate tags on one test, which breaks the taxonomy in
  `test/test_helper.exs`: `--include` beats `--exclude` in ExUnit, so
  `--include smoke` alone pulled them in and they failed with no
  `LLAMA_SMOKE_EMB_MODEL` set. CI only got away with it because its inference job
  happens to pass both flags. They also paid for loading the *generation* model in
  `SmokeTest`'s `setup_all`, which they never used.

      LLAMA_SMOKE_EMB_MODEL=/path/to/embedding-model.gguf \\
        mix test test/embeddings_test.exs --include embeddings

  """
  use ExUnit.Case, async: false

  @moduletag :embeddings
  @moduletag timeout: 120_000

  setup_all do
    :ok = LlamaCppEx.init()
    {:ok, em} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:emb), n_gpu_layers: -1)
    {:ok, emb_model: em}
  end

  test "embed/2 and embed_batch/2 return numeric vectors", %{emb_model: em} do
    assert {:ok, vec} = LlamaCppEx.embed(em, "Elixir is a functional language.")
    assert is_list(vec) and vec != []
    assert Enum.all?(vec, &is_float/1)

    assert {:ok, [a, b]} =
             LlamaCppEx.embed_batch(em, ["functional programming", "a sunny afternoon"])

    assert length(a) == length(b)
    assert length(a) == length(vec)
  end

  # Guards the multi-sequence batch refactor: batched embeddings (single
  # context, many sequences) must match one-context-per-text, including when
  # grouping splits the batch (max_batch_sequences forces multiple groups).
  #
  # Compared by cosine similarity, not elementwise. A batch runs different
  # kernels than a single text on GPU backends (CUDA: up to 0.0027 per element
  # with bge-small, cos 0.99988), so an absolute bound tuned on CPU fails there.
  # The bugs this guards against are far away: rows swapped between sequences
  # give the cosine of two different texts (at most 0.70 on bge-small,
  # embeddinggemma and Qwen3-Embedding), and the wrong pooling at most 0.93.
  test "embed_batch matches per-text embed (incl. multi-group)", %{emb_model: em} do
    texts = [
      "Elixir is a functional programming language.",
      "Erlang runs on the BEAM virtual machine.",
      "The weather today is sunny and warm.",
      "Cats and dogs are common household pets.",
      "Quantum computing uses qubits."
    ]

    reference =
      Enum.map(texts, fn t ->
        {:ok, e} = LlamaCppEx.embed(em, t)
        e
      end)

    assert {:ok, batched} = LlamaCppEx.embed_batch(em, texts)
    assert {:ok, grouped} = LlamaCppEx.embed_batch(em, texts, max_batch_sequences: 2)

    assert length(batched) == length(texts)
    assert min_pairwise_cosine(batched, reference) > 0.999
    assert min_pairwise_cosine(grouped, reference) > 0.999
  end

  # Regression: llama_context reserves its prompt graph for n_ubatch rounded up
  # to a multiple of n_seq_max while n_outputs stays at n_ubatch; mean pooling
  # multiplies the two and GGML_ASSERTs when they differ, taking the VM down at
  # context creation. Five texts do not divide any power-of-two batch, and
  # :mean forces the pooling path on models that default to :last.
  test "embed_batch with mean pooling survives a sequence count that does not divide the batch",
       %{emb_model: em} do
    texts = for i <- 1..5, do: "sentence number #{i}"

    assert {:ok, vectors} = LlamaCppEx.embed_batch(em, texts, pooling_type: :mean)
    assert length(vectors) == 5
    assert Enum.all?(vectors, &(length(&1) == LlamaCppEx.Model.n_embd_out(em)))
  end

  # Lowest cosine similarity between the i-th vectors of two equally long lists.
  defp min_pairwise_cosine(a, b) do
    a
    |> Enum.zip(b)
    |> Enum.map(fn {va, vb} ->
      dot = va |> Enum.zip(vb) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
      dot / (norm(va) * norm(vb))
    end)
    |> Enum.min()
  end

  defp norm(v), do: :math.sqrt(Enum.reduce(v, 0.0, &(&1 * &1 + &2)))
end
