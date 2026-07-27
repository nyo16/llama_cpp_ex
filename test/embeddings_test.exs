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

      GGML_METAL_NO_RESIDENCY=1 \\
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
    assert max_elementwise_diff(batched, reference) < 1.0e-3
    assert max_elementwise_diff(grouped, reference) < 1.0e-3
  end

  defp max_elementwise_diff(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {va, vb} ->
      Enum.zip(va, vb) |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.max()
    end)
    |> Enum.max()
  end
end
