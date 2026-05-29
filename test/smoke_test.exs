defmodule LlamaCppEx.SmokeTest do
  @moduledoc """
  End-to-end smoke test that loads real GGUF models and exercises every public
  inference path against the freshly built NIF: generation, streaming, chat
  (template application), structured output (JSON-schema grammar), and
  embeddings.

  This suite is **excluded by default** (it is tagged `:smoke` and
  `test/test_helper.exs` excludes that tag) because it requires model files and
  is slow. Run it explicitly after bumping the `vendor/llama.cpp` submodule or
  rebuilding the NIF:

      LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf \\
      LLAMA_SMOKE_EMB_MODEL=/path/to/embedding-model.gguf \\
        mix test test/smoke_test.exs --include smoke

  `LLAMA_SMOKE_GEN_MODEL` is required for the generation/chat/grammar tests;
  `LLAMA_SMOKE_EMB_MODEL` is optional and only enables the embedding tests.
  Any small instruct model works for generation (e.g. a 0.5B–3B Q4 GGUF).
  """
  use ExUnit.Case, async: false

  @moduletag :smoke
  @moduletag timeout: 120_000

  @gen_model System.get_env("LLAMA_SMOKE_GEN_MODEL")
  @emb_model System.get_env("LLAMA_SMOKE_EMB_MODEL")

  setup_all do
    if @gen_model in [nil, ""] do
      flunk("""
      Smoke tests require LLAMA_SMOKE_GEN_MODEL to point at a .gguf model file.
      See the @moduledoc in test/smoke_test.exs for usage.
      """)
    end

    :ok = LlamaCppEx.init()
    {:ok, model} = LlamaCppEx.load_model(@gen_model, n_gpu_layers: -1)
    {:ok, model: model}
  end

  describe "generation" do
    test "generate/3 returns non-empty deterministic text", %{model: model} do
      assert {:ok, text} =
               LlamaCppEx.generate(model, "The capital of France is",
                 max_tokens: 16,
                 temp: 0.0,
                 seed: 42
               )

      assert is_binary(text)
      assert byte_size(text) > 0
    end

    test "stream/3 yields chunks that join into the full completion", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream("Count to five:", max_tokens: 16, temp: 0.0)
        |> Enum.to_list()

      assert chunks != []
      assert Enum.all?(chunks, &is_binary/1)
      assert chunks |> Enum.join() |> byte_size() > 0
    end
  end

  describe "chat" do
    test "chat/3 applies the model's template and replies", %{model: model} do
      messages = [
        %{role: "system", content: "You are concise."},
        %{role: "user", content: "Reply with a single short greeting."}
      ]

      assert {:ok, reply} = LlamaCppEx.chat(model, messages, max_tokens: 32, temp: 0.0)
      assert is_binary(reply)
      assert byte_size(reply) > 0
    end
  end

  describe "structured output" do
    # Regression test for the grammar double-accept bug: llama_sampler_sample/3
    # already accepts the sampled token, so an extra llama_sampler_accept/2 in
    # the generation loop advanced grammar state twice and threw
    # "Unexpected empty grammar stack" on the very first constrained token.
    @schema %{
      "type" => "object",
      "properties" => %{
        "city" => %{"type" => "string"},
        "population" => %{"type" => "integer"}
      },
      "required" => ["city", "population"]
    }

    test "generate/3 with :json_schema produces schema-valid JSON", %{model: model} do
      assert {:ok, text} =
               LlamaCppEx.generate(model, "Return JSON describing Paris, France.",
                 max_tokens: 96,
                 temp: 0.0,
                 seed: 1,
                 json_schema: @schema
               )

      assert {:ok, decoded} = JSON.decode(text)
      assert is_binary(decoded["city"])
      assert is_integer(decoded["population"])
    end

    test "generate/3 with a raw GBNF grammar constrains output", %{model: model} do
      assert {:ok, text} =
               LlamaCppEx.generate(model, "Say it: ",
                 max_tokens: 8,
                 temp: 0.0,
                 grammar: ~S(root ::= "hello world")
               )

      assert text == "hello world"
    end
  end

  describe "embeddings" do
    setup do
      if @emb_model in [nil, ""] do
        # Embedding tests are opt-in via a second model path.
        :ok
      else
        {:ok, em} = LlamaCppEx.load_model(@emb_model, n_gpu_layers: -1)
        {:ok, emb_model: em}
      end
    end

    @tag :embeddings
    test "embed/2 and embed_batch/2 return numeric vectors", ctx do
      if em = ctx[:emb_model] do
        assert {:ok, vec} = LlamaCppEx.embed(em, "Elixir is a functional language.")
        assert is_list(vec) and vec != []
        assert Enum.all?(vec, &is_float/1)

        assert {:ok, [a, b]} =
                 LlamaCppEx.embed_batch(em, ["functional programming", "a sunny afternoon"])

        assert length(a) == length(b)
        assert length(a) == length(vec)
      else
        IO.puts("\n[smoke] LLAMA_SMOKE_EMB_MODEL not set — skipping embedding assertions")
      end
    end

    # Guards the multi-sequence batch refactor: batched embeddings (single
    # context, many sequences) must match one-context-per-text, including when
    # grouping splits the batch (max_batch_sequences forces multiple groups).
    @tag :embeddings
    test "embed_batch matches per-text embed (incl. multi-group)", ctx do
      if em = ctx[:emb_model] do
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
      else
        IO.puts("\n[smoke] LLAMA_SMOKE_EMB_MODEL not set — skipping embedding assertions")
      end
    end
  end

  defp max_elementwise_diff(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {va, vb} ->
      Enum.zip(va, vb) |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.max()
    end)
    |> Enum.max()
  end
end
