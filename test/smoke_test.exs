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

      GGML_METAL_NO_RESIDENCY=1 \\
      LLAMA_SMOKE_GEN_MODEL=/path/to/chat-model.gguf \\
      LLAMA_SMOKE_EMB_MODEL=/path/to/embedding-model.gguf \\
        mix test test/smoke_test.exs --include smoke --include embeddings

  On Metal, `GGML_METAL_NO_RESIDENCY=1` keeps a passing run from aborting with
  exit 134 while the VM tears down; `test/test_helper.exs` explains why.

  `LLAMA_SMOKE_GEN_MODEL` is required for the generation/chat/grammar tests. The
  embedding tests carry the separate `:embeddings` tag, also excluded by default;
  including that tag requires `LLAMA_SMOKE_EMB_MODEL`.
  Any small instruct model works for generation (e.g. a 0.5B–3B Q4 GGUF).
  """
  use ExUnit.Case, async: false

  @moduletag :smoke
  @moduletag timeout: 120_000

  setup_all do
    :ok = LlamaCppEx.init()

    model_path =
      LlamaCppEx.TestModels.path(:gen) ||
        flunk("""
        Smoke tests require #{LlamaCppEx.TestModels.var(:gen)} to point at a .gguf model file.
        See the @moduledoc in test/smoke_test.exs for usage.
        """)

    {:ok, model} = LlamaCppEx.load_model(model_path, n_gpu_layers: -1)
    {:ok, model: model}
  end

  describe "generation" do
    test "generate/3 returns non-empty deterministic text", %{model: model} do
      opts = [max_tokens: 16, temp: 0.0, seed: 42]

      assert {:ok, text} = LlamaCppEx.generate(model, "The capital of France is", opts)
      assert is_binary(text)
      assert byte_size(text) > 0
      assert String.valid?(text)

      # The test was named for determinism but generated once, so it asserted
      # only non-emptiness. Greedy sampling with a fixed seed must reproduce the
      # same text from a fresh context — asserting *that* is what makes this
      # independent of which model the env var points at.
      assert {:ok, ^text} = LlamaCppEx.generate(model, "The capital of France is", opts)
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

  # A failed or timed-out generation used to be reported as success:
  # `chat_completion/3` wrapped both in `{:ok, %ChatCompletion{}}` with
  # finish_reason "stop"/"length", and the four facade streams silently halted.
  # A caller matching only `{:ok, _}` therefore consumed truncated or empty
  # output and could not tell. These are breaking changes, so they are pinned.
  #
  # `timeout: 0` rather than a small positive number: the receive is `after 0`
  # with an empty mailbox at the first pull, so the timeout branch is taken
  # deterministically instead of racing the first token.
  describe "error semantics" do
    @messages [%{role: "user", content: "Write a very long story about a cat."}]

    test "chat_completion/3 returns {:error, :timeout}, not a truncated success", %{model: model} do
      result = LlamaCppEx.chat_completion(model, @messages, max_tokens: 512, timeout: 0)

      assert result == {:error, :timeout}
      refute match?({:ok, %LlamaCppEx.ChatCompletion{}}, result)
    end

    test "chat_completion/3 still succeeds with a workable timeout", %{model: model} do
      # The control: the timeout branch above is reached because it timed out,
      # not because the call is broken.
      assert {:ok, %LlamaCppEx.ChatCompletion{} = completion} =
               LlamaCppEx.chat_completion(model, @messages, max_tokens: 8, temp: 0.0)

      [choice] = completion.choices
      assert choice.finish_reason in ["stop", "length"]
      assert is_binary(choice.message.content)
    end

    test "stream/3 emits a final {:error, :timeout} instead of truncating", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream("Write a very long story:", max_tokens: 512, timeout: 0)
        |> Enum.to_list()

      assert List.last(chunks) == {:error, :timeout}
      assert chunks |> Enum.drop(-1) |> Enum.all?(&is_binary/1)
    end

    test "stream_chat/3 emits a final {:error, :timeout}", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream_chat(@messages, max_tokens: 512, timeout: 0)
        |> Enum.to_list()

      assert List.last(chunks) == {:error, :timeout}
    end

    test "stream_chat_completion/3 emits a final {:error, :timeout}", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream_chat_completion(@messages, max_tokens: 512, timeout: 0)
        |> Enum.to_list()

      assert List.last(chunks) == {:error, :timeout}

      # Everything before the error is still a well-formed chunk.
      assert chunks
             |> Enum.drop(-1)
             |> Enum.all?(&match?(%LlamaCppEx.ChatCompletionChunk{}, &1))
    end

    test "a stream that is not interrupted contains no error element", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream("Count to three:", max_tokens: 8, temp: 0.0)
        |> Enum.to_list()

      assert chunks != []
      assert Enum.all?(chunks, &is_binary/1)
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
      {:ok, em} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:emb), n_gpu_layers: -1)
      {:ok, emb_model: em}
    end

    @tag :embeddings
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
    @tag :embeddings
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
  end

  defp max_elementwise_diff(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {va, vb} ->
      Enum.zip(va, vb) |> Enum.map(fn {x, y} -> abs(x - y) end) |> Enum.max()
    end)
    |> Enum.max()
  end
end
