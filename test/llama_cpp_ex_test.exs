defmodule LlamaCppExTest do
  use ExUnit.Case

  @model_path System.get_env("LLAMA_MODEL_PATH")
  @embedding_model_path System.get_env("LLAMA_EMBEDDING_MODEL_PATH")

  test "backend init" do
    assert :ok = LlamaCppEx.init()
  end

  test "backend init is idempotent" do
    assert :ok = LlamaCppEx.init()
    assert :ok = LlamaCppEx.init()
  end

  if @model_path && File.exists?(@model_path) do
    describe "model loading" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "model info", %{model: model} do
        assert LlamaCppEx.Model.n_ctx_train(model) > 0
        assert LlamaCppEx.Model.n_embd(model) > 0
        assert is_binary(LlamaCppEx.Model.desc(model))
        assert LlamaCppEx.Model.size(model) > 0
        assert LlamaCppEx.Model.n_params(model) > 0
      end

      test "load_model returns error for missing file" do
        assert {:error, msg} = LlamaCppEx.load_model("/nonexistent/model.gguf")
        assert is_binary(msg)
      end
    end

    describe "tokenizer" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "tokenize roundtrip", %{model: model} do
        text = "Hello, world!"
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, text, add_special: false)
        assert is_list(tokens)
        assert length(tokens) > 0

        {:ok, decoded} = LlamaCppEx.Tokenizer.decode(model, tokens)
        assert decoded == text
      end

      test "encode with special tokens", %{model: model} do
        {:ok, without_special} = LlamaCppEx.Tokenizer.encode(model, "Hi", add_special: false)
        {:ok, with_special} = LlamaCppEx.Tokenizer.encode(model, "Hi", add_special: true)
        # With special tokens should have at least BOS prepended
        assert length(with_special) >= length(without_special)
      end

      test "vocab queries", %{model: model} do
        assert LlamaCppEx.Tokenizer.vocab_size(model) > 0
        bos = LlamaCppEx.Tokenizer.bos_token(model)
        eos = LlamaCppEx.Tokenizer.eos_token(model)
        assert is_integer(bos)
        assert is_integer(eos)
        assert LlamaCppEx.Tokenizer.eog?(model, eos)
      end

      test "token_to_piece returns binary", %{model: model} do
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "hello", add_special: false)
        [first | _] = tokens
        piece = LlamaCppEx.Tokenizer.token_to_piece(model, first)
        assert is_binary(piece)
        assert byte_size(piece) > 0
      end
    end

    describe "generation" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "generate text", %{model: model} do
        {:ok, text} = LlamaCppEx.generate(model, "Once upon a time", max_tokens: 32, seed: 42)
        assert is_binary(text)
        assert byte_size(text) > 0
      end

      test "generate with greedy sampling (temp 0)", %{model: model} do
        {:ok, text} = LlamaCppEx.generate(model, "1 + 1 =", max_tokens: 16, temp: 0.0)
        assert is_binary(text)
      end

      test "generate is deterministic with same seed", %{model: model} do
        opts = [max_tokens: 16, seed: 12345, temp: 0.0]
        {:ok, text1} = LlamaCppEx.generate(model, "The answer is", opts)
        {:ok, text2} = LlamaCppEx.generate(model, "The answer is", opts)
        assert text1 == text2
      end
    end

    describe "streaming" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "stream tokens", %{model: model} do
        chunks =
          model
          |> LlamaCppEx.stream("Once upon a time", max_tokens: 16, seed: 42)
          |> Enum.to_list()

        assert length(chunks) > 0
        assert Enum.all?(chunks, &is_binary/1)

        text = Enum.join(chunks)
        assert byte_size(text) > 0
      end

      test "stream with early halt (Enum.take)", %{model: model} do
        chunks =
          model
          |> LlamaCppEx.stream("The capital of France is", max_tokens: 100, seed: 42)
          |> Enum.take(3)

        assert length(chunks) == 3
      end

      test "stream produces same text as generate", %{model: model} do
        opts = [max_tokens: 16, seed: 42, temp: 0.0]

        {:ok, generated} = LlamaCppEx.generate(model, "Hello", opts)

        streamed =
          model
          |> LlamaCppEx.stream("Hello", opts)
          |> Enum.join()

        assert generated == streamed
      end
    end

    describe "chat" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)

        case LlamaCppEx.Model.chat_template(model) do
          nil -> %{model: model, has_template: false}
          _tmpl -> %{model: model, has_template: true}
        end
      end

      test "apply_template formats messages", %{model: model, has_template: has_template} do
        if has_template do
          {:ok, prompt} =
            LlamaCppEx.Chat.apply_template(model, [
              %{role: "system", content: "You are a helpful assistant."},
              %{role: "user", content: "Say hello."}
            ])

          assert is_binary(prompt)
          assert byte_size(prompt) > 0
          assert prompt =~ "Say hello"
        end
      end

      test "chat generate", %{model: model, has_template: has_template} do
        if has_template do
          {:ok, reply} =
            LlamaCppEx.chat(
              model,
              [%{role: "user", content: "Say just the word 'hello' and nothing else."}],
              max_tokens: 32,
              seed: 42
            )

          assert is_binary(reply)
          assert byte_size(reply) > 0
        end
      end

      test "stream_chat", %{model: model, has_template: has_template} do
        if has_template do
          chunks =
            LlamaCppEx.stream_chat(
              model,
              [%{role: "user", content: "Count to 3."}],
              max_tokens: 32,
              seed: 42
            )
            |> Enum.to_list()

          assert length(chunks) > 0
        end
      end
    end

    describe "context" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "create context with custom n_ctx", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 512)
        assert LlamaCppEx.Context.n_ctx(ctx) == 512
      end

      test "create context with n_seq_max", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 512, n_seq_max: 4)
        assert LlamaCppEx.Context.n_seq_max(ctx) == 4
      end

      test "clear context", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 512)
        assert :ok = LlamaCppEx.Context.clear(ctx)
      end

      test "context + sampler generate", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 2048)
        {:ok, sampler} = LlamaCppEx.Sampler.create(model, temp: 0.0)
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "The answer is")
        {:ok, text} = LlamaCppEx.Context.generate(ctx, sampler, tokens, max_tokens: 16)

        assert is_binary(text)
        assert byte_size(text) > 0
      end
    end

    describe "sampler" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "create with defaults", %{model: model} do
        {:ok, sampler} = LlamaCppEx.Sampler.create(model)
        assert %LlamaCppEx.Sampler{} = sampler
      end

      test "create with custom options", %{model: model} do
        {:ok, sampler} =
          LlamaCppEx.Sampler.create(model,
            temp: 0.5,
            top_k: 20,
            top_p: 0.9,
            min_p: 0.1,
            seed: 42,
            penalty_repeat: 1.2
          )

        assert %LlamaCppEx.Sampler{} = sampler
      end

      test "reset sampler", %{model: model} do
        {:ok, sampler} = LlamaCppEx.Sampler.create(model)
        assert :ok = LlamaCppEx.Sampler.reset(sampler)
      end
    end

    describe "grammar" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "create sampler with grammar", %{model: model} do
        json_grammar = ~S"""
        root   ::= object
        value  ::= object | array | string | number | ("true" | "false" | "null") ws

        object ::=
          "{" ws (
            string ":" ws value
            ("," ws string ":" ws value)*
          )? "}" ws

        array  ::=
          "[" ws (
            value
            ("," ws value)*
          )? "]" ws

        string ::=
          "\"" (
            [^\\"\x7F\x00-\x1F] |
            "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
          )* "\"" ws

        number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))? ws

        ws ::= ([ \t\n] ws)?
        """

        {:ok, sampler} = LlamaCppEx.Sampler.create(model, grammar: json_grammar, temp: 0.0)
        assert %LlamaCppEx.Sampler{} = sampler
      end

      @tag :slow
      test "grammar-constrained generation constrains output", %{model: model} do
        # Grammar that constrains to "yes" or "no" followed by optional whitespace/newlines
        # (avoids empty grammar stack error when generation continues past the constrained portion)
        yesno_grammar = ~S"""
        root ::= answer rest
        answer ::= "yes" | "no"
        rest ::= [^\x00]*
        """

        {:ok, text} =
          LlamaCppEx.generate(model, "Is the sky blue? Answer yes or no: ",
            grammar: yesno_grammar,
            max_tokens: 8,
            temp: 0.0
          )

        assert is_binary(text)
        trimmed = String.trim(text)
        assert String.starts_with?(trimmed, "yes") or String.starts_with?(trimmed, "no")
      end
    end

    describe "prefill and batching NIFs" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "prefill returns token count", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 2048, n_seq_max: 2)
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Hello world")

        {:ok, n_past} = LlamaCppEx.NIF.prefill(ctx.ref, tokens, 0)
        assert n_past == length(tokens)
      end

      test "decode_token with explicit seq_id and position", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 2048, n_seq_max: 2)
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Hello world")
        {:ok, n_past} = LlamaCppEx.NIF.prefill(ctx.ref, tokens, 0)

        # Sample a token
        {:ok, sampler} = LlamaCppEx.Sampler.create(model, temp: 0.0)
        token = LlamaCppEx.NIF.sampler_sample(sampler.ref, ctx.ref)
        assert is_integer(token)

        # Decode it at the right position
        assert :ok = LlamaCppEx.NIF.decode_token(ctx.ref, token, n_past, 0)
      end

      test "memory_seq_keep", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 512, n_seq_max: 2)
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Test", add_special: false)

        {:ok, _} = LlamaCppEx.NIF.prefill(ctx.ref, tokens, 0)
        assert :ok = LlamaCppEx.NIF.memory_seq_keep(ctx.ref, 0)
      end

      test "memory_seq_pos_max", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 512)
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "Hello", add_special: false)

        {:ok, n_past} = LlamaCppEx.NIF.prefill(ctx.ref, tokens, 0)
        pos_max = LlamaCppEx.NIF.memory_seq_pos_max(ctx.ref, 0)
        assert pos_max == n_past - 1
      end
    end

    describe "server" do
      @tag timeout: 120_000
      @tag :slow
      setup do
        :ok = LlamaCppEx.init()

        {:ok, server} =
          LlamaCppEx.Server.start_link(
            model_path: @model_path,
            n_gpu_layers: 0,
            n_parallel: 2,
            n_ctx: 2048
          )

        %{server: server}
      end

      test "generate text", %{server: server} do
        {:ok, text} = LlamaCppEx.Server.generate(server, "Once upon a time", max_tokens: 16)
        assert is_binary(text)
        assert byte_size(text) > 0
      end

      test "stream text", %{server: server} do
        chunks =
          LlamaCppEx.Server.stream(server, "Hello", max_tokens: 8)
          |> Enum.to_list()

        assert length(chunks) > 0
        assert Enum.all?(chunks, &is_binary/1)
      end

      test "concurrent generation", %{server: server} do
        tasks =
          for i <- 1..2 do
            Task.async(fn ->
              LlamaCppEx.Server.generate(server, "Count to #{i}:", max_tokens: 16)
            end)
          end

        results = Task.await_many(tasks, 60_000)
        assert length(results) == 2

        for result <- results do
          assert {:ok, text} = result
          assert is_binary(text)
          assert byte_size(text) > 0
        end
      end

      test "returns error when no slots available", %{server: server} do
        # Fill both slots
        tasks =
          for _ <- 1..2 do
            Task.async(fn ->
              LlamaCppEx.Server.generate(server, "Long story:", max_tokens: 100)
            end)
          end

        # Give tasks time to acquire slots
        Process.sleep(100)

        # Third request should fail
        assert {:error, :no_slots} =
                 LlamaCppEx.Server.generate(server, "Another one", max_tokens: 1)

        Task.await_many(tasks, 60_000)
      end
    end
  else
    @tag :skip
    test "model tests require LLAMA_MODEL_PATH env var" do
      flunk("Set LLAMA_MODEL_PATH to a .gguf file to run model tests")
    end
  end

  if @embedding_model_path && File.exists?(@embedding_model_path) do
    describe "embeddings" do
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@embedding_model_path, n_gpu_layers: 0)
        %{model: model}
      end

      test "embed single text", %{model: model} do
        {:ok, embedding} = LlamaCppEx.embed(model, "Hello world")
        assert is_list(embedding)
        assert length(embedding) > 0
        assert Enum.all?(embedding, &is_float/1)
      end

      test "embedding dimensions match n_embd", %{model: model} do
        n_embd = LlamaCppEx.Model.n_embd(model)
        {:ok, embedding} = LlamaCppEx.embed(model, "Test text")
        assert length(embedding) == n_embd
      end

      test "L2-normalized embeddings have unit length", %{model: model} do
        {:ok, embedding} = LlamaCppEx.embed(model, "Test normalization", normalize: 2)

        # Compute L2 norm
        norm = :math.sqrt(Enum.reduce(embedding, 0.0, fn x, acc -> acc + x * x end))
        assert_in_delta norm, 1.0, 1.0e-5
      end

      test "unnormalized embeddings differ from normalized", %{model: model} do
        {:ok, normalized} = LlamaCppEx.embed(model, "Test text", normalize: 2)
        {:ok, raw} = LlamaCppEx.embed(model, "Test text", normalize: -1)

        # They should produce different values (unless the raw embedding happens to be unit length)
        norm = :math.sqrt(Enum.reduce(raw, 0.0, fn x, acc -> acc + x * x end))
        # Raw embedding is very unlikely to be exactly unit length
        if abs(norm - 1.0) > 1.0e-3 do
          refute normalized == raw
        end
      end

      test "embed_batch", %{model: model} do
        texts = ["Hello", "World", "Elixir is great"]
        {:ok, embeddings} = LlamaCppEx.embed_batch(model, texts)

        assert length(embeddings) == 3
        n_embd = LlamaCppEx.Model.n_embd(model)

        for emb <- embeddings do
          assert length(emb) == n_embd
          assert Enum.all?(emb, &is_float/1)
        end
      end

      test "different texts produce different embeddings", %{model: model} do
        {:ok, emb1} = LlamaCppEx.embed(model, "The cat sat on the mat")
        {:ok, emb2} = LlamaCppEx.embed(model, "Quantum mechanics is complex")

        # Cosine similarity should be < 1.0 for different texts
        dot = Enum.zip(emb1, emb2) |> Enum.reduce(0.0, fn {a, b}, acc -> acc + a * b end)
        # Both are L2-normalized, so dot product IS cosine similarity
        assert dot < 0.99
      end

      test "same text produces same embedding", %{model: model} do
        {:ok, emb1} = LlamaCppEx.embed(model, "Hello world")
        {:ok, emb2} = LlamaCppEx.embed(model, "Hello world")
        assert emb1 == emb2
      end

      test "embed with pooling_type option", %{model: model} do
        {:ok, embedding} = LlamaCppEx.embed(model, "Test", pooling_type: :last)
        assert is_list(embedding)
        assert length(embedding) > 0
      end
    end
  else
    @tag :skip
    test "embedding tests require LLAMA_EMBEDDING_MODEL_PATH env var" do
      flunk("Set LLAMA_EMBEDDING_MODEL_PATH to an embedding .gguf file to run embedding tests")
    end
  end
end
