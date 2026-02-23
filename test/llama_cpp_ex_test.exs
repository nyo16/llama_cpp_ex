defmodule LlamaCppExTest do
  use ExUnit.Case

  @model_path System.get_env("LLAMA_MODEL_PATH")

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

      test "clear context", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 512)
        assert :ok = LlamaCppEx.Context.clear(ctx)
      end

      test "context + sampler generate", %{model: model} do
        {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 2048)
        {:ok, sampler} = LlamaCppEx.Sampler.create(temp: 0.0)
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, "The answer is")
        {:ok, text} = LlamaCppEx.Context.generate(ctx, sampler, tokens, max_tokens: 16)

        assert is_binary(text)
        assert byte_size(text) > 0
      end
    end

    describe "sampler" do
      test "create with defaults" do
        {:ok, sampler} = LlamaCppEx.Sampler.create()
        assert %LlamaCppEx.Sampler{} = sampler
      end

      test "create with custom options" do
        {:ok, sampler} =
          LlamaCppEx.Sampler.create(
            temp: 0.5,
            top_k: 20,
            top_p: 0.9,
            min_p: 0.1,
            seed: 42,
            penalty_repeat: 1.2
          )

        assert %LlamaCppEx.Sampler{} = sampler
      end

      test "reset sampler" do
        {:ok, sampler} = LlamaCppEx.Sampler.create()
        assert :ok = LlamaCppEx.Sampler.reset(sampler)
      end
    end
  else
    @tag :skip
    test "model tests require LLAMA_MODEL_PATH env var" do
      flunk("Set LLAMA_MODEL_PATH to a .gguf file to run model tests")
    end
  end
end
