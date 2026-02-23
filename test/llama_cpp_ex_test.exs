defmodule LlamaCppExTest do
  use ExUnit.Case

  test "backend init" do
    assert :ok = LlamaCppEx.init()
  end

  test "backend init is idempotent" do
    assert :ok = LlamaCppEx.init()
    assert :ok = LlamaCppEx.init()
  end

  describe "with model" do
    @describetag :requires_model

    setup do
      :ok = LlamaCppEx.init()
      path = System.get_env("LLAMA_MODEL_PATH") || "test/fixtures/tiny.gguf"

      case LlamaCppEx.load_model(path, n_gpu_layers: 0) do
        {:ok, model} -> {:ok, model: model}
        {:error, reason} -> {:skip, reason}
      end
    end

    test "model info", %{model: model} do
      assert LlamaCppEx.Model.n_ctx_train(model) > 0
      assert LlamaCppEx.Model.n_embd(model) > 0
      assert is_binary(LlamaCppEx.Model.desc(model))
      assert LlamaCppEx.Model.size(model) > 0
      assert LlamaCppEx.Model.n_params(model) > 0
    end

    test "tokenize roundtrip", %{model: model} do
      text = "Hello, world!"
      {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, text, add_special: false)
      assert is_list(tokens)
      assert length(tokens) > 0

      {:ok, decoded} = LlamaCppEx.Tokenizer.decode(model, tokens)
      assert decoded == text
    end

    test "vocab queries", %{model: model} do
      assert LlamaCppEx.Tokenizer.vocab_size(model) > 0
      bos = LlamaCppEx.Tokenizer.bos_token(model)
      eos = LlamaCppEx.Tokenizer.eos_token(model)
      assert is_integer(bos)
      assert is_integer(eos)
      assert LlamaCppEx.Tokenizer.eog?(model, eos)
    end

    test "generate text", %{model: model} do
      {:ok, text} = LlamaCppEx.generate(model, "Once upon a time", max_tokens: 32)
      assert is_binary(text)
      assert byte_size(text) > 0
    end

    test "stream tokens", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream("Once upon a time", max_tokens: 16, seed: 42)
        |> Enum.to_list()

      assert length(chunks) > 0
      assert Enum.all?(chunks, &is_binary/1)

      # Concatenated chunks should form coherent text
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

    test "chat template", %{model: model} do
      case LlamaCppEx.Model.chat_template(model) do
        nil ->
          :skip

        _template ->
          {:ok, prompt} =
            LlamaCppEx.Chat.apply_template(model, [
              %{role: "system", content: "You are a helpful assistant."},
              %{role: "user", content: "Say hello."}
            ])

          assert is_binary(prompt)
          assert byte_size(prompt) > 0
          # Template should include the message content
          assert prompt =~ "Say hello"
      end
    end

    test "chat generate", %{model: model} do
      case LlamaCppEx.Model.chat_template(model) do
        nil ->
          :skip

        _template ->
          {:ok, reply} =
            LlamaCppEx.chat(
              model,
              [
                %{role: "user", content: "Say just the word 'hello' and nothing else."}
              ],
              max_tokens: 32,
              seed: 42
            )

          assert is_binary(reply)
          assert byte_size(reply) > 0
      end
    end

    test "stream chat", %{model: model} do
      case LlamaCppEx.Model.chat_template(model) do
        nil ->
          :skip

        _template ->
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
end
