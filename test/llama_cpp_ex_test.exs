defmodule LlamaCppExTest do
  use ExUnit.Case

  describe "Grammar (JSON Schema to GBNF)" do
    test "from_json_schema converts simple object schema" do
      :ok = LlamaCppEx.init()

      schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        },
        "required" => ["name", "age"]
      }

      assert {:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(schema)
      assert is_binary(gbnf)
      assert byte_size(gbnf) > 0
      # Should contain root rule
      assert gbnf =~ "root"
    end

    test "from_json_schema converts simple string schema" do
      :ok = LlamaCppEx.init()

      schema = %{"type" => "string"}
      assert {:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(schema)
      assert is_binary(gbnf)
    end

    test "from_json_schema! raises on a schema llama.cpp rejects" do
      :ok = LlamaCppEx.init()

      # There used to be a bare `rescue _ -> :ok` after this assert_raise, which
      # swallowed the ExUnit.AssertionError it raises on failure — the test could
      # not fail, in either direction.
      assert_raise ArgumentError,
                   ~r/failed to convert JSON schema to grammar.*Unrecognized schema/s,
                   fn ->
                     LlamaCppEx.Grammar.from_json_schema!(%{
                       "type" => "invalid_type_that_does_not_exist"
                     })
                   end
    end

    test "from_json_schema! returns the grammar when conversion succeeds" do
      :ok = LlamaCppEx.init()

      gbnf = LlamaCppEx.Grammar.from_json_schema!(%{"type" => "boolean"})
      assert is_binary(gbnf)
      assert gbnf =~ "root"
    end

    test "from_json_schema returns {:error, reason} instead of raising" do
      :ok = LlamaCppEx.init()

      # The error branch of from_json_schema/1 itself — the previous test with
      # this name reached past it into the NIF, so nothing covered the public
      # function's own contract.
      assert {:error, reason} =
               LlamaCppEx.Grammar.from_json_schema(%{"type" => "not_a_json_schema_type"})

      assert is_binary(reason)
      assert reason =~ "Unrecognized schema"

      assert {:error, ref_error} =
               LlamaCppEx.Grammar.from_json_schema(%{"$ref" => "#/definitions/missing"})

      assert ref_error =~ "Error resolving ref"
    end

    test "from_json_schema refuses schemas that would blow the C++ stack" do
      :ok = LlamaCppEx.init()

      # Unbounded recursion in llama.cpp's converter was a SIGSEGV, not an error.
      deep =
        Enum.reduce(1..300, %{"type" => "string"}, fn _, acc ->
          %{"type" => "object", "properties" => %{"a" => acc}, "required" => ["a"]}
        end)

      assert {:error, "schema nested too deeply"} = LlamaCppEx.Grammar.from_json_schema(deep)

      assert {:error, "schema too large"} =
               LlamaCppEx.Grammar.from_json_schema(%{
                 "type" => "string",
                 "pattern" => String.duplicate("a", 2_000_000)
               })
    end

    test "the NIF rejects a malformed JSON string" do
      :ok = LlamaCppEx.init()

      # from_json_schema/1 encodes the map, so malformed JSON can only reach the
      # NIF directly.
      assert {:error, reason} = LlamaCppEx.NIF.json_schema_to_grammar_nif("{invalid json")
      assert is_binary(reason)
    end

    test "raises when both :grammar and :json_schema provided" do
      :ok = LlamaCppEx.init()

      assert_raise ArgumentError, ~r/cannot use both/, fn ->
        # This should raise before even trying to load a model
        LlamaCppEx.generate(
          %LlamaCppEx.Model{ref: nil},
          "test",
          grammar: "root ::= \"hello\"",
          json_schema: %{"type" => "string"}
        )
      end
    end
  end

  if Code.ensure_loaded?(Ecto.Schema) do
    defmodule TestPerson do
      use Ecto.Schema

      @primary_key false
      embedded_schema do
        field(:name, :string)
        field(:age, :integer)
        field(:active, :boolean)
        field(:score, :float)
        field(:tags, {:array, :string})
      end
    end

    describe "Schema (Ecto to JSON Schema)" do
      test "to_json_schema converts basic schema" do
        schema = LlamaCppEx.Schema.to_json_schema(TestPerson)

        assert schema["type"] == "object"
        assert schema["properties"]["name"] == %{"type" => "string"}
        assert schema["properties"]["age"] == %{"type" => "integer"}
        assert schema["properties"]["active"] == %{"type" => "boolean"}
        assert schema["properties"]["score"] == %{"type" => "number"}

        assert schema["properties"]["tags"] == %{
                 "type" => "array",
                 "items" => %{"type" => "string"}
               }

        assert "name" in schema["required"]
        assert "age" in schema["required"]
      end

      test "to_json_schema raises for non-schema module" do
        assert_raise ArgumentError, fn ->
          LlamaCppEx.Schema.to_json_schema(String)
        end
      end

      test "end-to-end: Ecto schema -> JSON Schema -> GBNF" do
        :ok = LlamaCppEx.init()

        schema = LlamaCppEx.Schema.to_json_schema(TestPerson)
        assert {:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(schema)
        assert is_binary(gbnf)
        assert byte_size(gbnf) > 0
      end
    end
  end

  test "backend init" do
    assert :ok = LlamaCppEx.init()
  end

  test "backend init is idempotent" do
    assert :ok = LlamaCppEx.init()
    assert :ok = LlamaCppEx.init()
  end

  test "load_model returns error for missing file" do
    :ok = LlamaCppEx.init()
    assert {:error, msg} = LlamaCppEx.load_model("/nonexistent/model.gguf")
    assert is_binary(msg)
  end

  # These used to build a struct from a literal and assert the literal back at
  # itself — three tests that could not fail, because nothing in `lib/` took
  # part. What the structs actually owe callers is the OpenAI wire shape, so
  # that is what is pinned here: the exact field set, `nil` defaults, and
  # rejection of a field that is not part of the format.
  describe "ChatCompletion struct" do
    test "carries exactly the OpenAI chat.completion fields" do
      assert %LlamaCppEx.ChatCompletion{} |> Map.from_struct() |> Map.keys() |> Enum.sort() ==
               [:choices, :created, :id, :model, :object, :usage]
    end

    test "every field defaults to nil, so a half-built response is visible" do
      empty = %LlamaCppEx.ChatCompletion{}
      assert empty |> Map.from_struct() |> Map.values() |> Enum.all?(&is_nil/1)
    end

    test "rejects a field outside the format" do
      assert_raise KeyError, fn ->
        struct!(LlamaCppEx.ChatCompletion, %{finish_reason: "stop"})
      end
    end
  end

  describe "ChatCompletionChunk struct" do
    test "carries exactly the OpenAI chat.completion.chunk fields" do
      assert %LlamaCppEx.ChatCompletionChunk{} |> Map.from_struct() |> Map.keys() |> Enum.sort() ==
               [:choices, :created, :id, :model, :object]
    end

    test "has no :usage field — chunks never carry token counts" do
      refute Map.has_key?(%LlamaCppEx.ChatCompletionChunk{}, :usage)

      assert_raise KeyError, fn ->
        struct!(LlamaCppEx.ChatCompletionChunk, %{usage: %{total_tokens: 1}})
      end
    end
  end

  describe "model loading" do
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
      %{model: model}
    end

    test "model info", %{model: model} do
      assert LlamaCppEx.Model.n_ctx_train(model) > 0
      assert LlamaCppEx.Model.n_embd(model) > 0
      assert is_binary(LlamaCppEx.Model.desc(model))
      assert LlamaCppEx.Model.size(model) > 0
      assert LlamaCppEx.Model.n_params(model) > 0
    end
  end

  describe "tokenizer" do
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
      %{model: model}
    end

    test "tokenize roundtrip", %{model: model} do
      text = "Hello, world!"
      {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, text, add_special: false)
      assert is_list(tokens)
      assert tokens != []

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
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
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
      opts = [max_tokens: 16, seed: 12_345, temp: 0.0]
      {:ok, text1} = LlamaCppEx.generate(model, "The answer is", opts)
      {:ok, text2} = LlamaCppEx.generate(model, "The answer is", opts)
      assert text1 == text2
    end
  end

  describe "streaming" do
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
      %{model: model}
    end

    test "stream tokens", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream("Once upon a time", max_tokens: 16, seed: 42)
        |> Enum.to_list()

      assert chunks != []
      assert Enum.all?(chunks, &is_binary/1)

      text = Enum.join(chunks)
      assert byte_size(text) > 0
    end

    test "stream with early halt (Enum.take)", %{model: model} do
      chunks =
        model
        |> LlamaCppEx.stream("Once upon a time", max_tokens: 100, seed: 42)
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
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
      template = LlamaCppEx.Model.chat_template(model)

      if is_nil(template) do
        flunk("""
        The model at #{LlamaCppEx.TestModels.var(:gen)} embeds no chat template, so the
        "chat" tests cannot run. Point #{LlamaCppEx.TestModels.var(:gen)} at an
        instruct/chat .gguf that ships a tokenizer.chat_template.
        """)
      end

      %{model: model}
    end

    test "apply_template formats messages", %{model: model} do
      {:ok, prompt} =
        LlamaCppEx.Chat.apply_template(model, [
          %{role: "system", content: "You are a helpful assistant."},
          %{role: "user", content: "Say hello."}
        ])

      assert is_binary(prompt)
      assert byte_size(prompt) > 0
      assert prompt =~ "Say hello"
    end

    test "apply_template with enable_thinking option", %{model: model} do
      messages = [
        %{role: "user", content: "Hello"}
      ]

      {:ok, prompt_thinking} =
        LlamaCppEx.Chat.apply_template(model, messages, enable_thinking: true)

      {:ok, prompt_no_thinking} =
        LlamaCppEx.Chat.apply_template(model, messages, enable_thinking: false)

      # Both should be valid prompts
      assert is_binary(prompt_thinking)
      assert is_binary(prompt_no_thinking)
      assert byte_size(prompt_thinking) > 0
      assert byte_size(prompt_no_thinking) > 0

      # For models that support enable_thinking (like Qwen3), the prompts
      # will differ. For models that don't, they may be the same.
      # Either way, both should contain the user message.
      assert prompt_thinking =~ "Hello"
      assert prompt_no_thinking =~ "Hello"
    end

    test "chat generate", %{model: model} do
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

    test "stream_chat", %{model: model} do
      chunks =
        LlamaCppEx.stream_chat(
          model,
          [%{role: "user", content: "Count to 3."}],
          max_tokens: 32,
          seed: 42
        )
        |> Enum.to_list()

      assert chunks != []
    end

    test "chat_completion returns ChatCompletion struct", %{model: model} do
      {:ok, completion} =
        LlamaCppEx.chat_completion(
          model,
          [%{role: "user", content: "Say hello."}],
          max_tokens: 16,
          seed: 42
        )

      assert %LlamaCppEx.ChatCompletion{} = completion
      assert String.starts_with?(completion.id, "chatcmpl-")
      assert completion.object == "chat.completion"
      assert is_integer(completion.created)
      assert is_binary(completion.model)

      [choice] = completion.choices
      assert choice.index == 0
      assert choice.message.role == "assistant"
      assert is_binary(choice.message.content)
      assert byte_size(choice.message.content) > 0
      assert choice.finish_reason in ["stop", "length"]

      assert completion.usage.prompt_tokens > 0
      assert completion.usage.completion_tokens > 0

      assert completion.usage.total_tokens ==
               completion.usage.prompt_tokens + completion.usage.completion_tokens
    end

    test "stream_chat_completion emits ChatCompletionChunk structs", %{model: model} do
      chunks =
        LlamaCppEx.stream_chat_completion(
          model,
          [%{role: "user", content: "Say hello."}],
          max_tokens: 16,
          seed: 42
        )
        |> Enum.to_list()

      assert length(chunks) >= 2

      # All chunks are ChatCompletionChunk structs
      assert Enum.all?(chunks, &match?(%LlamaCppEx.ChatCompletionChunk{}, &1))

      # All chunks share the same id and created
      [first | _] = chunks
      assert String.starts_with?(first.id, "chatcmpl-")
      assert Enum.all?(chunks, fn c -> c.id == first.id end)
      assert Enum.all?(chunks, fn c -> c.created == first.created end)

      # First chunk has role delta
      first_choice = hd(first.choices)
      assert first_choice.delta.role == "assistant"
      assert first_choice.finish_reason == nil

      # Last chunk has finish_reason
      last = List.last(chunks)
      last_choice = hd(last.choices)
      assert last_choice.finish_reason in ["stop", "length"]

      # Middle chunks have content deltas
      middle = Enum.slice(chunks, 1..-2//1)

      for chunk <- middle do
        choice = hd(chunk.choices)
        assert is_binary(choice.delta.content)
        assert choice.finish_reason == nil
      end
    end
  end

  describe "context" do
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
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
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
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

    test "create with penalty_present and penalty_freq", %{model: model} do
      {:ok, sampler} =
        LlamaCppEx.Sampler.create(model,
          temp: 1.0,
          top_k: 20,
          top_p: 0.95,
          penalty_present: 1.5,
          penalty_freq: 0.5
        )

      assert %LlamaCppEx.Sampler{} = sampler
    end

    test "reset sampler", %{model: model} do
      {:ok, sampler} = LlamaCppEx.Sampler.create(model)
      assert :ok = LlamaCppEx.Sampler.reset(sampler)
    end
  end

  describe "grammar" do
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
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

    test "json_schema produces valid GBNF grammar with model loaded", %{model: model} do
      schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        },
        "required" => ["name", "age"],
        "additionalProperties" => false
      }

      {:ok, gbnf} = LlamaCppEx.Grammar.from_json_schema(schema)
      assert gbnf =~ "root"
      assert gbnf =~ "name"
      assert gbnf =~ "age"
      assert gbnf =~ "integer"
      assert gbnf =~ "string"

      # Verify the grammar can be used to create a sampler (proves it's valid GBNF)
      {:ok, _sampler} = LlamaCppEx.Sampler.create(model, grammar: gbnf, temp: 0.0)
    end

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
    @describetag :smoke

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen))
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
    # One gate tag, and the timeout as a describetag: `@tag` before a `setup`
    # block is inert, so the 120s here never applied and the `:slow` never
    # excluded anything (`--include smoke` beats `--exclude slow`).
    @describetag :smoke
    @describetag timeout: 120_000

    setup do
      :ok = LlamaCppEx.init()
      model_path = LlamaCppEx.TestModels.path!(:gen)

      {:ok, server} =
        LlamaCppEx.Server.start_link(
          model_path: model_path,
          n_parallel: 2,
          n_ctx: 8192,
          temp: 0.0,
          seed: 42
        )

      %{server: server, model_path: model_path}
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

      assert chunks != []
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

    test "requests queue when all slots busy", %{server: server} do
      # n_parallel=2, so fire 4 requests — 2 will queue
      tasks =
        for i <- 1..4 do
          Task.async(fn ->
            LlamaCppEx.Server.generate(server, "Count to #{i}:", max_tokens: 8)
          end)
        end

      results = Task.await_many(tasks, 120_000)
      assert length(results) == 4

      for result <- results do
        assert {:ok, text} = result
        assert is_binary(text)
        assert byte_size(text) > 0
      end
    end

    test "get_stats returns server state", %{server: server} do
      stats = LlamaCppEx.Server.get_stats(server)
      assert stats.n_parallel == 2
      assert stats.idle_slots == 2
      assert stats.active_slots == 0
      assert stats.queue_depth == 0
    end

    test "chunked prefill with long prompt", %{server: server} do
      # Generate a prompt long enough to require chunking (chunk_size defaults to 512)
      long_prompt = String.duplicate("The quick brown fox jumps over the lazy dog. ", 50)

      {:ok, text} =
        LlamaCppEx.Server.generate(server, long_prompt, max_tokens: 8, timeout: 120_000)

      assert is_binary(text)
      assert byte_size(text) > 0
    end

    test "telemetry events fire on request completion", %{server: server} do
      ref = make_ref()
      test_pid = self()

      :telemetry.attach(
        "test-request-start-#{inspect(ref)}",
        [:llama_cpp_ex, :server, :request, :start],
        fn _event, measurements, metadata, _config ->
          send(test_pid, {:start_telemetry, measurements, metadata})
        end,
        nil
      )

      :telemetry.attach(
        "test-request-done-#{inspect(ref)}",
        [:llama_cpp_ex, :server, :request, :done],
        fn _event, measurements, metadata, _config ->
          send(test_pid, {:telemetry, measurements, metadata})
        end,
        nil
      )

      :telemetry.attach(
        "test-tick-#{inspect(ref)}",
        [:llama_cpp_ex, :server, :tick],
        fn _event, measurements, _metadata, _config ->
          send(test_pid, {:tick_telemetry, measurements})
        end,
        nil
      )

      {:ok, _text} = LlamaCppEx.Server.generate(server, "Hello", max_tokens: 4)

      # :start fires synchronously inside the handle_call, before :done.
      assert_receive {:start_telemetry, start_measurements, start_metadata}, 5_000
      assert is_number(start_measurements.prompt_tokens)
      assert start_measurements.prompt_tokens > 0
      assert is_number(start_measurements.prefix_cache_tokens)
      assert is_pid(start_metadata.server)
      assert is_integer(start_metadata.seq_id)
      assert start_metadata.mode == :generate

      assert_receive {:telemetry, measurements, metadata}, 5_000

      assert is_number(measurements.prompt_tokens)
      assert measurements.prompt_tokens > 0
      assert is_number(measurements.generated_tokens)
      assert is_number(measurements.duration_ms)
      assert measurements.duration_ms > 0
      assert is_number(measurements.ttft_ms)
      assert is_number(measurements.prompt_eval_rate)
      assert is_number(measurements.generation_rate)

      # ttft_ms should be a real measurement, not a fallback to duration_ms.
      # With max_tokens=4 there will be tokens after the first, so ttft must
      # be strictly less than duration.
      assert measurements.ttft_ms < measurements.duration_ms

      assert is_pid(metadata.server)
      assert is_integer(metadata.seq_id)
      assert metadata.mode == :generate
      assert metadata.stop_reason in [:eog, :max_tokens]

      # Should also have received tick telemetry
      assert_receive {:tick_telemetry, tick_measurements}, 1_000
      assert is_number(tick_measurements.batch_size)
      assert tick_measurements.batch_size > 0
      assert is_number(tick_measurements.eval_ms)

      :telemetry.detach("test-request-start-#{inspect(ref)}")
      :telemetry.detach("test-request-done-#{inspect(ref)}")
      :telemetry.detach("test-tick-#{inspect(ref)}")
    end

    test "cache_prompt sequential requests don't crash on hybrid models", %{
      model_path: model_path
    } do
      # Regression test for the M-RoPE positional-mismatch abort that fired
      # when partial seq_rm silently no-op'd on hybrid GDN models (Qwen 3.5
      # / 3.6). The fix: probe common_context_can_seq_rm at server init and
      # fall back to a full slot reset when the model only supports `:full`
      # range trims.
      :ok = LlamaCppEx.init()

      {:ok, server} =
        LlamaCppEx.Server.start_link(
          model_path: model_path,
          n_parallel: 1,
          n_ctx: 4096,
          cache_prompt: true,
          temp: 0.0,
          seed: 42
        )

      parent = self()
      handler = {__MODULE__, :hybrid_cache, make_ref()}

      :telemetry.attach(
        handler,
        [:llama_cpp_ex, :server, :request, :start],
        fn _e, m, _meta, _ -> send(parent, {:started, m}) end,
        nil
      )

      on_exit(fn -> :telemetry.detach(handler) end)

      shared = "System: respond briefly.\nUser: "

      {:ok, t1} =
        LlamaCppEx.Server.generate(server, shared <> "Say one word.", max_tokens: 12)

      assert_receive {:started, %{prefix_cache_tokens: 0}}, 60_000

      # Second request shares a prefix with the first but diverges — this is
      # the path that previously triggered the M-RoPE crash on hybrid models.
      {:ok, t2} =
        LlamaCppEx.Server.generate(server, shared <> "Pick a color.", max_tokens: 12)

      assert_receive {:started, %{prefix_cache_tokens: reused}}, 60_000

      # Which path the divergent request takes is decided by the model, and both
      # are under test here:
      #
      #   :part — partial trim is supported, so the shared prefix is reused and
      #           the trimming path itself ran rather than being skipped by a
      #           cache miss.
      #   :full — the hybrid GDN case this test is named for. The documented fix
      #           is to detect `:full` at server init and fall back to a full
      #           slot reset, so reuse is *declined*: a non-zero count here would
      #           mean the fallback did not fire and the M-RoPE positional
      #           mismatch is reachable again.
      case LlamaCppEx.TestModels.seq_rm_kind(:gen) do
        :part ->
          assert reused > 0

        kind when kind in [:full, :rs] ->
          assert reused == 0,
                 "#{kind} model reported #{reused} reused tokens; the full-reset " <>
                   "fallback did not fire and a partial trim was attempted anyway"
      end

      # A small model can legitimately emit EOG immediately, so the content is
      # not the subject here — surviving the divergent second request is. The
      # server answering a third one proves it.
      assert is_binary(t1)
      assert is_binary(t2)
      assert Process.alive?(server)
      assert {:ok, t3} = LlamaCppEx.Server.generate(server, shared <> "Count.", max_tokens: 4)
      assert is_binary(t3)

      GenServer.stop(server, :normal, 10_000)
    end
  end

  describe "embeddings" do
    @describetag :embeddings

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:emb))
      %{model: model}
    end

    test "embed single text", %{model: model} do
      {:ok, embedding} = LlamaCppEx.embed(model, "Hello world")
      assert is_list(embedding)
      assert embedding != []
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

      # A raw embedding is vanishingly unlikely to already be unit length; if it
      # is, the comparison below is meaningless, so assert the premise too.
      norm = :math.sqrt(Enum.reduce(raw, 0.0, fn x, acc -> acc + x * x end))

      assert abs(norm - 1.0) > 1.0e-3,
             "raw embedding was already unit length (norm=#{norm}); normalize: -1 is a no-op"

      refute normalized == raw
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
      assert embedding != []
    end
  end

  describe "MTP speculative decoding" do
    @describetag :mtp
    @describetag timeout: 300_000

    setup do
      :ok = LlamaCppEx.init()

      {:ok, model} =
        LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp),
          n_gpu_layers: 999,
          load_mtp: true
        )

      {:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 4096)
      on_exit(fn -> :ok end)
      %{model: model, mtp: mtp}
    end

    test "contexts use no recurrent-state rollback slots", %{mtp: mtp} do
      # Matching upstream server, the draft ctx is created with n_rs_seq=0 —
      # MTP rolls back via cached hidden states (pending_h / verify_h), not
      # recurrent-state snapshots. Both contexts report 0.
      assert LlamaCppEx.Context.n_rs_seq(mtp.mtp_ctx) == 0
      assert LlamaCppEx.Context.n_rs_seq(mtp.main_ctx) == 0
    end

    test "stream produces text and a sensible acceptance rate", %{mtp: mtp} do
      chunks =
        mtp
        |> LlamaCppEx.MTP.stream("Briefly explain prime numbers:",
          max_tokens: 96,
          temp: 0.7,
          seed: 42
        )
        |> Enum.to_list()

      text = IO.iodata_to_binary(chunks)
      assert byte_size(text) > 0

      stats = LlamaCppEx.MTP.stats(mtp)
      assert stats.tokens_emitted > 0
      # Loose floor — upstream reports ~0.75 on Qwen 3.6 MTP, but the
      # observed rate is hardware-sensitive: on Apple Silicon we see
      # ~0.20–0.25 (the verify-batch on Metal is slow enough that we
      # don't gain much from larger n_draft; see upstream #23011 / #23114).
      # Anything under 0.15 indicates the draft/verify wiring is broken.
      assert stats.acceptance_rate > 0.15,
             "acceptance_rate=#{stats.acceptance_rate} (expected > 0.15); " <>
               "stats=#{inspect(stats)}"
    end

    test "live stats are readable mid-stream and advance monotonically", %{mtp: mtp} do
      parent = self()

      gen_task =
        Task.async(fn ->
          mtp
          |> LlamaCppEx.MTP.stream("Count from one to one hundred:",
            max_tokens: 64,
            temp: 0.0
          )
          |> Enum.into("")
          |> then(&send(parent, {:gen_done, &1}))
        end)

      # Sample a few snapshots while generation is in flight.
      snapshots =
        for _ <- 1..5 do
          Process.sleep(50)
          LlamaCppEx.MTP.stats(mtp)
        end

      Task.await(gen_task, 60_000)
      assert_receive {:gen_done, _text}, 1_000

      token_counts = Enum.map(snapshots, & &1.tokens_emitted)

      assert token_counts == Enum.sort(token_counts),
             "tokens_emitted should be monotonically non-decreasing, got #{inspect(token_counts)}"
    end
  end
end
