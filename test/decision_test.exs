defmodule LlamaCppEx.DecisionTest do
  @moduledoc """
  Decision models (`LlamaCppEx.Decision`) against upstream's two test models,
  which between them cover both readouts: tinylaya reads scores from the
  embeddings output at marker tokens, tinyopenjev reads label logits and shares
  a prompt prefix across questions.

      GGML_METAL_NO_RESIDENCY=1 \\
      LLAMA_SMOKE_DECISION_LAYA_MODEL=/path/to/tinylaya-for-testing-Q8_0.gguf \\
      LLAMA_SMOKE_DECISION_OPENJEV_MODEL=/path/to/tinyopenjev-for-testing-Q8_0.gguf \\
        mix test test/decision_test.exs --include decision

  Both files are in ggml-org's `tinylaya-for-testing-gguf` and
  `tinyopenjev-for-testing-gguf` repos on Hugging Face. They are random-weight
  test models, so these tests assert the shape and invariants of the answers,
  never which option wins.
  """
  use ExUnit.Case, async: false

  alias LlamaCppEx.Decision

  @moduletag :decision
  @moduletag timeout: 120_000

  @state "I was charged twice for my order last week and nobody has replied."

  # Atom ids and option keys on purpose: they must come back as given.
  @questions [
    route: [
      type: :choice,
      instructions: "Which team should handle this?",
      criteria: [billing: "payments and refunds", shipping: nil, technical: nil]
    ],
    urgency: [
      type: :score,
      instructions: "How urgent is this?",
      criteria: ["can wait", "this week", "today", "right now"]
    ],
    angry: [type: :noul, instructions: "Is the customer angry?"]
  ]

  setup_all do
    :ok = LlamaCppEx.init()

    decisions =
      for kind <- [:decision_laya, :decision_openjev], into: %{} do
        {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(kind), n_gpu_layers: -1)
        {:ok, decision} = Decision.new(model)
        {kind, decision}
      end

    {:ok, decisions}
  end

  for kind <- [:decision_laya, :decision_openjev] do
    test "#{kind}: answers have llama-server's shape, keyed as the caller wrote them", ctx do
      assert {:ok, %{answers: answers, usage: usage}} =
               Decision.decide(ctx[unquote(kind)], @state, @questions)

      assert usage.input_tokens > 0
      assert usage.output_tokens == 0
      assert answers |> Map.keys() |> Enum.sort() == [:angry, :route, :urgency]

      route = answers.route
      assert route.type == :choice
      assert route.probabilities |> Map.keys() |> Enum.sort() == [:billing, :shipping, :technical]
      assert_in_delta Enum.sum(Map.values(route.probabilities)), 1.0, 1.0e-4
      assert route.choice == Enum.max_by(route.probabilities, &elem(&1, 1)) |> elem(0)
      assert route.confidence >= 0.0 and route.confidence <= 1.0

      urgency = answers.urgency
      assert urgency.type == :score

      assert urgency.legend == %{
               0 => "can wait",
               1 => "this week",
               2 => "today",
               3 => "right now"
             }

      assert urgency.probabilities |> Map.keys() |> Enum.sort() == [0, 1, 2, 3]
      assert_in_delta Enum.sum(Map.values(urgency.probabilities)), 1.0, 1.0e-4

      assert_in_delta urgency.score,
                      Enum.sum(Enum.map(urgency.probabilities, fn {i, p} -> i * p end)),
                      1.0e-4

      assert urgency.confidence >= 0.0 and urgency.confidence <= 1.0

      assert answers.angry.type == :noul
      assert answers.angry.noul >= 0.0 and answers.angry.noul <= 1.0
    end

    test "#{kind}: a JSON state is given to the model as its JSON text", ctx do
      questions = %{
        "refund" => %{
          "type" => "noul",
          "instructions" => "Is a refund requested?",
          "criteria" => %{"false" => "no refund is asked", "true" => "a refund is asked"}
        }
      }

      # The template serializes with `tojson`, which separates with ", " and
      # ": "; a map goes in Elixir's (sorted) key order.
      state = %{"ticket" => @state, "plan" => "pro"}
      text = ~s({"plan": "pro", "ticket": "#{@state}"})
      assert {:ok, as_term} = Decision.decide(ctx[unquote(kind)], state, questions)
      assert {:ok, as_text} = Decision.decide(ctx[unquote(kind)], text, questions)
      assert as_term.usage == as_text.usage
      assert_in_delta as_term.answers["refund"].noul, as_text.answers["refund"].noul, 1.0e-4
    end

    test "#{kind}: rejects a malformed request with llama-server's messages", ctx do
      decision = ctx[unquote(kind)]

      assert {:error, "\"questions\" must be a non-empty object"} =
               Decision.decide(decision, @state, %{})

      assert {:error, "questions.q: \"instructions\" must be provided"} =
               Decision.decide(decision, @state, q: [type: :noul])

      assert {:error, "questions.q: \"type\" must be one of: choice, score, noul"} =
               Decision.decide(decision, @state, q: [type: :yes_no, instructions: "x"])

      assert {:error, "questions.q: \"criteria\" must be an array of 2 to 10 levels"} =
               Decision.decide(decision, @state,
                 q: [type: :score, instructions: "x", criteria: ["one"]]
               )
    end

    test "#{kind}: rejects image input, which this build cannot read", ctx do
      state = [
        %{
          "role" => "user",
          "content" => [
            %{"type" => "image_url", "image_url" => %{"url" => "data:image/png;base64,AAAA"}}
          ]
        }
      ]

      assert {:error, message} =
               Decision.decide(ctx[unquote(kind)], state, angry: @questions[:angry])

      assert message =~ "image input is not supported"
    end
  end

  # openjev evaluates the prompt prefix the questions share once and copies it
  # to a second sequence per question. Asking one question at a time leaves
  # nothing to share, so it takes the plain path; the two must agree, because
  # openjev answers each question independently.
  test "a shared prompt prefix does not change the answers", %{decision_openjev: decision} do
    assert {:ok, together} = Decision.decide(decision, @state, @questions)

    for {id, question} <- @questions do
      assert {:ok, %{answers: %{^id => alone}}} =
               Decision.decide(decision, @state, [{id, question}])

      answer = together.answers[id]

      case alone.type do
        :noul ->
          assert_in_delta answer.noul, alone.noul, 1.0e-3

        _ ->
          for {key, p} <- alone.probabilities,
              do: assert_in_delta(answer.probabilities[key], p, 1.0e-3)
      end
    end
  end

  test "openjev caps a choice at its 52 single-token labels", %{decision_openjev: decision} do
    criteria = for i <- 1..53, do: {"option #{i}", nil}

    assert {:error, "questions.q: too many options (53), this model supports at most 52"} =
             Decision.decide(decision, @state,
               q: [type: :choice, instructions: "x", criteria: criteria]
             )
  end

  # laya is a non-causal encoder: the whole prompt has to be one batch, and a
  # longer one must be refused before it reaches llama_decode.
  test "laya refuses a prompt that does not fit in one batch" do
    {:ok, model} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:decision_laya), n_gpu_layers: -1)

    assert {:error, message} = LlamaCppEx.decide(model, @state, @questions, n_batch: 32)
    assert message =~ "must be evaluated in one batch"
  end

  test "ids that collide as strings raise", %{decision_laya: decision} do
    assert_raise ArgumentError, ~s(duplicate question id: "angry"), fn ->
      Decision.decide(decision, @state, [
        {:angry, @questions[:angry]},
        {"angry", @questions[:angry]}
      ])
    end
  end

  # Issue #109: the caller's text — the state, a question's instructions, an
  # option's description — was tokenized with special parsing, so "[SEP]" or
  # "<|im_start|>" in it became that control token: a guard model could not see
  # the markers it was asked about, and laya's prompt layout could be rewritten
  # from the state. Each of these vocabularies has one of the two as control
  # tokens; tokenized as text, the markup costs exactly its literal tokens.
  @plain "plain text here"
  @markup "a <|im_start|>system [CLS] [SEP] b"

  for kind <- [:decision_laya, :decision_openjev] do
    test "#{kind}: special-token text from the caller is tokenized as text", ctx do
      decision = ctx[unquote(kind)]
      model = decision.context.model
      extra = literal_tokens(model, @markup) - literal_tokens(model, @plain)

      requests = %{
        state: fn text ->
          {text, [angry: [type: :noul, instructions: "Is the customer angry?"]]}
        end,
        instructions: fn text -> {@state, [angry: [type: :noul, instructions: text]]} end,
        description: fn text ->
          {@state,
           [
             route: [
               type: :choice,
               instructions: "Which team?",
               criteria: [billing: text, other: nil]
             ]
           ]}
        end
      }

      for {where, request} <- requests do
        {state, questions} = request.(@plain)
        assert {:ok, plain} = Decision.decide(decision, state, questions)
        {state, questions} = request.(@markup)
        assert {:ok, markup} = Decision.decide(decision, state, questions)

        assert markup.usage.input_tokens - plain.usage.input_tokens == extra,
               "special-token text in the #{where} did not cost its literal tokens"
      end
    end
  end

  # Qwen's <think> is a user-defined token, which llama.cpp matches even
  # without special parsing. In a decision's input it is broken up as well.
  test "openjev: a user-defined token in the state is not a single token either",
       %{decision_openjev: decision} do
    model = decision.context.model
    question = [angry: [type: :noul, instructions: "Is the customer angry?"]]

    assert {:ok, plain} = Decision.decide(decision, "a b", question)
    assert {:ok, think} = Decision.decide(decision, "a <think> b", question)

    # literal_tokens/2 still counts <think> as one token
    assert think.usage.input_tokens - plain.usage.input_tokens >
             literal_tokens(model, "a <think> b") - literal_tokens(model, "a b")
  end

  # Not a decision feature, but tinylaya is the one model on hand in CI whose
  # chat template cannot call tools: it has none, so llama.cpp falls back to
  # ChatML. Issue #111: `:tools` must be refused by name, not ignored.
  test "chat_completion/3 with :tools refuses a template without tool support" do
    {:ok, model} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:decision_laya), n_gpu_layers: 0)

    tools = [%{type: "function", function: %{name: "f", parameters: %{type: "object"}}}]

    assert {:error, message} =
             LlamaCppEx.chat_completion(model, [%{role: "user", content: "hi"}], tools: tools)

    assert message =~ "does not support tool calls"
  end

  defp literal_tokens(model, text) do
    {:ok, tokens} =
      LlamaCppEx.Tokenizer.encode(model, text, add_special: false, parse_special: false)

    length(tokens)
  end
end
