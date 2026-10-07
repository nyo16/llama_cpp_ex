defmodule LlamaCppEx.DecisionClefTest do
  @moduledoc """
  The clef readout against a real Clef Flash, behind `:decision_clef`.

  Clef decides every question of a request jointly, in one prompt, reading the
  score of option i from the embeddings output at row i. No tiny random-weight
  clef exists, so unlike `LlamaCppEx.DecisionTest` this runs the real 9B model
  (Cloudflare/clef-flash; bartowski/Cloudflare_clef-flash-GGUF or
  ggml-org/Clef-Flash-GGUF) and asserts what it answers, not just the shape.

      GGML_METAL_NO_RESIDENCY=1 \\
      LLAMA_SMOKE_DECISION_CLEF_MODEL=/path/to/Cloudflare_clef-flash-Q8_0.gguf \\
        mix test test/decision_clef_test.exs --include decision_clef

  The model is loaded with `n_gpu_layers: -1` on purpose: the routing test asks
  whether clef works on the build and device you have. It caught llama.cpp's
  Metal MUL_MAT+ADD fusion picking the wrong residual (billing 0.28 where the
  CPU gave 0.977; fixed upstream in #30100, in the pin since v0.8.56).
  """
  use ExUnit.Case, async: false

  alias LlamaCppEx.Decision

  @moduletag :decision_clef
  @moduletag timeout: 300_000

  @questions [
    route: [
      type: :choice,
      instructions: "Which team should handle this?",
      # Deliberately not in key order: clef shows a choice's options sorted by
      # key, so the answer must be mapped back to the keys as the caller gave them.
      criteria: [technical: nil, billing: nil, shipping: nil]
    ],
    angry: [type: :noul, instructions: "Is the customer angry?"],
    urgency: [
      type: :score,
      instructions: "How urgent is this?",
      criteria: ["can wait", "this week", "today", "right now"]
    ]
  ]

  setup_all do
    :ok = LlamaCppEx.init()

    {:ok, model} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:decision_clef), n_gpu_layers: -1)

    assert Decision.model_type(model) == :clef
    {:ok, decision} = Decision.new(model)
    {:ok, decision: decision}
  end

  test "routes by the content of the state, keyed as the caller wrote the options",
       %{decision: decision} do
    billing =
      "Customer message: I was charged twice for my order last week and nobody has replied."

    crash = "Customer message: The app crashes when I open the settings page on Android 15."

    assert {:ok, %{answers: a}} = Decision.decide(decision, billing, @questions)
    assert {:ok, %{answers: b}} = Decision.decide(decision, crash, @questions)

    assert a.route.choice == :billing
    assert b.route.choice == :technical
    assert a.route.probabilities.billing > 0.9
    assert b.route.probabilities.technical > 0.9

    # noul and score are probabilities, so the readout has to be comparable
    # across requests: the double charge with no reply is the angrier of the two.
    assert a.angry.noul > b.angry.noul
    assert_in_delta Enum.sum(Map.values(a.urgency.probabilities)), 1.0, 1.0e-4
  end

  # Clef pays for the state once per request, not once per question. Asking the
  # three questions together must cost fewer input tokens than asking each one
  # on its own, which is what distinguishes the joint path from openjev's.
  test "decides all questions in one prompt", %{decision: decision} do
    state = "Customer message: I was charged twice for my order last week and nobody has replied."

    assert {:ok, %{usage: %{input_tokens: together}}} =
             Decision.decide(decision, state, @questions)

    separately =
      for {id, question} <- @questions, reduce: 0 do
        acc ->
          assert {:ok, %{usage: %{input_tokens: n}}} =
                   Decision.decide(decision, state, [{id, question}])

          acc + n
      end

    assert together < separately
  end

  test "caps a choice at 255 options", %{decision: decision} do
    criteria = for i <- 1..256, do: {"option #{i}", nil}

    assert {:error, "questions.q: too many options (256), this model supports at most 255"} =
             Decision.decide(decision, "x",
               q: [type: :choice, instructions: "pick", criteria: criteria]
             )
  end
end
