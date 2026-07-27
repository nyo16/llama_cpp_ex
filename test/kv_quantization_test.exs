defmodule LlamaCppEx.KVQuantizationTest do
  @moduledoc """
  Regression tests comparing KV cache quantization (Q8_0) against full
  precision (F16).

  What this suite can honestly assert is *structural*: greedy decoding under
  each KV dtype must be reproducible and must produce valid, non-empty UTF-8.
  It used to assert that the output contained a specific word ("Paris",
  "Pacific", "cold"), which measures the model's world knowledge rather than
  the KV cache, fails outright on a small model, and turns every
  `vendor/llama.cpp` bump into a flake. The F16 and Q8_0 texts are still
  compared and any difference reported, but agreement is not asserted —
  quantization is lossy by definition, so divergence is expected behaviour, not
  a regression.

  Run with:
    LLAMA_SMOKE_GEN_MODEL=path/to/model.gguf mix test test/kv_quantization_test.exs --include slow
  """
  use ExUnit.Case

  @seed 20_260_727

  @test_cases [
    %{name: "arithmetic", prompt: "What is 2 + 2? Answer with just the number:", max_tokens: 8},
    %{name: "counting", prompt: "Count from 1 to 5, separated by commas:", max_tokens: 16},
    %{name: "capital city", prompt: "The capital of France is", max_tokens: 8},
    %{name: "largest ocean", prompt: "The largest ocean on Earth is the", max_tokens: 8},
    %{name: "opposite", prompt: "The opposite of hot is", max_tokens: 8},
    %{name: "sequence completion", prompt: "Complete the sequence: 2, 4, 6, 8,", max_tokens: 8},
    %{name: "color knowledge", prompt: "The color of the sky on a clear day is", max_tokens: 8},
    %{name: "continent", prompt: "Japan is a country in", max_tokens: 8},
    %{name: "basic math", prompt: "10 multiplied by 5 equals", max_tokens: 8},
    %{name: "water formula", prompt: "The chemical formula for water is", max_tokens: 8}
  ]

  describe "KV cache quantization regression" do
    # One gate tag for the suite. `@tag` on a `setup` block does nothing, and a
    # per-test repeat of the describetag is noise — both were here before.
    @describetag :slow
    @describetag timeout: 120_000

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen), n_gpu_layers: -1)
      %{model: model}
    end

    for tc <- @test_cases do
      test "#{tc.name}: F16 and Q8_0 KV caches both decode deterministically", %{model: model} do
        tc = unquote(Macro.escape(tc))
        {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, tc.prompt)

        text_f16 = generate(model, tokens, :f16, tc.max_tokens)
        text_q8 = generate(model, tokens, :q8_0, tc.max_tokens)

        for {label, text} <- [{"F16", text_f16}, {"Q8_0", text_q8}] do
          assert byte_size(text) > 0, "#{label} output was empty for #{tc.name}"

          assert String.valid?(text),
                 "#{label} output for #{tc.name} was not valid UTF-8: #{inspect(text)}"
        end

        # Greedy sampling with a fixed seed is deterministic by contract. A KV
        # cache reading uninitialized or stale memory breaks exactly this, and
        # unlike a word-match it holds for whichever model the env var names.
        assert generate(model, tokens, :f16, tc.max_tokens) == text_f16,
               "F16 decoding was not reproducible for #{tc.name}"

        assert generate(model, tokens, :q8_0, tc.max_tokens) == text_q8,
               "Q8_0 decoding was not reproducible for #{tc.name}"

        report(tc.name, text_f16, text_q8)
      end
    end
  end

  # A fresh context and sampler per call: reproducibility *across contexts* is
  # the property under test, so no state may be carried between runs.
  defp generate(model, tokens, kv_type, max_tokens) do
    {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 2048, type_k: kv_type, type_v: kv_type)
    {:ok, sampler} = LlamaCppEx.Sampler.create(model, temp: 0.0, seed: @seed)
    {:ok, text} = LlamaCppEx.Context.generate(ctx, sampler, tokens, max_tokens: max_tokens)
    text
  end

  # Reported, never asserted: quantization is lossy, so the two texts may
  # legitimately diverge. This is what a human reads after a llama.cpp bump to
  # judge whether the loss got worse.
  defp report(name, text, text), do: IO.puts("  = #{name}: IDENTICAL")

  defp report(name, text_f16, text_q8) do
    IO.puts("  ~ #{name}: DIVERGED")
    IO.puts("    F16:  #{inspect(String.trim(text_f16))}")
    IO.puts("    Q8_0: #{inspect(String.trim(text_q8))}")
  end
end
