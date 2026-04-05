defmodule LlamaCppEx.KVQuantizationTest do
  @moduledoc """
  Regression tests comparing KV cache quantization (Q8_0) against full
  precision (F16). Validates that quantized KV cache produces equivalent
  output for deterministic prompts.

  Run with:
    LLAMA_MODEL_PATH=path/to/model.gguf mix test test/kv_quantization_test.exs --include slow
  """
  use ExUnit.Case

  @model_path System.get_env("LLAMA_MODEL_PATH")

  @test_cases [
    %{
      name: "arithmetic",
      prompt: "What is 2 + 2? Answer with just the number:",
      expect_contains: "4",
      max_tokens: 8
    },
    %{
      name: "counting",
      prompt: "Count from 1 to 5, separated by commas:",
      expect_contains: "1",
      max_tokens: 16
    },
    %{
      name: "capital city",
      prompt: "The capital of France is",
      expect_contains: "Paris",
      max_tokens: 8
    },
    %{
      name: "largest ocean",
      prompt: "The largest ocean on Earth is the",
      expect_contains: "Pacific",
      max_tokens: 8
    },
    %{
      name: "opposite",
      prompt: "The opposite of hot is",
      expect_contains: "cold",
      max_tokens: 8
    },
    %{
      name: "sequence completion",
      prompt: "Complete the sequence: 2, 4, 6, 8,",
      expect_contains: "10",
      max_tokens: 8
    },
    %{
      name: "color knowledge",
      prompt: "The color of the sky on a clear day is",
      expect_contains: "blue",
      max_tokens: 8
    },
    %{
      name: "continent",
      prompt: "Japan is a country in",
      expect_contains: "Asia",
      max_tokens: 8
    },
    %{
      name: "basic math",
      prompt: "10 multiplied by 5 equals",
      expect_contains: "50",
      max_tokens: 8
    },
    %{
      name: "water formula",
      prompt: "The chemical formula for water is",
      expect_contains: "H2O",
      max_tokens: 8
    }
  ]

  if @model_path && File.exists?(@model_path) do
    describe "KV cache quantization regression" do
      @tag timeout: 120_000
      @tag :slow
      setup do
        :ok = LlamaCppEx.init()
        {:ok, model} = LlamaCppEx.load_model(@model_path, n_gpu_layers: -1)
        %{model: model}
      end

      for tc <- @test_cases do
        @tag :slow
        test "#{tc.name}: F16 vs Q8_0 produce equivalent output", %{model: model} do
          tc = unquote(Macro.escape(tc))

          # Generate with F16 (default, full precision)
          {:ok, ctx_f16} =
            LlamaCppEx.Context.create(model, n_ctx: 2048, type_k: :f16, type_v: :f16)

          {:ok, sampler_f16} = LlamaCppEx.Sampler.create(model, temp: 0.0)
          {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, tc.prompt)

          {:ok, text_f16} =
            LlamaCppEx.Context.generate(ctx_f16, sampler_f16, tokens, max_tokens: tc.max_tokens)

          # Generate with Q8_0 (quantized)
          {:ok, ctx_q8} =
            LlamaCppEx.Context.create(model, n_ctx: 2048, type_k: :q8_0, type_v: :q8_0)

          {:ok, sampler_q8} = LlamaCppEx.Sampler.create(model, temp: 0.0)

          {:ok, text_q8} =
            LlamaCppEx.Context.generate(ctx_q8, sampler_q8, tokens, max_tokens: tc.max_tokens)

          # Both should produce valid non-empty output
          assert byte_size(text_f16) > 0, "F16 output was empty for #{tc.name}"
          assert byte_size(text_q8) > 0, "Q8_0 output was empty for #{tc.name}"

          # Both should contain the expected answer
          f16_lower = String.downcase(text_f16)
          q8_lower = String.downcase(text_q8)
          expected_lower = String.downcase(tc.expect_contains)

          assert String.contains?(f16_lower, expected_lower),
                 "F16 output for #{tc.name} missing '#{tc.expect_contains}': got #{inspect(text_f16)}"

          assert String.contains?(q8_lower, expected_lower),
                 "Q8_0 output for #{tc.name} missing '#{tc.expect_contains}': got #{inspect(text_q8)}"

          # Log comparison for review
          if text_f16 == text_q8 do
            IO.puts("  ✓ #{tc.name}: IDENTICAL")
          else
            IO.puts("  ~ #{tc.name}: EQUIVALENT (both contain '#{tc.expect_contains}')")
            IO.puts("    F16:  #{inspect(String.trim(text_f16))}")
            IO.puts("    Q8_0: #{inspect(String.trim(text_q8))}")
          end
        end
      end
    end
  else
    @tag :skip
    test "KV quantization tests require LLAMA_MODEL_PATH env var" do
      flunk("Set LLAMA_MODEL_PATH to a .gguf file to run KV quantization tests")
    end
  end
end
