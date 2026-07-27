defmodule LlamaCppEx.NIFGuardsTest do
  @moduledoc """
  Boundary guards at the NIF surface.

  Every call below used to reach a `GGML_ASSERT` and `abort()` the whole BEAM —
  no exception, no supervisor, no `erl_crash.dump` worth reading. `GGML_ASSERT`
  is not `NDEBUG`-gated, so a bad integer arriving from Elixir was a process
  image kill; the `memory_seq_cp` case was found only because a benchmark
  happened to trigger it.

  So each test does two things: assert the guard's error, and then assert the
  context is *still usable* — a real decode and sample on the same context. A
  guard that returned an error but left the KV cache or the graph in a bad state
  would pass the first half and fail the second.

  Runs against `LLAMA_SMOKE_GEN_MODEL`; see `test/test_helper.exs`.
  """
  use ExUnit.Case, async: false

  alias LlamaCppEx.{Context, Model, NIF, Sampler, Tokenizer}

  @moduletag :smoke
  @moduletag timeout: 300_000

  # n_seq_max: 2 puts the out-of-range boundary at 2 rather than somewhere large,
  # and keeps the KV allocation small. n_gpu_layers: 0 keeps this on the CPU: the
  # guards are backend-independent and a 135M model is fast enough there.
  @n_seq_max 2

  setup_all do
    :ok = LlamaCppEx.init()
    {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen), n_gpu_layers: 0)
    %{model: model}
  end

  setup %{model: model} do
    {:ok, ctx} = Context.create(model, n_ctx: 512, n_seq_max: @n_seq_max, kv_unified: false)
    {:ok, tokens} = Tokenizer.encode(model, "The capital of France is")
    %{ctx: ctx, tokens: tokens}
  end

  # Out-of-range seq ids, both directions. `@n_seq_max` itself is the first
  # invalid value — off-by-one in the guard would let it through.
  @bad_seq_ids [@n_seq_max, @n_seq_max + 1, 99, -1, -99]

  defp assert_erlang_error(reason, fun) do
    error = assert_raise ErlangError, fun
    assert error.original == reason
  end

  # The context still works: the KV clears, a prefill lands at the positions it
  # should, and sampling reads logits from it. Sequence 0 is cleared first so the
  # check does not depend on what the test did before calling it.
  defp assert_context_healthy(ctx, tokens, model) do
    assert NIF.memory_seq_rm(ctx.ref, 0, 0, -1) == true
    assert :ok = NIF.decode(ctx.ref, tokens)
    assert NIF.memory_seq_pos_max(ctx.ref, 0) == length(tokens) - 1

    {:ok, sampler} = Sampler.create(model, temp: 0.0)
    assert is_integer(NIF.sampler_sample(sampler.ref, ctx.ref))
  end

  describe "seq_id bounds (raising NIFs)" do
    test "memory_seq_rm/4", %{ctx: ctx, tokens: tokens, model: model} do
      for seq <- @bad_seq_ids do
        assert_erlang_error(:invalid_seq_id, fn -> NIF.memory_seq_rm(ctx.ref, seq, 0, -1) end)
      end

      assert NIF.memory_seq_rm(ctx.ref, 0, 0, -1) == true
      assert NIF.memory_seq_rm(ctx.ref, @n_seq_max - 1, 0, -1) == true
      assert_context_healthy(ctx, tokens, model)
    end

    test "memory_seq_keep/2", %{ctx: ctx, tokens: tokens, model: model} do
      for seq <- @bad_seq_ids do
        assert_erlang_error(:invalid_seq_id, fn -> NIF.memory_seq_keep(ctx.ref, seq) end)
      end

      assert NIF.memory_seq_keep(ctx.ref, 0) == :ok
      assert_context_healthy(ctx, tokens, model)
    end

    test "memory_seq_pos_max/2", %{ctx: ctx, tokens: tokens, model: model} do
      for seq <- @bad_seq_ids do
        assert_erlang_error(:invalid_seq_id, fn -> NIF.memory_seq_pos_max(ctx.ref, seq) end)
      end

      assert is_integer(NIF.memory_seq_pos_max(ctx.ref, 0))
      assert_context_healthy(ctx, tokens, model)
    end

    test "state_seq_get_size/2", %{ctx: ctx, tokens: tokens, model: model} do
      for seq <- @bad_seq_ids do
        assert_erlang_error(:invalid_seq_id, fn -> NIF.state_seq_get_size(ctx.ref, seq) end)
      end

      assert NIF.state_seq_get_size(ctx.ref, 0) > 0
      assert_context_healthy(ctx, tokens, model)
    end

    test "state_seq_get_data/2", %{ctx: ctx, tokens: tokens, model: model} do
      for seq <- @bad_seq_ids do
        assert_erlang_error(:invalid_seq_id, fn -> NIF.state_seq_get_data(ctx.ref, seq) end)
      end

      assert {:ok, blob} = NIF.state_seq_get_data(ctx.ref, 0)
      assert byte_size(blob) > 0
      assert_context_healthy(ctx, tokens, model)
    end
  end

  describe "seq_id bounds (error-returning NIFs)" do
    test "memory_seq_cp/5 rejects either end being out of range", %{
      ctx: ctx,
      tokens: tokens,
      model: model
    } do
      for seq <- @bad_seq_ids do
        assert NIF.memory_seq_cp(ctx.ref, seq, 0, 0, -1) == {:error, :invalid_seq_id}
        assert NIF.memory_seq_cp(ctx.ref, 0, seq, 0, -1) == {:error, :invalid_seq_id}
      end

      assert_context_healthy(ctx, tokens, model)
    end

    test "state_seq_set_data/3 rejects an out-of-range destination", %{
      ctx: ctx,
      tokens: tokens,
      model: model
    } do
      :ok = NIF.decode(ctx.ref, tokens)
      {:ok, blob} = NIF.state_seq_get_data(ctx.ref, 0)

      for seq <- @bad_seq_ids do
        assert NIF.state_seq_set_data(ctx.ref, blob, seq) == {:error, :invalid_seq_id}
      end

      # The same blob into a valid sequence still works, so the guard rejected
      # the id rather than the payload.
      assert {:ok, bytes} = NIF.state_seq_set_data(ctx.ref, blob, 1)
      assert bytes == byte_size(blob)
      assert_context_healthy(ctx, tokens, model)
    end
  end

  # `vendor/llama.cpp/src/llama-kv-cache.cpp:502` calls `ggml_abort` on a partial
  # cross-sequence copy when the KV cache is not unified. `Server` guards it via
  # `cross_slot_sharing`; the NIF boundary did not, so any direct caller killed
  # the VM. Reproduced by a benchmark, which is how it was found.
  describe "memory_seq_cp/5 on a split KV cache" do
    test "a partial cross-sequence copy is refused", %{ctx: ctx, tokens: tokens, model: model} do
      assert NIF.memory_seq_cp(ctx.ref, 0, 1, 0, 5) == {:error, :unsupported}
      assert NIF.memory_seq_cp(ctx.ref, 0, 1, 3, -1) == {:error, :unsupported}
      assert NIF.memory_seq_cp(ctx.ref, 0, 1, 2, 7) == {:error, :unsupported}
      assert_context_healthy(ctx, tokens, model)
    end

    test "a full copy (p0 <= 0, p1 < 0) is allowed", %{ctx: ctx} do
      assert NIF.memory_seq_cp(ctx.ref, 0, 1, 0, -1) == :ok
      assert NIF.memory_seq_cp(ctx.ref, 0, 1, -1, -1) == :ok
    end

    test "a copy within one sequence is not a cross-sequence copy", %{ctx: ctx} do
      assert NIF.memory_seq_cp(ctx.ref, 0, 0, 0, 5) == :ok
    end

    test "a unified cache permits the partial copy", %{model: model} do
      {:ok, unified} = Context.create(model, n_ctx: 512, n_seq_max: 2, kv_unified: true)

      assert NIF.memory_seq_cp(unified.ref, 0, 1, 0, 5) == :ok
    end
  end

  describe "logits index bounds" do
    test "sampling before any decode has no logits to read", %{
      ctx: ctx,
      tokens: tokens,
      model: model
    } do
      {:ok, sampler} = Sampler.create(model, temp: 0.0)

      assert_erlang_error(:invalid_index, fn -> NIF.sampler_sample(sampler.ref, ctx.ref) end)

      assert_erlang_error(:invalid_index, fn -> NIF.sampler_sample_at(sampler.ref, ctx.ref, 0) end)

      assert_context_healthy(ctx, tokens, model)
    end

    test "sampler_sample_at/3 rejects an out-of-range index", %{
      ctx: ctx,
      tokens: tokens,
      model: model
    } do
      {:ok, sampler} = Sampler.create(model, temp: 0.0)
      :ok = NIF.decode(ctx.ref, tokens)

      # -1 is llama.cpp's "last output" convention and is legitimate; anything
      # past the batch, or further back than the one output it produced, is not.
      for index <- [length(tokens), length(tokens) + 1, 100, 9999, -2, -7, -9999] do
        assert_erlang_error(:invalid_index, fn ->
          NIF.sampler_sample_at(sampler.ref, ctx.ref, index)
        end)
      end

      # Only the last prompt position requested logits, so the earlier ones have
      # none — a valid index into the batch is still an invalid index into the
      # logits, which is the case that used to abort.
      for index <- 0..(length(tokens) - 2) do
        assert_erlang_error(:invalid_index, fn ->
          NIF.sampler_sample_at(sampler.ref, ctx.ref, index)
        end)
      end

      # The indices that do have logits work, and agree with sampler_sample/2.
      last = length(tokens) - 1
      token = NIF.sampler_sample_at(sampler.ref, ctx.ref, last)
      assert is_integer(token)
      assert NIF.sampler_sample_at(sampler.ref, ctx.ref, -1) == token
      assert NIF.sampler_sample(sampler.ref, ctx.ref) == token
    end
  end

  # Deserializing an untrusted binary straight into the KV cache. A short or
  # corrupt blob used to be read as though it were a valid header.
  describe "state_seq_set_data/3 blob validation" do
    test "refuses blobs that cannot be a KV state", %{ctx: ctx, tokens: tokens, model: model} do
      blobs = [
        {"empty", <<>>},
        {"3 bytes", <<1, 2, 3>>},
        {"4 KiB of noise", :crypto.strong_rand_bytes(4096)},
        {"a truncated real blob", truncated_state(ctx, tokens)}
      ]

      for {label, blob} <- blobs do
        assert NIF.state_seq_set_data(ctx.ref, blob, 1) == {:error, :invalid_state},
               "#{label} should be refused"
      end

      assert_context_healthy(ctx, tokens, model)
    end

    test "refuses a term that is not a binary at all", %{ctx: ctx} do
      for term <- [:nope, 42, [1, 2, 3]] do
        assert NIF.state_seq_set_data(ctx.ref, term, 1) == {:error, :invalid_state}
      end
    end

    test "a round-tripped blob is accepted", %{ctx: ctx, tokens: tokens} do
      :ok = NIF.decode(ctx.ref, tokens)
      {:ok, blob} = NIF.state_seq_get_data(ctx.ref, 0)

      assert {:ok, bytes} = NIF.state_seq_set_data(ctx.ref, blob, 1)
      assert bytes == byte_size(blob)
      assert NIF.memory_seq_pos_max(ctx.ref, 1) == length(tokens) - 1
    end
  end

  defp truncated_state(ctx, tokens) do
    :ok = NIF.decode(ctx.ref, tokens)
    {:ok, blob} = NIF.state_seq_get_data(ctx.ref, 0)
    :ok = NIF.memory_seq_keep(ctx.ref, 0)
    binary_part(blob, 0, div(byte_size(blob), 2))
  end

  # A grammar that fails to compile used to be dropped silently: `if (grammar)
  # chain_add(...)`, so a caller who asked for JSON got unconstrained output.
  # That is a validation bypass, not a cosmetic issue. Unbounded recursion in the
  # parser was separately a stack overflow.
  describe "Sampler.create/2 grammar validation" do
    test "unparseable GBNF is rejected, not silently ignored", %{model: model} do
      bad = [
        "this is not gbnf {{{",
        ~s(::= "x"),
        "root ::= <unclosed",
        ~s(root ::= "unterminated),
        "root ::= undefined_rule",
        "root ::= [a-",
        # No `root` rule at all — parses, then fails to resolve.
        ~s(rot ::= "x")
      ]

      for grammar <- bad do
        assert Sampler.create(model, grammar: grammar) == {:error, :invalid_grammar},
               "#{inspect(grammar)} should be rejected"
      end
    end

    test "an empty grammar means no grammar, not a broken one", %{model: model} do
      # `Sampler.create/2` defaults `:grammar` to "", so an empty string has to
      # stay the "unconstrained" signal rather than becoming a parse error.
      assert {:ok, %Sampler{}} = Sampler.create(model, grammar: "")
      assert {:ok, %Sampler{}} = Sampler.create(model, temp: 0.0)
    end

    test "a grammar over 1 MiB is rejected before parsing", %{model: model} do
      over = ~s(root ::= ") <> String.duplicate("a", 1_048_577) <> ~s(")

      assert Sampler.create(model, grammar: over) == {:error, :invalid_grammar}

      assert Sampler.create(model, grammar: String.duplicate("a", 2_000_000)) ==
               {:error, :invalid_grammar}
    end

    test "nesting deeper than 64 groups is rejected", %{model: model} do
      assert Sampler.create(model, grammar: nested(200)) == {:error, :invalid_grammar}
      assert Sampler.create(model, grammar: nested(65)) == {:error, :invalid_grammar}

      # The bound is where it is documented, not "every nested grammar fails".
      assert {:ok, %Sampler{}} = Sampler.create(model, grammar: nested(60))
    end

    test "valid GBNF still compiles", %{model: model} do
      assert {:ok, %Sampler{} = sampler} =
               Sampler.create(model, grammar: ~s(root ::= "yes" | "no"))

      assert is_reference(sampler.ref)
    end

    test "a JSON schema grammar constrains generation end to end", %{model: model} do
      gbnf = LlamaCppEx.Grammar.from_json_schema!(%{"type" => "boolean"})

      assert {:ok, %Sampler{}} = Sampler.create(model, grammar: gbnf)
    end
  end

  defp nested(depth) do
    "root ::= " <> String.duplicate("(", depth) <> ~s("x") <> String.duplicate(")", depth)
  end

  # The use-after-free: `sampler_init` captured a raw `const llama_vocab*` from
  # the model with no ownership link, so dropping the model term left the sampler
  # dereferencing freed heap on the next reset/accept.
  describe "a sampler keeps its model alive" do
    test "%Sampler{} carries the model it was built from", %{model: model} do
      {:ok, sampler} = Sampler.create(model, temp: 0.0)
      assert %Model{} = sampler.model
      assert sampler.model.ref == model.ref
    end

    test "reset/1 works after the caller's model term is gone" do
      # The model is loaded and dropped inside the closure, so the only remaining
      # reference is the one the sampler holds.
      sampler =
        (fn ->
           {:ok, m} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen), n_gpu_layers: 0)
           {:ok, s} = Sampler.create(m, grammar: ~s(root ::= "yes" | "no"), temp: 0.0)
           s
         end).()

      :erlang.garbage_collect()
      Process.sleep(50)
      :erlang.garbage_collect()

      assert Sampler.reset(sampler) == :ok
      assert %Model{} = sampler.model
    end
  end

  describe "the VM survives every guard" do
    test "a full generation still works after tripping all of them", %{
      ctx: ctx,
      model: model,
      tokens: tokens
    } do
      # Belt and braces: an abort() would take the whole node down, so what this
      # really asserts is that none of the guards left the library unusable.
      for seq <- @bad_seq_ids do
        assert_erlang_error(:invalid_seq_id, fn -> NIF.memory_seq_rm(ctx.ref, seq, 0, -1) end)
        assert NIF.memory_seq_cp(ctx.ref, seq, 0, 0, -1) == {:error, :invalid_seq_id}
      end

      assert NIF.memory_seq_cp(ctx.ref, 0, 1, 0, 5) == {:error, :unsupported}
      assert NIF.state_seq_set_data(ctx.ref, <<1, 2, 3>>, 1) == {:error, :invalid_state}
      assert Sampler.create(model, grammar: "{{{") == {:error, :invalid_grammar}

      assert node() == :nonode@nohost or is_atom(node())

      assert {:ok, text} =
               LlamaCppEx.generate(model, "The capital of France is", max_tokens: 8, temp: 0.0)

      assert byte_size(text) > 0
      assert_context_healthy(ctx, tokens, model)
    end
  end
end
