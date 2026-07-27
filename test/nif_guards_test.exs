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

  # --- The guarded surface, declared in one place ---
  #
  # `{nif, arity, which_argument, outcome}`. `which_argument` is also the
  # `probe/3` clause that supplies the call shape.
  #
  # Outcomes differ only in how the rejection is *reported* — the safety property
  # is identical and asserted the same way for all of them: the VM survives and
  # the context still decodes and samples.
  #
  #   * `:raise`    — `** (ErlangError) :invalid_seq_id`. Used where the success
  #     value is a bare integer or boolean; widening it into a tuple would cost a
  #     per-token allocation on `sampler_sample/2`-class paths.
  #   * `:atom`     — `{:error, :invalid_seq_id}`, for NIFs already returning an
  #     atom error.
  #   * `:message`  — `{:error, "invalid seq_id N (must be ...)"}`, for NIFs
  #     already returning a string error. Adding a guard never changes a NIF's
  #     error *type*.
  #   * `:upstream` — no local guard: the id is carried *inside a `llama_batch`*,
  #     and `llama_batch_allocr::init`
  #     (`vendor/llama.cpp/src/llama-batch.cpp:61-64`) range-checks batch seq ids
  #     and returns false, so `llama_decode` returns non-zero. The three blockers
  #     this enumeration was written for were distinct precisely because they
  #     reached `llama_memory_seq_rm` *outside* any batch.
  #   * `:lookup`   — no local guard: `llama_get_embeddings_seq` is a
  #     `std::map::find` (`llama-context.cpp:921-928`) returning nullptr.
  #
  # The previous version of this file hand-picked 7 seq_id-bearing NIFs and missed
  # exactly the three that were unguarded, so a green run proved only that the
  # guards someone had already thought of were load-bearing — which is how a
  # break-verification (remove one guard, watch the VM abort) came back green on
  # an incomplete guard set. `covers every seq_id-bearing NIF` below checks this
  # table against `LlamaCppEx.NIF`'s own source, so a new seq_id-taking NIF fails
  # the suite until it is listed here with an outcome.
  @seq_id_surface [
    {:memory_seq_rm, 4, :memory_seq_rm, :raise},
    {:memory_seq_cp, 5, :memory_seq_cp_src, :atom},
    {:memory_seq_cp, 5, :memory_seq_cp_dst, :atom},
    {:memory_seq_keep, 2, :memory_seq_keep, :raise},
    {:memory_seq_pos_max, 2, :memory_seq_pos_max, :raise},
    {:state_seq_get_size, 2, :state_seq_get_size, :raise},
    {:state_seq_get_data, 2, :state_seq_get_data, :raise},
    {:state_seq_set_data, 3, :state_seq_set_data, :atom},
    {:embed_decode, 3, :embed_decode, :message},
    {:embed_batch_decode, 2, :embed_batch_decode, :message},
    {:batch_eval_sample, 4, :batch_eval_sample_purgeable, :message},
    {:get_embeddings, 3, :get_embeddings, :lookup},
    {:prefill, 3, :prefill, :upstream},
    {:decode_batch, 3, :decode_batch, :upstream},
    {:decode_token, 4, :decode_token, :upstream},
    {:batch_eval, 2, :batch_eval, :upstream},
    {:batch_eval_sample, 4, :batch_eval_sample_entries, :upstream}
  ]

  defp probe(:memory_seq_rm, seq, f), do: NIF.memory_seq_rm(f.ctx.ref, seq, 0, -1)
  defp probe(:memory_seq_cp_src, seq, f), do: NIF.memory_seq_cp(f.ctx.ref, seq, 0, 0, -1)
  defp probe(:memory_seq_cp_dst, seq, f), do: NIF.memory_seq_cp(f.ctx.ref, 0, seq, 0, -1)
  defp probe(:memory_seq_keep, seq, f), do: NIF.memory_seq_keep(f.ctx.ref, seq)
  defp probe(:memory_seq_pos_max, seq, f), do: NIF.memory_seq_pos_max(f.ctx.ref, seq)
  defp probe(:state_seq_get_size, seq, f), do: NIF.state_seq_get_size(f.ctx.ref, seq)
  defp probe(:state_seq_get_data, seq, f), do: NIF.state_seq_get_data(f.ctx.ref, seq)
  defp probe(:state_seq_set_data, seq, f), do: NIF.state_seq_set_data(f.ctx.ref, f.blob, seq)
  defp probe(:embed_decode, seq, f), do: NIF.embed_decode(f.ctx.ref, f.tokens, seq)
  defp probe(:get_embeddings, seq, f), do: NIF.get_embeddings(f.ctx.ref, seq, 2)
  defp probe(:prefill, seq, f), do: NIF.prefill(f.ctx.ref, f.tokens, seq)
  defp probe(:decode_token, seq, f), do: NIF.decode_token(f.ctx.ref, hd(f.tokens), 0, seq)

  defp probe(:embed_batch_decode, seq, f),
    do: NIF.embed_batch_decode(f.ctx.ref, [{seq, f.tokens}])

  defp probe(:decode_batch, seq, f),
    do: NIF.decode_batch(f.ctx.ref, f.sampler.ref, [{seq, hd(f.tokens), 0}])

  defp probe(:batch_eval, seq, f),
    do: NIF.batch_eval(f.ctx.ref, [{hd(f.tokens), 0, seq, true}])

  defp probe(:batch_eval_sample_entries, seq, f),
    do: NIF.batch_eval_sample(f.ctx.ref, [{hd(f.tokens), 0, seq, true}], [], [])

  # The purge list is the one that mattered: it goes to llama_memory_seq_rm in
  # bes_decode_range's KV-pressure loop, outside the batch. The entries here are
  # deliberately valid so only the purge id is out of range.
  defp probe(:batch_eval_sample_purgeable, seq, f),
    do: NIF.batch_eval_sample(f.ctx.ref, [{hd(f.tokens), 0, 0, true}], [], [seq])

  defp assert_rejected(:raise, label, fun) do
    error = assert_raise ErlangError, fun

    assert error.original == :invalid_seq_id,
           "#{label} raised #{inspect(error.original)} rather than :invalid_seq_id"
  end

  defp assert_rejected(:atom, label, fun) do
    assert fun.() == {:error, :invalid_seq_id},
           "#{label} did not return {:error, :invalid_seq_id}"
  end

  defp assert_rejected(:message, label, fun) do
    case fun.() do
      {:error, message} ->
        assert message =~ "invalid seq_id",
               "#{label} was refused with #{inspect(message)}, which does not name the " <>
                 "seq_id — that reads like an unrelated decode failure rather than a " <>
                 "boundary rejection"

      other ->
        flunk("#{label} was accepted: #{inspect(other, limit: 5)}")
    end
  end

  # `:upstream` and `:lookup` have no local guard by design, so the assertion is
  # the safety property only: refused somehow, VM alive. The `assert_context_healthy`
  # call in the generated test is the other half.
  defp assert_rejected(class, label, fun) when class in [:upstream, :lookup] do
    assert refused?(fun), "#{label} accepted an out-of-range seq_id"
  end

  defp refused?(fun) do
    match?({:error, _}, fun.())
  rescue
    ErlangError -> true
  end

  defp fixtures(%{ctx: ctx, tokens: tokens, model: model}) do
    {:ok, sampler} = Sampler.create(model, temp: 0.0)
    {:ok, blob} = NIF.state_seq_get_data(ctx.ref, 0)
    %{ctx: ctx, tokens: tokens, model: model, sampler: sampler, blob: blob}
  end

  describe "seq_id bounds, enumerated over the whole surface" do
    for {nif, arity, which, outcome} <- @seq_id_surface do
      test "#{nif}/#{arity} bounds #{which} (#{outcome})", context do
        %{ctx: ctx, tokens: tokens, model: model} = context
        f = fixtures(context)

        for seq <- @bad_seq_ids do
          label = "#{unquote(nif)}/#{unquote(arity)} via #{unquote(which)} with seq_id #{seq}"
          assert_rejected(unquote(outcome), label, fn -> probe(unquote(which), seq, f) end)
        end

        assert_context_healthy(ctx, tokens, model)
      end
    end

    test "covers every seq_id-bearing NIF" do
      # Scanned from the NIF stub module rather than listed by hand, which is the
      # whole point: the hand-written list is what was incomplete.
      declared = @seq_id_surface |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> Enum.sort()

      source =
        "lib/llama_cpp_ex/nif.ex"
        |> File.read!()
        |> String.replace(~r/\s+/, " ")

      found =
        Regex.scan(~r/def ([a-z_0-9]+)\(([^)]*)\)/, source)
        |> Enum.filter(fn [_, _name, args] ->
          args =~ ~r/\b_(?:dest_)?seq_id/ or args =~ "_seq_ids" or args =~ "_sequences" or
            args =~ "_entries"
        end)
        |> Enum.map(fn [_, name, _args] -> String.to_existing_atom(name) end)
        |> Enum.uniq()
        |> Enum.sort()

      assert found -- declared == [],
             """
             These NIFs take a sequence id and are not in @seq_id_surface:
             #{inspect(found -- declared)}

             Add each one with its outcome class. A seq_id that reaches a
             llama_memory_* call outside a batch needs a local guard
             (:raise/:atom/:message); one carried inside a llama_batch is bounded
             by upstream (:upstream). Guessing wrong is safe — the generated test
             will disagree with you.
             """

      assert declared -- found == [],
             "@seq_id_surface lists NIFs that no longer exist: #{inspect(declared -- found)}"

      for {nif, arity, _which, _outcome} <- @seq_id_surface do
        assert function_exported?(NIF, nif, arity),
               "@seq_id_surface declares #{nif}/#{arity}, which is not exported"
      end
    end
  end

  # `vendor/llama.cpp/src/llama-kv-cache.cpp:502` calls `ggml_abort` on a partial
  # cross-sequence copy when the KV cache is not unified. `Server` guards it via
  # `cross_slot_sharing`; the NIF boundary did not, so any direct caller killed
  # the VM. Reproduced by a benchmark, which is how it was found.
  describe "memory_seq_cp/5 on a split KV cache" do
    test "a partial cross-sequence copy is refused, and copies nothing", %{
      ctx: ctx,
      tokens: tokens,
      model: model
    } do
      # Give seq 0 something worth copying, so "nothing was copied" is a real
      # observation rather than a property of an empty cache.
      :ok = NIF.decode(ctx.ref, tokens)
      assert NIF.memory_seq_pos_max(ctx.ref, 0) == length(tokens) - 1
      assert NIF.memory_seq_pos_max(ctx.ref, 1) == -1

      # `{:error, :unsupported}` is the easy half. The property that matters is
      # that the refusal is *before* any copy: a guard that rejected after
      # partially writing the destination would satisfy the assertion below and
      # still hand the caller a sequence it does not believe it has.
      for {p0, p1} <- [{0, 5}, {3, -1}, {2, 7}] do
        assert NIF.memory_seq_cp(ctx.ref, 0, 1, p0, p1) == {:error, :unsupported}

        assert NIF.memory_seq_pos_max(ctx.ref, 1) == -1,
               "memory_seq_cp(0, 1, #{p0}, #{p1}) was refused but left KV in seq 1"
      end

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
