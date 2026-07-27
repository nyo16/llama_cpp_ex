defmodule LlamaCppEx.MTPTest do
  use ExUnit.Case, async: true

  alias LlamaCppEx.MTP

  # `LlamaCppEx.MTP` had no tests at all: the audit's coverage gap listed it as
  # gated on an env var that was never set anywhere. Its input validation and
  # struct contract need no model and belong in the default suite; everything
  # that reaches the NIF is tagged `:mtp` and runs against
  # LLAMA_SMOKE_MTP_MODEL.

  describe "init/2 argument validation" do
    # The n_draft guard runs before any context is created, so a model struct
    # with a nil ref never reaches the NIF — the same trick the option-validation
    # tests use.
    @unloaded %LlamaCppEx.Model{ref: nil}

    test "rejects a non-positive :n_draft" do
      for n <- [0, -1, -10] do
        assert MTP.init(@unloaded, n_draft: n) == {:error, ":n_draft must be a positive integer"},
               "n_draft: #{n} should be rejected"
      end
    end

    test "rejects a :n_draft that is not an integer" do
      for n <- [1.0, 2.5, "3", :three, nil, [3]] do
        assert MTP.init(@unloaded, n_draft: n) == {:error, ":n_draft must be a positive integer"},
               "n_draft: #{inspect(n)} should be rejected"
      end
    end
  end

  describe "the %MTP{} struct" do
    test "enforces every field, because each one is a live NIF resource" do
      # A partially built MTP would hand a nil reference to the NIF.
      for missing <- [:main_ctx, :mtp_ctx, :spec_ref, :n_draft] do
        fields =
          %{main_ctx: :ctx, mtp_ctx: :ctx, spec_ref: :ref, n_draft: 3} |> Map.delete(missing)

        assert_raise ArgumentError, ~r/#{missing}/, fn -> struct!(MTP, fields) end
      end
    end
  end

  describe "speculative decoding against a real MTP model" do
    @describetag :mtp
    @moduletag timeout: 300_000

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp), n_gpu_layers: -1)
      {:ok, mtp} = MTP.init(model, n_ctx: 2048, n_draft: 3)
      %{model: model, mtp: mtp}
    end

    test "init/2 builds both contexts and defaults n_draft to 3", %{model: model} do
      assert {:ok, mtp} = MTP.init(model, n_ctx: 512)
      assert mtp.n_draft == 3
      assert %LlamaCppEx.Context{} = mtp.main_ctx
      assert %LlamaCppEx.Context{} = mtp.mtp_ctx
      assert is_reference(mtp.spec_ref)
    end

    test "generate/3 returns non-empty deterministic text", %{mtp: mtp} do
      opts = [max_tokens: 16, temp: 0.0, seed: 7]

      assert {:ok, text} = MTP.generate(mtp, "The capital of France is", opts)
      assert byte_size(text) > 0
      assert String.valid?(text)

      # Speculative decoding must not change *what* is generated, only how fast.
      assert {:ok, ^text} = MTP.generate(mtp, "The capital of France is", opts)
    end

    test "stream/3 yields the same text generate/3 returns", %{mtp: mtp} do
      opts = [max_tokens: 16, temp: 0.0, seed: 7]

      streamed = mtp |> MTP.stream("Count to five:", opts) |> Enum.join()
      assert {:ok, ^streamed} = MTP.generate(mtp, "Count to five:", opts)
    end

    test "stream_events/3 emits only documented events, ending with a terminal one", %{mtp: mtp} do
      events =
        mtp
        |> MTP.stream_events("The capital of France is", max_tokens: 8, temp: 0.0)
        |> Enum.to_list()

      assert events != []

      for event <- events do
        assert event_kind(event) != :undocumented, "undocumented event: #{inspect(event)}"
      end

      assert event_kind(List.last(events)) == :terminal
    end

    test "emit_stats_every yields stats events that stream/3 filters out", %{mtp: mtp} do
      opts = [max_tokens: 16, temp: 0.0, emit_stats_every: 2]

      events = mtp |> MTP.stream_events("Tell me a story:", opts) |> Enum.to_list()
      assert Enum.any?(events, &match?({:stats, _}, &1))

      # stream/3 is the text-only view of the same events.
      pieces = mtp |> MTP.stream("Tell me a story:", opts) |> Enum.to_list()
      assert Enum.all?(pieces, &is_binary/1)
    end

    test "stats/1 counts drafts and accepts after generation", %{mtp: mtp} do
      before = MTP.stats(mtp)
      assert before.n_draft == 3

      {:ok, _} = MTP.generate(mtp, "The capital of France is", max_tokens: 16, temp: 0.0)
      after_gen = MTP.stats(mtp)

      assert after_gen.iters > before.iters
      assert after_gen.drafts_generated > before.drafts_generated
      assert after_gen.tokens_emitted > before.tokens_emitted
      assert after_gen.acceptance_rate >= 0.0 and after_gen.acceptance_rate <= 1.0
      assert after_gen.drafts_accepted <= after_gen.drafts_generated
      assert %{draft: _, verify: _, sample: _, total: _} = after_gen.timing_us
    end

    test "print_stats/1 returns :ok", %{mtp: mtp} do
      assert MTP.print_stats(mtp) == :ok
    end

    test "a halted stream stops generation instead of running to max_tokens", %{mtp: mtp} do
      taken = mtp |> MTP.stream("Write an endless story:", max_tokens: 400) |> Enum.take(3)
      assert length(taken) == 3

      # The MTP value is still usable afterwards.
      assert {:ok, _} = MTP.generate(mtp, "2 + 2 =", max_tokens: 4, temp: 0.0)
    end
  end

  # B-2, and it needs no MTP model: the setup failure happens in
  # `start_mtp_stream/6` before anything reaches `generate_mtp_tokens/9`, so a
  # hand-built `%MTP{}` over an ordinary context reproduces it exactly. One gate
  # tag for this describe (`:smoke`, the generation model) — it must not carry
  # `:mtp` as well, because `--include` beats `--exclude` and a second gate tag
  # would drag it into runs that have no model for it.
  describe "a stream that fails during setup" do
    @describetag :smoke
    @describetag timeout: 120_000

    setup do
      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:gen), n_gpu_layers: 0)
      {:ok, ctx} = LlamaCppEx.Context.create(model, n_ctx: 256)

      # spec_ref is never dereferenced: Sampler.create/2 fails on the grammar
      # first, and that is the whole point of the test.
      %{mtp: %MTP{main_ctx: ctx, mtp_ctx: ctx, spec_ref: make_ref(), n_draft: 2}}
    end

    # `start_mtp_stream/6` reports failure by *adding* a `:setup_error` key rather
    # than flipping the `:done?` discriminant the clauses dispatch on, so
    # `%{state | done?: true}` still matched the `%{setup_error: _}` clause and the
    # stream emitted the same error element forever. Driven through a Task with a
    # timeout so a regression fails this test instead of wedging the suite with a
    # growing heap — which is what `generate/3`'s `Enum.to_list/1` did.
    test "emits exactly one error element and halts", %{mtp: mtp} do
      task =
        Task.async(fn -> mtp |> MTP.stream_events("hi", grammar: "{{{") |> Enum.to_list() end)

      assert {:ok, elements} = Task.yield(task, 2_000) || Task.shutdown(task, :brutal_kill),
             "stream_events/3 did not terminate — the halt clause is being shadowed again"

      assert elements == [{:error, :invalid_grammar}]
    end

    test "generate/3 returns the error instead of hanging", %{mtp: mtp} do
      task = Task.async(fn -> MTP.generate(mtp, "hi", grammar: "{{{") end)

      assert {:ok, result} = Task.yield(task, 2_000) || Task.shutdown(task, :brutal_kill),
             "generate/3 did not terminate — it drives stream_events/3 with Enum.to_list/1"

      assert result == {:error, :invalid_grammar}
    end

    test "stream/3 yields nothing rather than looping", %{mtp: mtp} do
      task = Task.async(fn -> mtp |> MTP.stream("hi", grammar: "{{{") |> Enum.to_list() end)

      assert {:ok, []} = Task.yield(task, 2_000) || Task.shutdown(task, :brutal_kill)
    end
  end

  # The event vocabulary stream_events/3 documents.
  defp event_kind({:token, id, text}) when is_integer(id) and is_binary(text), do: :token
  defp event_kind({:stats, snapshot}) when is_map(snapshot), do: :stats
  defp event_kind({:done, snapshot}) when is_map(snapshot), do: :terminal
  defp event_kind({:eog, nil}), do: :terminal
  defp event_kind({:error, _reason}), do: :terminal
  defp event_kind(_), do: :undocumented
end
