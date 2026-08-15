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

  describe "init/2 requires a model loaded with load_mtp: true" do
    # Regression test for a silent break: upstream #26296 made the MTP head's
    # tensors opt-in at load time via a flag defaulting to false. Nothing on the
    # way in objects — both contexts build and common_speculative_init returns
    # ok — and the first draft then fails with "verify decode failed: code=-1",
    # far from the cause. The guard runs before any context is created, so a nil
    # ref never reaches the NIF.
    test "refuses a model loaded without the flag, naming the remedy" do
      assert {:error, message} = MTP.init(%LlamaCppEx.Model{ref: nil}, n_draft: 3)
      assert message =~ "load_mtp: true"
      assert message =~ "reload it"
    end

    test "the flag is recorded on the struct, not re-derived" do
      # Model.load/2 is what sets this; a hand-built struct defaults to false so
      # that the guard fails closed rather than open.
      refute %LlamaCppEx.Model{ref: nil}.load_mtp
    end

    test "n_draft is still validated first, so its error is not masked" do
      assert MTP.init(%LlamaCppEx.Model{ref: nil, load_mtp: true}, n_draft: 0) ==
               {:error, ":n_draft must be a positive integer"}

      assert MTP.init(%LlamaCppEx.Model{ref: nil, load_mtp: false}, n_draft: 0) ==
               {:error, ":n_draft must be a positive integer"}
    end
  end

  # Qwen 3.8 ships the MTP head as a sidecar GGUF: the target carries zero nextn
  # layers and the head file carries nothing else, so the pair only works if the
  # draft context can be built from a *different* model than the target. These
  # guards all run before any context is created, so a nil ref never reaches the
  # NIF — the ones that must read model metadata (nextn count, hidden width) need
  # a real file and live in MTPModelTest.
  describe "init/2 with a separate :draft_model" do
    @unloaded %LlamaCppEx.Model{ref: nil}
    @loaded_mtp %LlamaCppEx.Model{ref: nil, load_mtp: true}

    test "refuses a sidecar loaded without load_mtp: true, naming the remedy" do
      assert {:error, message} = MTP.init(@loaded_mtp, draft_model: @unloaded)
      assert message =~ "draft_model was loaded without load_mtp: true"
      assert message =~ "reload it"
    end

    test "the sidecar's flag is what matters, not the target's" do
      # The target legitimately has no head of its own in this shape, so its own
      # load_mtp is not what gates the session — a target without the flag and a
      # sidecar with it must get past the flag check rather than be refused for
      # the target's sake. Reaching the metadata read (which a nil ref cannot
      # survive) is the proof it got that far.
      assert catch_error(MTP.init(@unloaded, draft_model: @loaded_mtp))
    end

    test "rejects a :draft_model that is not a Model" do
      for bad <- ["mtp.gguf", :mtp, 42, %{ref: nil}] do
        assert {:error, message} = MTP.init(@loaded_mtp, draft_model: bad)
        assert message =~ ":draft_model must be a LlamaCppEx.Model"
      end
    end

    test "nil :draft_model is the in-target-head path, not a bad argument" do
      # Explicitly passing nil must behave exactly like omitting the option.
      assert MTP.init(@unloaded, draft_model: nil) == MTP.init(@unloaded)
    end

    test "n_draft is still validated first, so its error is not masked" do
      assert MTP.init(@loaded_mtp, draft_model: @unloaded, n_draft: 0) ==
               {:error, ":n_draft must be a positive integer"}
    end
  end

  # A checkpoint with no MTP head is a different failure from a model loaded
  # without the flag, and no flag recovers it — only a different file. There are
  # two such files, and the message has to name both: an MTP-preserving
  # conversion of the whole model (unsloth's Qwen3.6-35B-A3B-UD-Q4_K_XL has zero
  # nextn layers while their separate -MTP build of it has them), or the
  # publisher's head-only sidecar passed as `:draft_model` (Qwen 3.8 ships only
  # that shape). This is the case a user actually lands on first.
  #
  # One gate tag (`:smoke`), never `:mtp` as well: the generation model is an
  # ordinary checkpoint, which is exactly what makes it the right fixture here.
  describe "init/2 on a checkpoint with no MTP head" do
    @describetag :smoke

    setup do
      path = LlamaCppEx.TestModels.path!(:gen)
      {:ok, model} = LlamaCppEx.load_model(path, n_gpu_layers: 0, load_mtp: true)
      %{model: model}
    end

    test "reports zero nextn layers rather than guessing", %{model: model} do
      assert LlamaCppEx.NIF.model_n_layer_nextn(model.ref) == 0
    end

    test "refuses with the reason, not 'failed to create context'", %{model: model} do
      assert {:error, message} = MTP.init(model, n_draft: 3)
      assert message =~ "no MTP head"
      # Naming the remedy is the point of the guard, so assert on both routes out
      # rather than on any one phrasing of the diagnosis.
      assert message =~ "MTP-preserving conversion"
      assert message =~ "draft_model"
      # The bare context error is what this guard exists to replace.
      refute message =~ "failed to create context"
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
end
