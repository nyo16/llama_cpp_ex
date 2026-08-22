defmodule LlamaCppEx.MTPModelTest do
  # The MTP tests that need a real MTP-enabled GGUF. They live in their own
  # module rather than a `describe` inside `mtp_test.exs` because the model has to
  # be loaded in `setup_all`, which ExUnit only allows at module level — and
  # putting it there in `mtp_test.exs` would force a model load on the pure
  # argument-validation tests, which are meant to run in the default suite.
  #
  # Loading once per module is not a tidiness preference. An MTP GGUF is large
  # and each `MTP.init/2` reserves two contexts whose Metal compute buffers come
  # to ~1.5 GB combined. A per-test load kept several copies alive at once — NIF
  # resources are freed on GC, not on scope exit — and the third test aborted the
  # VM through `GGML_ASSERT(buffer_id >= 0)` in ggml-alloc once a graph
  # allocation could no longer be satisfied. Each test passed in isolation; only
  # the accumulation failed.
  #
  #   GGML_METAL_NO_RESIDENCY=1 LLAMA_BACKEND=auto \
  #   LLAMA_SMOKE_MTP_MODEL=/path/to/mtp-model.gguf mix test --include mtp
  #
  # async: false — one GPU, and the session holds the whole model.
  use ExUnit.Case, async: false

  alias LlamaCppEx.MTP

  @moduletag :mtp
  @moduletag timeout: 300_000

  # One model *and* one session for the whole module. Sharing the session is not
  # just a second economy: with the model loaded once, the two contexts each
  # `MTP.init/2` reserves became the thing that accumulated, and a per-test
  # session still aborted the VM after enough of them. `MTP` documents a session
  # as reusable across calls — it clears both KV caches on entry — and the one
  # test that reads cumulative counters asserts on deltas rather than absolutes,
  # so nothing here needs a private session.
  setup_all do
    :ok = LlamaCppEx.init()

    {:ok, model} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp),
        n_gpu_layers: -1,
        load_mtp: true
      )

    {:ok, mtp} = MTP.init(model, n_ctx: 2048, n_draft: 3)

    %{model: model, session: mtp}
  end

  test "init/2 builds both contexts and defaults n_draft to 3", %{model: model} do
    assert {:ok, mtp} = MTP.init(model, n_ctx: 512)
    assert mtp.n_draft == 3
    assert %LlamaCppEx.Context{} = mtp.main_ctx
    assert %LlamaCppEx.Context{} = mtp.mtp_ctx
    assert is_reference(mtp.spec_ref)
  end

  test "generate/3 returns non-empty deterministic text", %{session: mtp} do
    opts = [max_tokens: 16, temp: 0.0, seed: 7]

    assert {:ok, text} = MTP.generate(mtp, "The capital of France is", opts)
    assert byte_size(text) > 0
    assert String.valid?(text)

    # Speculative decoding must not change *what* is generated, only how fast.
    assert {:ok, ^text} = MTP.generate(mtp, "The capital of France is", opts)
  end

  test "stream/3 yields the same text generate/3 returns", %{session: mtp} do
    opts = [max_tokens: 16, temp: 0.0, seed: 7]

    streamed = mtp |> MTP.stream("Count to five:", opts) |> Enum.join()
    assert {:ok, ^streamed} = MTP.generate(mtp, "Count to five:", opts)
  end

  # Regression: the verify loop emits up to `1 + n_draft` tokens per iteration,
  # and used to check the caller's budget only on iteration entry — so a request
  # for 16 tokens came back with 16, 17 or 18 of them depending on how many
  # drafts the target accepted in the final iteration. Acceptance varies between
  # runs on a reused session, so this was observable as `stream/3` and
  # `generate/3` returning different-length prefixes of the same greedy
  # continuation. `max_tokens` is a bound, not a hint.
  test "max_tokens is an exact upper bound regardless of draft acceptance", %{
    model: model,
    session: mtp
  } do
    for max_tokens <- [1, 4, 16] do
      assert {:ok, text} =
               MTP.generate(mtp, "Count to twenty:", max_tokens: max_tokens, temp: 0.0)

      assert {:ok, tokens} = LlamaCppEx.Tokenizer.encode(model, text, add_special: false)

      assert length(tokens) <= max_tokens,
             "max_tokens: #{max_tokens} produced #{length(tokens)} tokens: #{inspect(text)}"
    end
  end

  test "stream_events/3 emits only documented events, ending with a terminal one", %{session: mtp} do
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

  test "emit_stats_every yields stats events that stream/3 filters out", %{session: mtp} do
    opts = [max_tokens: 16, temp: 0.0, emit_stats_every: 2]

    events = mtp |> MTP.stream_events("Tell me a story:", opts) |> Enum.to_list()
    assert Enum.any?(events, &match?({:stats, _}, &1))

    # stream/3 is the text-only view of the same events.
    pieces = mtp |> MTP.stream("Tell me a story:", opts) |> Enum.to_list()
    assert Enum.all?(pieces, &is_binary/1)
  end

  test "stats/1 counts drafts and accepts after generation", %{session: mtp} do
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

  test "print_stats/1 returns :ok", %{session: mtp} do
    assert MTP.print_stats(mtp) == :ok
  end

  # The event vocabulary stream_events/3 documents.
  defp event_kind({:token, id, text}) when is_integer(id) and is_binary(text), do: :token
  defp event_kind({:stats, snapshot}) when is_map(snapshot), do: :stats
  defp event_kind({:done, snapshot}) when is_map(snapshot), do: :terminal
  defp event_kind({:eog, nil}), do: :terminal
  defp event_kind({:error, _reason}), do: :terminal
  defp event_kind(_), do: :undocumented
end

defmodule LlamaCppEx.MTPCancelTest do
  # KNOWN BUG, kept as an executable record rather than deleted.
  #
  # At b10435 this test did not fail — it took the VM down, with
  # `GGML_ASSERT(offset + size <= ggml_nbytes(tensor))` or a plain SIGSEGV.
  # At b10582 it no longer aborts: it fails, racily, with
  # `{:error, "prompt decode failed: code=-1"}` or `"verify decode failed:
  # code=-1"` from the `generate/3` below — and sometimes passes. Measured on
  # M1 Max / Metal with Qwen3.6-35B-A3B-MTP, four runs each: b10435 aborted 4/4
  # (one exit 134, three exit 139), b10582 aborted 0/4 (3 failed, 1 passed).
  # The race is unchanged and unfixed; only its consequence moved from "kills
  # the BEAM" to "returns an error", so the two decode paths now refuse the
  # half-released context instead of writing through it.
  #
  # It therefore still lives in its own module carrying only `:mtp_cancel`, and
  # deliberately *not* `:mtp`: `--include` beats `--exclude` in ExUnit, so a
  # second gate tag would drag a flaky test back into `--include mtp` runs.
  # Run it on purpose:
  #
  #   GGML_METAL_NO_RESIDENCY=1 LLAMA_BACKEND=auto \
  #   LLAMA_SMOKE_MTP_MODEL=... mix test --include mtp_cancel
  #
  # Cancellation is fire-and-forget: `request_cancel` sets an atomic flag and the
  # MTP loop then "stops quietly" (`llama_nif.cpp:2009`) without emitting any
  # terminal event. `Generator.stop/1` cancels, unlinks and drains the mailbox,
  # but it has no completion signal to wait on, so it returns while the dirty
  # scheduler may still be inside `llama_decode`. For `LlamaCppEx.stream/3` that
  # is harmless — each call owns a context that dies with it. An `%MTP{}` session
  # is not: it holds two long-lived contexts and is documented as reusable across
  # calls, so the `generate/3` below can start decoding on the very contexts the
  # cancelled loop has not released. Two writers, one KV cache.
  #
  # The fix is to acknowledge cancellation — have the loop emit a terminal event
  # when it observes the flag and have `Generator.stop/1` await it — which changes
  # the NIF and the cancellation protocol that non-MTP streaming also uses. That
  # is why it is not bundled into a submodule bump.
  use ExUnit.Case, async: false

  alias LlamaCppEx.MTP

  @moduletag :mtp_cancel
  @moduletag timeout: 300_000

  setup_all do
    :ok = LlamaCppEx.init()

    {:ok, model} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp),
        n_gpu_layers: -1,
        load_mtp: true
      )

    {:ok, mtp} = MTP.init(model, n_ctx: 2048, n_draft: 3)
    %{session: mtp}
  end

  test "a halted stream stops generation instead of running to max_tokens", %{session: mtp} do
    taken = mtp |> MTP.stream("Write an endless story:", max_tokens: 400) |> Enum.take(3)
    assert length(taken) == 3

    # The MTP value is still usable afterwards.
    assert {:ok, _} = MTP.generate(mtp, "2 + 2 =", max_tokens: 4, temp: 0.0)
  end
end

defmodule LlamaCppEx.MTPSidecarTest do
  # The target/draft split, which is how Qwen 3.8 ships MTP: the target GGUF
  # carries zero nextn layers and a separate `mtp-*.gguf` carries the head. This
  # needs two files rather than one, so it gates on its own tag and its own env
  # var — `--include mtp` has only the single-file model:
  #
  #   GGML_METAL_NO_RESIDENCY=1 LLAMA_BACKEND=auto \
  #   LLAMA_SMOKE_MTP_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  #   LLAMA_SMOKE_MTP_DRAFT_MODEL=/path/to/mtp-Qwen3.8-27B-Q4_0.gguf \
  #     mix test --include mtp_sidecar
  #
  # async: false, and one session for the module, for the same reason
  # MTPModelTest says: one GPU, and each session reserves two contexts.
  use ExUnit.Case, async: false

  alias LlamaCppEx.MTP

  @moduletag :mtp_sidecar
  @moduletag timeout: 300_000

  setup_all do
    :ok = LlamaCppEx.init()

    {:ok, target} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp),
        n_gpu_layers: -1,
        load_mtp: true
      )

    {:ok, draft} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp_draft),
        n_gpu_layers: -1,
        load_mtp: true
      )

    {:ok, session} = MTP.init(target, draft_model: draft, n_ctx: 2048, n_draft: 1)

    %{target: target, draft: draft, session: session}
  end

  test "the sidecar carries the head and the target does not", %{target: t, draft: d} do
    # The premise of the whole pairing. If a future conversion moves the head
    # into the target this test is the thing that notices.
    assert LlamaCppEx.Model.n_layer_nextn(d) > 0
    assert LlamaCppEx.Model.n_layer_nextn(t) == 0

    # Upstream compares these two with a GGML_ASSERT, which aborts the VM rather
    # than failing a call, so the pair must agree before init/2 builds anything.
    assert LlamaCppEx.Model.n_embd_out(d) == LlamaCppEx.Model.n_embd_out(t)
  end

  test "init/2 builds a session from the pair", %{session: session} do
    assert %LlamaCppEx.Context{} = session.main_ctx
    assert %LlamaCppEx.Context{} = session.mtp_ctx
    assert is_reference(session.spec_ref)
  end

  test "the same target is refused without the sidecar", %{target: t} do
    # Not a redundant restatement of the unit test: there the nextn count comes
    # from a hand-built struct, here it is read off the real file.
    assert {:error, message} = MTP.init(t, n_ctx: 512)
    assert message =~ "no MTP head"
    assert message =~ "draft_model"
  end

  test "generate/3 produces text and the head actually drafts", %{session: session} do
    before = MTP.stats(session)

    assert {:ok, text} = MTP.generate(session, "2 + 2 =", max_tokens: 16, temp: 0.0)
    assert is_binary(text) and text != ""

    now = MTP.stats(session)

    # Deltas, not absolutes: the session is shared across this module's tests.
    assert now.drafts_generated > before.drafts_generated,
           "the sidecar head proposed no drafts at all"

    assert now.tokens_emitted > before.tokens_emitted
  end

  # A hybrid target (Qwen 3.8: SSM layers beside attention ones) cannot roll back
  # part of a sequence natively, so the loop snapshots the recurrent state every
  # iteration. That cost is the difference between speculation paying off and not,
  # so it gets its own bucket rather than hiding inside :other.
  test "timing_us reports a ckpt bucket", %{session: session} do
    assert {:ok, _} = MTP.generate(session, "Count to ten:", max_tokens: 24, temp: 0.0)

    timing = MTP.stats(session).timing_us

    for key <- [:draft, :verify, :sample, :ckpt, :other, :total] do
      assert Map.has_key?(timing, key), "timing_us is missing #{inspect(key)}"
    end

    assert timing.ckpt > 0,
           "a hybrid target should have paid for at least one recurrent-state snapshot"

    # The named buckets are carved out of total, never billed twice on top of it.
    assert timing.draft + timing.verify + timing.sample + timing.ckpt <= timing.total
  end

  test "greedy output matches plain greedy decode on the target", %{
    target: target,
    session: session
  } do
    # Speculation is exactness-preserving in principle. In practice ggml selects
    # a different matmul kernel by batch row count — ggml-metal-ops.cpp picks the
    # mul_mv_ext path for Q4_K only at ne11 >= 4, and its r1ptg by ne11 — and the
    # MTP verify batch is 1 + n_draft rows wide, so at a position where the top
    # two logits are within rounding error the argmax can differ from a 1-row
    # plain decode. That is not a bug and it is demonstrable *without* MTP: on
    # Qwen 3.8 / M1 Max, feeding one fixed prefix to plain greedy decode gives
    # " computational" at 1-3 rows and " latency" at 4+. So compare a short
    # continuation, where no such near-tie has come up.
    prompt = "The capital of France is"
    opts = [max_tokens: 8, temp: 0.0]

    assert {:ok, spec} = MTP.generate(session, prompt, opts)
    assert {:ok, plain} = LlamaCppEx.generate(target, prompt, opts ++ [n_ctx: 2048])

    assert spec == plain
  end
end
