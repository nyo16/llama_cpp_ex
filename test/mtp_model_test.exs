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
  # This test does not fail — it takes the VM down, with
  # `GGML_ASSERT(offset + size <= ggml_nbytes(tensor))` or a plain SIGSEGV. So it
  # lives in its own module carrying only `:mtp_cancel`, and deliberately *not*
  # `:mtp`: `--include` beats `--exclude` in ExUnit, so a second gate tag would
  # drag it back into `--include mtp` runs and abort them. Run it on purpose:
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
