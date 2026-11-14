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
  #   LLAMA_BACKEND=auto \
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

  # Regression: cancelling a stream is fire-and-forget, so the consumer gets
  # control back while the cancelled loop can still be inside a decode on the
  # session's two contexts. A call made right then decoded into the same KV
  # caches: from b10435 to 9daa0aa22 this aborted the VM in roughly half the
  # runs (last as `GGML_ASSERT(logits != nullptr)` in the MTP drafter) or
  # failed with "verify decode failed". `generate_mtp_tokens` now holds the
  # session for its whole run, so the next call waits for the cancelled one.
  # Three rounds, and the greedy text compared, because a lost race need not
  # crash: it can only corrupt the cache.
  test "a session is reusable right after a halted stream", %{session: mtp} do
    opts = [max_tokens: 8, temp: 0.0, seed: 7]
    assert {:ok, expected} = MTP.generate(mtp, "2 + 2 =", opts)

    for _ <- 1..3 do
      taken = mtp |> MTP.stream("Write an endless story:", max_tokens: 400) |> Enum.take(3)
      assert length(taken) == 3
      assert {:ok, ^expected} = MTP.generate(mtp, "2 + 2 =", opts)
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

defmodule LlamaCppEx.MTPSidecarTest do
  # The target/draft split, which is how Qwen 3.8 ships MTP: the target GGUF
  # carries zero nextn layers and a separate `mtp-*.gguf` carries the head. This
  # needs two files rather than one, so it gates on its own tag and its own env
  # var — `--include mtp` has only the single-file model:
  #
  #   LLAMA_BACKEND=auto \
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
  # part of a sequence from its KV cache alone. init/2 gives it one recurrent
  # snapshot per draft position, as upstream's server does, so a rejected draft
  # rolls back natively. Without them the loop saves the recurrent state every
  # iteration and re-decodes the accepted prefix on every partial accept, which
  # measured 21 tok/s against 27 at n_draft 2 on Qwen3.8-27B NVFP4 — the :ckpt
  # bucket is where that path would show.
  test "a hybrid target rolls back on recurrent snapshots, not checkpoints", %{
    session: session
  } do
    assert LlamaCppEx.Context.n_rs_seq(session.main_ctx) == session.n_draft
    assert LlamaCppEx.Context.n_rs_seq(session.mtp_ctx) == 0

    assert {:ok, _} = MTP.generate(session, "Count to ten:", max_tokens: 24, temp: 0.0)

    timing = MTP.stats(session).timing_us

    for key <- [:draft, :verify, :sample, :ckpt, :other, :total] do
      assert Map.has_key?(timing, key), "timing_us is missing #{inspect(key)}"
    end

    assert timing.ckpt == 0, "the loop fell back to recurrent-state checkpoints"

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

defmodule LlamaCppEx.DFlashTest do
  # DFlash block-diffusion drafting through the same session API, type: :dflash.
  # The drafter is a separate `dflash`-architecture GGUF trained against one
  # target: it reads that target's hidden states at fixed layer indices.
  #
  #   LLAMA_SMOKE_DFLASH_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  #   LLAMA_SMOKE_DFLASH_DRAFT_MODEL=/path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  #     mix test --include dflash
  #
  # async: false and one session for the module, as for the MTP modules above.
  use ExUnit.Case, async: false

  alias LlamaCppEx.MTP

  @moduletag :dflash
  @moduletag timeout: 300_000

  setup_all do
    :ok = LlamaCppEx.init()

    {:ok, target} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:dflash), n_gpu_layers: -1)

    {:ok, draft} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:dflash_draft), n_gpu_layers: -1)

    # Ask for more than any DFlash block holds, to see it clamped.
    {:ok, session} = MTP.init(target, type: :dflash, draft_model: draft, n_ctx: 2048, n_draft: 64)

    %{target: target, draft: draft, session: session}
  end

  test "n_draft is clamped to the drafter's block size", %{session: session} do
    # The loop sizes its verify batch from this; asking upstream for more than
    # the block holds would draft nothing past it.
    assert session.n_draft < 64
    assert session.n_draft == MTP.stats(session).n_draft
  end

  test "generate/3 drafts and the target accepts some of it", %{session: session} do
    before = MTP.stats(session)

    assert {:ok, text} =
             MTP.generate(session, "List the days of the week, comma separated:",
               max_tokens: 32,
               temp: 0.0
             )

    assert text =~ "Monday"

    now = MTP.stats(session)
    assert now.drafts_generated > before.drafts_generated

    assert now.drafts_accepted > before.drafts_accepted,
           "no draft accepted: the drafter is not seeing the target's hidden states"
  end

  test "greedy output matches plain greedy decode on the target", %{
    target: target,
    session: session
  } do
    # Short, for the near-tie reason MTPSidecarTest gives.
    prompt = "The capital of France is"
    opts = [max_tokens: 8, temp: 0.0]

    assert {:ok, spec} = MTP.generate(session, prompt, opts)
    assert {:ok, plain} = LlamaCppEx.generate(target, prompt, opts ++ [n_ctx: 2048])
    assert spec == plain
  end

  test "a model that is not a DFlash drafter is refused, not aborted", %{target: target} do
    # Upstream's constructor GGML_ASSERTs on a drafter with no target layers.
    assert {:error, message} = MTP.init(target, type: :dflash, draft_model: target, n_ctx: 512)
    assert message =~ "not a DFlash/DSpark drafter"
  end

  test "the drafter cannot stand in for an MTP head", %{target: target, draft: draft} do
    assert {:error, message} = MTP.init(target, draft_model: draft, n_ctx: 512)
    assert message =~ "load_mtp: true"
  end
end

defmodule LlamaCppEx.MTPE4BSidecarTest do
  # Gemma4 E4B's target/draft split: the target GGUF is `gemma4` with zero nextn
  # layers, and a separate `mtp-gemma-4-*.gguf` is a `gemma4-assistant` sidecar
  # (the sidecar itself has nextn layers). Same `:mtp_sidecar` tag, own env pair.
  # Unset E4B vars skip this module; MTPSidecarTest still needs the Qwen pair.
  # Add these two vars to the Qwen `--include mtp_sidecar` command:
  #
  #   LLAMA_SMOKE_MTP_E4B_MODEL=/path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  #   LLAMA_SMOKE_MTP_E4B_DRAFT_MODEL=/path/to/mtp-gemma-4-E4B-it-Q8_0.gguf
  #
  # async: false, and one session for the module, for the same reason
  # MTPModelTest says: one GPU, and each session reserves two contexts.
  use ExUnit.Case, async: false

  alias LlamaCppEx.MTP

  @moduletag :mtp_sidecar
  @moduletag timeout: 300_000

  if System.get_env("LLAMA_SMOKE_MTP_E4B_MODEL") in [nil, ""] or
       System.get_env("LLAMA_SMOKE_MTP_E4B_DRAFT_MODEL") in [nil, ""] do
    @moduletag skip: "set LLAMA_SMOKE_MTP_E4B_MODEL and LLAMA_SMOKE_MTP_E4B_DRAFT_MODEL"
  end

  setup_all do
    :ok = LlamaCppEx.init()

    {:ok, target} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp_e4b),
        n_gpu_layers: -1,
        load_mtp: true
      )

    {:ok, draft} =
      LlamaCppEx.load_model(LlamaCppEx.TestModels.path!(:mtp_e4b_draft),
        n_gpu_layers: -1,
        load_mtp: true
      )

    {:ok, session} = MTP.init(target, draft_model: draft, n_ctx: 2048, n_draft: 3)

    %{target: target, draft: draft, session: session}
  end

  test "the sidecar carries the head and the target does not", %{target: t, draft: d} do
    assert LlamaCppEx.Model.n_layer_nextn(d) > 0
    assert LlamaCppEx.Model.n_layer_nextn(t) == 0

    assert LlamaCppEx.Model.n_embd_out(d) == LlamaCppEx.Model.n_embd_out(t)
  end

  test "gemma4-assistant has a narrow draft width and a wide output width", %{draft: d} do
    # gemma4-assistant: n_embd is the assistant width, n_embd_out is the target.
    assert LlamaCppEx.Model.n_embd(d) != LlamaCppEx.Model.n_embd_out(d)
  end

  test "init/2 builds a session from the pair", %{session: session} do
    assert %LlamaCppEx.Context{} = session.main_ctx
    assert %LlamaCppEx.Context{} = session.mtp_ctx
    assert is_reference(session.spec_ref)

    # The Elixir peer field is the same %Context{} as main_ctx.
    assert %LlamaCppEx.Context{} = session.mtp_ctx.ctx_other
    assert session.mtp_ctx.ctx_other.ref == session.main_ctx.ref
    assert is_nil(session.main_ctx.ctx_other)
  end

  test "the same target is refused without the sidecar", %{target: t} do
    assert {:error, message} = MTP.init(t, n_ctx: 512)
    assert message =~ "no MTP head"
    assert message =~ "draft_model"
  end

  test "generate/3 produces text and the head actually drafts", %{
    target: target,
    session: session
  } do
    before = MTP.stats(session)

    {:ok, prompt} =
      LlamaCppEx.Chat.apply_template(target, [%{role: "user", content: "2 + 2 ="}],
        enable_thinking: false
      )

    assert {:ok, text} = MTP.generate(session, prompt, max_tokens: 16, temp: 0.0)
    assert is_binary(text) and text != ""

    now = MTP.stats(session)

    assert now.drafts_generated > before.drafts_generated,
           "the sidecar head proposed no drafts at all"

    assert now.tokens_emitted > before.tokens_emitted
  end

  # E4B is dense, not hybrid — ckpt stays 0. The Qwen sidecar test pins that a
  # hybrid target stays off the checkpoint path too.
  test "timing_us reports the expected buckets", %{target: target, session: session} do
    {:ok, prompt} =
      LlamaCppEx.Chat.apply_template(target, [%{role: "user", content: "Count to ten:"}],
        enable_thinking: false
      )

    assert {:ok, _} = MTP.generate(session, prompt, max_tokens: 24, temp: 0.0)

    timing = MTP.stats(session).timing_us

    for key <- [:draft, :verify, :sample, :ckpt, :other, :total] do
      assert Map.has_key?(timing, key), "timing_us is missing #{inspect(key)}"
    end

    assert timing.ckpt == 0

    assert timing.draft + timing.verify + timing.sample + timing.ckpt <= timing.total
  end
end
