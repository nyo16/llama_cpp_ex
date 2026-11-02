defmodule LlamaCppEx.MTP do
  @moduledoc """
  Multi-Token Prediction (MTP) speculative decoding.

  Drives a target/draft speculative loop where the draft model is an MTP head —
  either embedded in the target GGUF, or shipped beside it as a sidecar file. On
  Qwen 3.6 with `n_draft: 3` this typically yields ~2x token-generation
  throughput at ~75% draft acceptance.

  ## Usage

      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model("Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
                                            n_gpu_layers: 999, load_mtp: true)

      {:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 8192)

      mtp
      |> LlamaCppEx.MTP.stream("Write a haiku about the sea:", max_tokens: 200)
      |> Stream.each(&IO.write/1)
      |> Stream.run()

      stats = LlamaCppEx.MTP.stats(mtp)
      IO.puts("acceptance: \#{Float.round(stats.acceptance_rate * 100, 1)}%")

  ## Where the MTP head lives

  The head is a set of `*.nextn_predict_layers` and comes in two shapes, and
  either way the file carrying it must be loaded with `load_mtp: true`. Upstream
  defaults that flag to `false` so non-speculative callers do not pay for the
  head's tensors, and the layers cannot be attached afterwards, so `init/2`
  refuses a model loaded without it.

  **In the target GGUF** (e.g. `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`) — pass just
  the model, as above.

  **In a sidecar GGUF** — pass it as `:draft_model`. This is how Qwen 3.8 ships:
  `Qwen3.8-27B-Q4_K_M.gguf` carries no head at all (`n_layer_nextn == 0`) and
  `mtp-Qwen3.8-27B-Q4_0.gguf` carries nothing else. It is the binding's
  equivalent of upstream's `-hf <target> -hfd <draft> --spec-type draft-mtp`.

      {:ok, target} = LlamaCppEx.load_model("Qwen3.8-27B-Q4_K_M.gguf",
                                            n_gpu_layers: 999, load_mtp: true)
      {:ok, head}   = LlamaCppEx.load_model("mtp-Qwen3.8-27B-Q4_0.gguf",
                                            n_gpu_layers: 999, load_mtp: true)

      {:ok, mtp} = LlamaCppEx.MTP.init(target, draft_model: head, n_draft: 1)

  **Gemma4 E4B sidecar** — same `:draft_model` API: a `gemma4` target plus a
  `gemma4-assistant` head. `init/2` always passes the target as `:ctx_other`
  on the draft. `gemma4-assistant` keeps that pointer (shared KV and the
  target's token embeddings); Qwen's constructor leaves `cparams.ctx_other`
  as `nullptr`.

  > #### Hybrid models roll back on recurrent snapshots {: .info}
  >
  > A model that mixes recurrent (SSM) layers with attention ones — Qwen 3.8 is
  > 48 SSM layers to 16 attention layers — cannot roll back part of a sequence
  > from its KV cache alone. `init/2` therefore creates the target context with
  > one recurrent-state snapshot per draft position (`n_rs_seq: n_draft`), as
  > upstream's server does, and a rejected draft is rolled back natively. Before
  > that, every iteration saved the whole recurrent state (over 100 MiB at Qwen
  > 3.8's sizes) and every partial accept restored it and re-decoded the
  > accepted prefix: on GB10 that held Qwen3.8-27B NVFP4 to 21 tok/s at
  > `n_draft: 2` where the snapshot path does 27, and on an M1 Max it made MTP a
  > net slowdown. `stats/1`'s `timing_us.ckpt` is that save/restore, and stays
  > zero unless `n_draft` exceeds the snapshots.

  Upstream currently requires `n_parallel = 1` for MTP. This module reflects
  that — a single MTP session decodes one sequence at a time. Reuse the same
  `%MTP{}` value across calls to `stream/3` / `generate/3` to avoid rebuilding
  the contexts; KV caches are cleared on each call.

  > #### Do not reuse a session straight after abandoning a stream {: .warning}
  >
  > Cancellation is asynchronous and unacknowledged: abandoning a `stream/3`
  > early (`Enum.take/2`, `break`, an exception) sets a flag that the draft loop
  > notices and then exits without reporting that it has done so. A session's two
  > contexts are long-lived and shared by every call on it, so starting the next
  > `generate/3` or `stream/3` immediately can put a second writer on a KV cache
  > the cancelled loop has not finished with — which aborts the VM rather than
  > returning an error. Let an abandoned stream run to a terminal event, or build
  > a fresh session with `init/2`, before using the session again.

  MTP is the only speculative type this binding exposes. Upstream llama.cpp
  also implements EAGLE-3, DFlash (block-diffusion drafting), n-gram
  self-speculation and combinations of them behind the same
  `common_speculative` API; the NIF pins the MTP type, so `--spec-default`-style
  stacking of n-gram speculation on top of MTP is not reachable from here. See
  the "Speculative decoding" section of the README for the current status of
  DFlash on Apple Silicon.
  """

  alias LlamaCppEx.{Context, Model, Sampler, Tokenizer}

  @enforce_keys [:main_ctx, :mtp_ctx, :spec_ref, :n_draft]
  defstruct [:main_ctx, :mtp_ctx, :spec_ref, :n_draft]

  @type t :: %__MODULE__{
          main_ctx: Context.t(),
          mtp_ctx: Context.t(),
          spec_ref: reference(),
          n_draft: pos_integer()
        }

  # Context options forwarded to both the target and draft contexts. The list is
  # owned by Context (tuning_option_keys/0); MTP additionally lets the caller set
  # :n_ctx, which is structural everywhere else because each caller normally
  # computes it.
  defp forwardable_context_opts(opts) do
    Keyword.take(opts, [:n_ctx | Context.tuning_option_keys()])
  end

  @doc """
  Initializes a speculative session: builds the target context, the draft
  context, and the underlying `common_speculative` state.

  ## Options

    * `:type` - The drafter. Defaults to `:mtp`.
      * `:mtp` - a Multi-Token Prediction head, inside the target GGUF or in a
        sidecar passed as `:draft_model`. The draft context is built with
        `ctx_type: :mtp`.
      * `:dflash` - a DFlash / DFlash 2 block-diffusion drafter (GGUF
        architecture `dflash`), passed as `:draft_model`. It drafts a whole
        block in one pass from the target's hidden states, and is the fastest
        option measured on GB10: 3.3× plain decode on Qwen3.8-27B NVFP4 against
        2.1× for its MTP head (see `docs/dgx-spark.md`).
      * `:dspark` - a DSpark drafter (DFlash plus a Markov head), passed as
        `:draft_model`.
    * `:draft_model` - A separate `LlamaCppEx.Model` holding the drafter.
      Required for `:dflash` and `:dspark`, and loaded on every device the
      target uses (`LlamaCppEx.Model.devices/1`): a DFlash drafter computes
      with the target's embeddings and LM head. For a target split across an
      RPC worker, load the drafter with the same `:devices` and a
      `:tensor_split` that keeps its layers local. For `:mtp` it is the sidecar head
      for checkpoints that ship one rather than embedding it (Qwen 3.8:
      `Qwen3.8-27B-Q4_K_M.gguf` plus `mtp-Qwen3.8-27B-Q4_0.gguf`; Gemma4 E4B:
      a `gemma4` target plus a `gemma4-assistant` head), loaded with
      `load_mtp: true`. Defaults to `nil`, meaning the MTP head is expected
      inside the target model and the draft context is built against it.
    * `:n_draft` - Max draft tokens generated per iteration. Defaults to `3`.
      Larger values mean fewer model forward passes but lower per-iteration
      acceptance; 2–4 is the sweet spot for MTP in practice. A DFlash drafter
      caps it at its trained block size minus one (7 for Qwen3.8 DFlash 2),
      and that is also where it measured fastest.
    * `:n_ctx` - Context size for both contexts. Defaults to `2048`.
    * Any `Context` tuning option (e.g. `:n_threads`, `:flash_attn`,
      `:type_k`/`:type_v`, `:offload_kqv`). Applied to both contexts.
      `:ctx_other` is not a caller option; the draft always gets the target.

  Returns `{:ok, %MTP{}}` or `{:error, reason}`.
  """
  @spec init(Model.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def init(%Model{} = model, opts \\ []) do
    n_draft = Keyword.get(opts, :n_draft, 3)
    draft_model = Keyword.get(opts, :draft_model)
    type = Keyword.get(opts, :type, :mtp)

    with :ok <- validate_n_draft(n_draft),
         {:ok, head_model} <- validate_drafter(type, model, draft_model) do
      do_init(model, head_model, opts, n_draft, type)
    end
  end

  defp validate_n_draft(n) when is_integer(n) and n > 0, do: :ok
  defp validate_n_draft(_), do: {:error, ":n_draft must be a positive integer"}

  defp validate_drafter(:mtp, target, draft), do: validate_head(target, draft)

  # Whether the drafter really is a DFlash checkpoint for this target is checked
  # in the NIF, where the model's target-layer metadata is readable: upstream's
  # constructor asserts on it, so it has to be refused before init.
  #
  # The device check has to happen here, before the draft context exists. A
  # DFlash drafter computes with the target's token embeddings and LM head
  # (shared through ctx_other), so its context needs a backend for every device
  # the target placed weights on. A target split onto an RPC worker with the
  # drafter on CUDA0 alone aborted the VM with "pre-allocated tensor
  # (output.weight) in a buffer (RPC0) that cannot run the operation".
  defp validate_drafter(type, target, %Model{} = draft) when type in [:dflash, :dspark] do
    case Model.devices(target) -- Model.devices(draft) do
      [] ->
        {:ok, draft}

      missing ->
        {:error,
         "the drafter reads the target's embeddings and LM head, so it must be loaded " <>
           "on every device the target uses; it is missing #{Enum.join(missing, ", ")}. " <>
           "Load it with the target's :devices (a :tensor_split can still keep its " <>
           "own layers on one of them)"}
    end
  end

  defp validate_drafter(type, _target, nil) when type in [:dflash, :dspark],
    do: {:error, "type: #{inspect(type)} needs the drafter GGUF passed as draft_model:"}

  defp validate_drafter(type, _target, other) when type in [:dflash, :dspark],
    do: {:error, ":draft_model must be a LlamaCppEx.Model, got: #{inspect(other)}"}

  defp validate_drafter(type, _target, _draft),
    do: {:error, ":type must be :mtp, :dflash or :dspark, got: #{inspect(type)}"}

  # Which model the draft context is built from, and whether it can actually
  # serve as one. Two shapes reach this: the head inside the target GGUF (no
  # `:draft_model`), and the head in a sidecar GGUF alongside it.
  defp validate_head(%Model{} = target, nil) do
    cond do
      not target.load_mtp ->
        # Upstream gates the MTP head's tensors behind a load-time flag that
        # defaults to false (#26296), and nothing downstream notices they are
        # missing: both contexts build and `common_speculative_init` returns ok,
        # then the first draft fails with `verify decode failed: code=-1`. The
        # layers cannot be attached to an already-loaded model, so refuse here
        # with the actual remedy instead of surfacing that later error.
        {:error,
         "model was loaded without load_mtp: true, so its MTP head layers are " <>
           "absent; reload it with LlamaCppEx.load_model(path, load_mtp: true)"}

      Model.n_layer_nextn(target) == 0 ->
        # Distinct from the branch above and not fixable by any flag: the
        # checkpoint simply has no MTP head. llama.cpp logs "context type MTP
        # requested but model doesn't contain MTP layers" and returns null, which
        # reaches the caller as a bare "failed to create context" with the real
        # reason buried in engine output the caller may not even be showing.
        # Most GGUF conversions of an MTP-capable model drop the head; the
        # publisher usually ships it as a separate repository or sidecar file,
        # which is what `:draft_model` is for.
        {:error,
         "this GGUF contains no MTP head (0 nextn layers), so MTP speculative " <>
           "decoding is unavailable for it; either use an MTP-preserving " <>
           "conversion of the model, or pass the publisher's sidecar MTP GGUF " <>
           "as draft_model: (loaded with load_mtp: true)"}

      true ->
        {:ok, target}
    end
  end

  defp validate_head(%Model{} = target, %Model{} = draft) do
    cond do
      not draft.load_mtp ->
        {:error,
         "draft_model was loaded without load_mtp: true, so the sidecar's MTP " <>
           "head layers are absent; reload it with " <>
           "LlamaCppEx.load_model(path, load_mtp: true)"}

      Model.n_layer_nextn(draft) == 0 ->
        {:error,
         "draft_model contains no MTP head (0 nextn layers) — it is an ordinary " <>
           "model, not an MTP sidecar; pass the publisher's mtp-* GGUF instead"}

      # Upstream compares these two widths with a GGML_ASSERT in the draft-mtp
      # constructor, and GGML_ASSERT is an unconditional ggml_abort: a mismatched
      # pair would take the VM down instead of returning an error. Only a
      # separate drafter can be mismatched, so this is checked on this branch
      # alone, and before any context is built.
      Model.n_embd_out(draft) != Model.n_embd_out(target) ->
        {:error,
         "draft_model hidden width #{Model.n_embd_out(draft)} does not match the " <>
           "target's #{Model.n_embd_out(target)}; the sidecar belongs to a " <>
           "different model than the one it was paired with"}

      true ->
        {:ok, draft}
    end
  end

  defp validate_head(_target, other) do
    {:error, ":draft_model must be a LlamaCppEx.Model, got: #{inspect(other)}"}
  end

  defp do_init(model, head_model, opts, n_draft, type) do
    base_ctx_opts = forwardable_context_opts(opts)

    # Recurrent-state snapshots on the target, one per draft position, as
    # upstream's server sizes them (common_params_speculative.need_n_rs_seq).
    # A hybrid target (GDN layers beside attention) then rolls a rejected draft
    # back natively instead of saving its state every iteration and re-decoding
    # the accepted prefix on every partial accept: measured on Qwen3.8-27B
    # NVFP4 with DFlash 2, that re-decode was the largest cost in the loop. A
    # DFlash drafter clamps n_draft to its block size (at most 16) after init,
    # so the snapshots are capped there rather than sized from what was asked.
    n_rs_seq = if type == :mtp, do: n_draft, else: min(n_draft, 16)
    main_opts = Keyword.merge(base_ctx_opts, ctx_type: :default, n_rs_seq: n_rs_seq)
    # Match upstream (common_speculative_init_result): the draft context is
    # created with n_rs_seq=0, as an MTP context only for an MTP head. The
    # drafters handle rollback themselves (MTP via cached hidden states,
    # DFlash by re-injecting target features), not via recurrent snapshots.
    draft_ctx_type = if type == :mtp, do: :mtp, else: :default
    draft_opts = Keyword.merge(base_ctx_opts, ctx_type: draft_ctx_type, n_rs_seq: 0)

    with {:ok, main_ctx} <- Context.create(model, main_opts),
         {:ok, mtp_ctx} <-
           Context.create(head_model, Keyword.merge(draft_opts, ctx_other: main_ctx)),
         {:ok, spec_ref} <-
           LlamaCppEx.NIF.speculative_init(main_ctx.ref, mtp_ctx.ref, n_draft, type) do
      {:ok,
       %__MODULE__{
         main_ctx: main_ctx,
         mtp_ctx: mtp_ctx,
         spec_ref: spec_ref,
         # A DFlash drafter clamps n_draft to its block size; report what runs.
         n_draft: LlamaCppEx.NIF.speculative_stats(spec_ref).n_draft
       }}
    end
  end

  @doc """
  Returns a lazy stream of generated text pieces.

  ## Options

    * `:max_tokens` - Maximum tokens to generate (default `256`).
    * `:emit_stats_every` - When > 0, also emits `{:stats, snapshot_map}`
      events every Nth token via the underlying message stream. Note: these
      events are filtered out of this `String.t()` stream — to consume them
      use `stream_events/3` instead. Default `0` (off).
    * `:timeout` - Receive timeout in milliseconds (default `60_000`).
    * Any sampler option from `LlamaCppEx.Sampler.create/2` (`:temp`, `:top_k`,
      `:top_p`, `:min_p`, `:seed`, `:penalty_*`, `:grammar`, etc.).

  Each emitted element is the text piece for one accepted token. The stream
  ends on end-of-generation, max-tokens, or error.
  """
  @spec stream(t(), String.t(), keyword()) :: Enumerable.t()
  def stream(%__MODULE__{} = mtp, prompt, opts \\ []) when is_binary(prompt) do
    mtp
    |> stream_events(prompt, opts)
    |> Stream.flat_map(fn
      {:token, _id, text} -> [text]
      _ -> []
    end)
  end

  @doc """
  Like `stream/3`, but yields the raw event tuples emitted by the NIF:

    * `{:token, token_id, text_piece}` - one accepted token
    * `{:stats, snapshot_map}` - periodic stats (only when `:emit_stats_every > 0`)
    * `{:done, final_stats_map}` - generation completed normally
    * `{:eog, nil}` - model emitted an end-of-generation token

  The stream halts after `:done` / `:eog` / `:error`. The final stats map is
  available via `stats/1` on the MTP struct even after the stream ends.
  """
  @spec stream_events(t(), String.t(), keyword()) :: Enumerable.t()
  def stream_events(%__MODULE__{} = mtp, prompt, opts \\ []) when is_binary(prompt) do
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    emit_stats_every = Keyword.get(opts, :emit_stats_every, 0)
    timeout = LlamaCppEx.Options.timeout(opts, :blocking)

    sampler_opts = Keyword.take(opts, Sampler.option_keys())

    Stream.resource(
      fn ->
        start_mtp_stream(mtp, prompt, sampler_opts, max_tokens, emit_stats_every, timeout)
      end,
      fn
        # The halt clause MUST come first. `start_mtp_stream/6` reports failure by
        # adding a `:setup_error` key rather than by flipping the `:done?`
        # discriminant, so `%{state | done?: true}` still matches the
        # `%{setup_error: _}` pattern — with the clauses the other way round this
        # stream emitted the same `{:error, reason}` element forever. `generate/3`
        # drives it with `Enum.to_list/1`, so the caller hung with a growing heap.
        # Reachable from `grammar: "not gbnf"` or any `Tokenizer.encode/2` failure.
        %{done?: true} = state ->
          {:halt, state}

        %{setup_error: reason} = state ->
          {[{:error, reason}], %{state | done?: true}}

        %{ref: ref, timeout: timeout} = state ->
          receive do
            {^ref, {:token, id, text}} -> {[{:token, id, text}], state}
            {^ref, {:stats, snapshot}} -> {[{:stats, snapshot}], state}
            {^ref, {:done, snapshot}} -> {[{:done, snapshot}], %{state | done?: true}}
            {^ref, :eog} -> {[{:eog, nil}], %{state | done?: true}}
            {^ref, {:error, reason}} -> {[{:error, reason}], %{state | done?: true}}
          after
            timeout -> {[{:error, :timeout}], %{state | done?: true}}
          end
      end,
      fn
        %{gen: gen} -> LlamaCppEx.Generator.stop(gen)
        %{} -> :ok
      end
    )
  end

  # Extracted from the Stream.resource start-function so the `with` and the
  # Generator.start closure are not nested three deep inside a `fn` inside a
  # call.
  defp start_mtp_stream(mtp, prompt, sampler_opts, max_tokens, emit_stats_every, timeout) do
    with {:ok, tokens} <- Tokenizer.encode(mtp.main_ctx.model, prompt),
         {:ok, sampler} <- Sampler.create(mtp.main_ctx.model, sampler_opts) do
      {:ok, gen} =
        LlamaCppEx.Generator.start(
          &spec_generate(&1, mtp, sampler, tokens, max_tokens, emit_stats_every)
        )

      # Keep `sampler` alive for the duration of the stream so it doesn't get
      # GC'd while the NIF is still using it. `done?` records that a terminal
      # event was already emitted, so we halt on the next pull.
      %{gen: gen, ref: gen.ref, sampler: sampler, timeout: timeout, done?: false}
    else
      {:error, reason} -> %{setup_error: reason, done?: false}
    end
  end

  defp spec_generate({parent, ref, cancel}, mtp, sampler, tokens, max_tokens, emit_stats_every) do
    LlamaCppEx.NIF.generate_mtp_tokens(
      mtp.spec_ref,
      sampler.ref,
      tokens,
      max_tokens,
      emit_stats_every,
      parent,
      ref,
      cancel
    )
  end

  @doc """
  Synchronously generates text. Equivalent to running `stream/3` and joining
  the pieces into a single binary.

  Accepts the same options as `stream/3`.
  """
  @spec generate(t(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def generate(%__MODULE__{} = mtp, prompt, opts \\ []) when is_binary(prompt) do
    chunks =
      mtp
      |> stream_events(prompt, opts)
      |> Enum.to_list()

    error =
      Enum.find_value(chunks, fn
        {:error, reason} -> reason
        _ -> nil
      end)

    case error do
      nil ->
        text =
          chunks
          |> Enum.flat_map(fn
            {:token, _id, text} -> [text]
            _ -> []
          end)
          |> IO.iodata_to_binary()

        {:ok, text}

      reason ->
        {:error, reason}
    end
  end

  @doc """
  Returns the current MTP statistics snapshot (lock-free read of atomic
  counters). Safe to call at any time — including from another process while
  a stream is in flight.

  Returns a map with keys:

    * `:iters` - speculative loop iterations completed
    * `:drafts_generated` - draft tokens proposed by the MTP head
    * `:drafts_accepted` - draft tokens accepted by the target model
    * `:acceptance_rate` - `drafts_accepted / drafts_generated` (0.0–1.0)
    * `:tokens_emitted` - tokens streamed back to the caller
    * `:tokens_per_sec` - throughput over the active generation window
    * `:timing_us` - `%{draft: μs, verify: μs, sample: μs, ckpt: μs, other: μs,
      total: μs}`. `:ckpt` is the recurrent-state save/restore a hybrid target
      pays when it has fewer recurrent snapshots than `n_draft` (never, for a
      session built by `init/2`), and is zero elsewhere. `:other` is whatever
      falls outside the named buckets, dominated on Metal by GPU-sync waits
      from the previous iteration's async verify decode.
    * `:n_draft` - max draft length configured at init

  Counters are cumulative across all `stream/3` / `generate/3` calls on this
  MTP value.
  """
  @spec stats(t()) :: map()
  def stats(%__MODULE__{spec_ref: ref}), do: LlamaCppEx.NIF.speculative_stats(ref)

  @doc """
  Writes upstream's own speculative stats summary to stdout (via llama.cpp
  logging). Useful when cross-checking acceptance rates against the upstream
  llama-server benchmark output.
  """
  @spec print_stats(t()) :: :ok
  def print_stats(%__MODULE__{spec_ref: ref}) do
    LlamaCppEx.NIF.speculative_print_stats(ref)
    :ok
  end
end
