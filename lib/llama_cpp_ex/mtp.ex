defmodule LlamaCppEx.MTP do
  @moduledoc """
  Multi-Token Prediction (MTP) speculative decoding.

  Drives a target/draft speculative loop where the draft model is the MTP head
  embedded in the same GGUF as the target. On Qwen 3.6 with `n_draft: 3` this
  typically yields ~2x token-generation throughput at ~75% draft acceptance.

  ## Usage

      :ok = LlamaCppEx.init()
      {:ok, model} = LlamaCppEx.load_model("Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
                                            n_gpu_layers: 999)

      {:ok, mtp} = LlamaCppEx.MTP.init(model, n_draft: 3, n_ctx: 8192)

      mtp
      |> LlamaCppEx.MTP.stream("Write a haiku about the sea:", max_tokens: 200)
      |> Stream.each(&IO.write/1)
      |> Stream.run()

      stats = LlamaCppEx.MTP.stats(mtp)
      IO.puts("acceptance: \#{Float.round(stats.acceptance_rate * 100, 1)}%")

  The model GGUF must contain MTP head layers (e.g.
  `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`). Loading a non-MTP quant with `init/2`
  will return `{:error, _}` from `common_speculative_init`.

  Upstream currently requires `n_parallel = 1` for MTP. This module reflects
  that — a single MTP session decodes one sequence at a time. Reuse the same
  `%MTP{}` value across calls to `stream/3` / `generate/3` to avoid rebuilding
  the contexts; KV caches are cleared on each call.
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

  @context_opt_keys [
    :n_threads,
    :n_threads_batch,
    :n_batch,
    :n_ubatch,
    :type_k,
    :type_v,
    :flash_attn,
    :offload_kqv,
    :op_offload,
    :rope_scaling_type,
    :rope_freq_base,
    :rope_freq_scale,
    :yarn_ext_factor,
    :yarn_attn_factor,
    :yarn_beta_fast,
    :yarn_beta_slow,
    :yarn_orig_ctx,
    :attention_type,
    :no_perf,
    :swa_full,
    :n_ctx
  ]

  @doc """
  Initializes an MTP speculative session: builds the target context, the MTP
  draft context (`ctx_type: :mtp`), and the underlying `common_speculative`
  state.

  ## Options

    * `:n_draft` - Max draft tokens generated per iteration. Defaults to `3`.
      Larger values mean fewer model forward passes but lower per-iteration
      acceptance; 2–4 is the sweet spot in practice.
    * `:n_ctx` - Context size for both contexts. Defaults to `2048`.
    * Any `LlamaCppEx.Context` option (e.g. `:n_threads`, `:flash_attn`,
      `:type_k`/`:type_v`, `:offload_kqv`). The same options are applied to
      both the target and draft contexts.

  Returns `{:ok, %MTP{}}` or `{:error, reason}`.
  """
  @spec init(Model.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def init(%Model{} = model, opts \\ []) do
    n_draft = Keyword.get(opts, :n_draft, 3)

    if not (is_integer(n_draft) and n_draft > 0) do
      {:error, ":n_draft must be a positive integer"}
    else
      base_ctx_opts = Keyword.take(opts, @context_opt_keys)
      main_opts = Keyword.merge(base_ctx_opts, ctx_type: :default)
      # Match upstream server: MTP draft context is created with n_rs_seq=0.
      # The MTP impl handles state rollback internally via cached hidden
      # states (pending_h / verify_h), not via recurrent-state snapshots.
      draft_opts = Keyword.merge(base_ctx_opts, ctx_type: :mtp, n_rs_seq: 0)

      with {:ok, main_ctx} <- Context.create(model, main_opts),
           {:ok, mtp_ctx} <- Context.create(model, draft_opts),
           {:ok, spec_ref} <-
             LlamaCppEx.NIF.speculative_init(main_ctx.ref, mtp_ctx.ref, n_draft) do
        {:ok,
         %__MODULE__{
           main_ctx: main_ctx,
           mtp_ctx: mtp_ctx,
           spec_ref: spec_ref,
           n_draft: n_draft
         }}
      end
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
    timeout = Keyword.get(opts, :timeout, 60_000)

    sampler_opts =
      Keyword.take(opts, [
        :seed,
        :temp,
        :top_k,
        :top_p,
        :min_p,
        :penalty_repeat,
        :penalty_freq,
        :penalty_present,
        :grammar,
        :grammar_root
      ])

    Stream.resource(
      fn ->
        {:ok, tokens} = Tokenizer.encode(mtp.main_ctx.model, prompt)
        {:ok, sampler} = Sampler.create(mtp.main_ctx.model, sampler_opts)

        ref = make_ref()
        parent = self()

        gen_pid =
          spawn_link(fn ->
            LlamaCppEx.NIF.generate_mtp_tokens(
              mtp.spec_ref,
              sampler.ref,
              tokens,
              max_tokens,
              emit_stats_every,
              parent,
              ref
            )
          end)

        # Keep `sampler` alive for the duration of the stream so it doesn't
        # get GC'd while the NIF is still using it. `done?` records that a
        # terminal event was already emitted, so we halt on the next pull.
        %{ref: ref, gen_pid: gen_pid, sampler: sampler, timeout: timeout, done?: false}
      end,
      fn
        %{done?: true} = state ->
          {:halt, state}

        %{ref: ref, timeout: timeout} = state ->
          receive do
            {^ref, {:token, id, text}} -> {[{:token, id, text}], state}
            {^ref, {:stats, snapshot}} -> {[{:stats, snapshot}], state}
            {^ref, {:done, snapshot}} -> {[{:done, snapshot}], %{state | done?: true}}
            {^ref, :eog} -> {[{:eog, nil}], %{state | done?: true}}
            {^ref, {:error, reason}} -> {[{:error, reason}], %{state | done?: true}}
          after
            timeout -> {:halt, state}
          end
      end,
      fn %{ref: ref, gen_pid: gen_pid} ->
        Process.unlink(gen_pid)
        Process.exit(gen_pid, :kill)
        flush(ref)
      end
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
    * `:timing_us` - `%{draft: μs, verify: μs, sample: μs, total: μs}`
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

  defp flush(ref) do
    receive do
      {^ref, _} -> flush(ref)
    after
      0 -> :ok
    end
  end
end
