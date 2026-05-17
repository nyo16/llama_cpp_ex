defmodule LlamaCppEx.Server do
  @moduledoc """
  GenServer for continuous batched multi-sequence inference.

  Manages a shared model/context and serves multiple concurrent callers
  using a slot pool with continuous batching — one forward pass per tick
  with decode tokens and prefill chunks mixed in a single batch.

  ## Example

      {:ok, server} = LlamaCppEx.Server.start_link(
        model_path: "model.gguf",
        n_gpu_layers: -1,
        n_parallel: 4,
        n_ctx: 8192
      )

      # Sync generation
      {:ok, text} = LlamaCppEx.Server.generate(server, "Once upon a time", max_tokens: 100)

      # Streaming
      LlamaCppEx.Server.stream(server, "Tell me a story", max_tokens: 200)
      |> Enum.each(&IO.write/1)

  ## Telemetry

  The server emits the following telemetry events:

  ### `[:llama_cpp_ex, :server, :tick]`

  Emitted after each batch forward pass.

  Measurements:

    * `:batch_size` - Total tokens in the batch.
    * `:decode_tokens` - Number of decode (generation) tokens.
    * `:prefill_tokens` - Number of prefill (prompt) tokens.
    * `:active_slots` - Slots currently prefilling or generating.
    * `:queue_depth` - Requests waiting for a slot.
    * `:eval_ms` - Forward pass wall time in milliseconds.

  Metadata:

    * `:server` - PID of the server process.

  ### `[:llama_cpp_ex, :server, :request, :start]`

  Emitted when a slot is assigned to a request and prefill begins.

  Measurements:

    * `:prompt_tokens` - Number of prompt tokens.
    * `:prefix_cache_tokens` - Number of prompt tokens reused from the KV
      prefix cache (`0` when `cache_prompt: false`).

  Metadata:

    * `:server` - PID of the server process.
    * `:seq_id` - Slot sequence ID.
    * `:mode` - `:generate` or `:stream`.

  ### `[:llama_cpp_ex, :server, :request, :done]`

  Emitted when a request (generate or stream) completes.

  Measurements:

    * `:prompt_tokens` - Number of prompt tokens.
    * `:generated_tokens` - Number of tokens generated.
    * `:duration_ms` - Total request duration in milliseconds.
    * `:ttft_ms` - Time to first token in milliseconds.
    * `:prompt_eval_rate` - Prompt evaluation speed (tokens/sec).
    * `:generation_rate` - Generation speed (tokens/sec).
    * `:prefix_cache_tokens` - Number of prompt tokens skipped via prefix cache.
    * `:prefix_cache_ratio` - Ratio of cached to total prompt tokens (0.0–1.0).

  Metadata:

    * `:server` - PID of the server process.
    * `:seq_id` - Slot sequence ID (integer).
    * `:mode` - `:generate` or `:stream`.
    * `:stop_reason` - `:eog` (end-of-generation token sampled) or
      `:max_tokens` (request `max_tokens` reached).

  ### `[:llama_cpp_ex, :server, :request, :exception]`

  Emitted when an inference error aborts an active request (e.g. the
  underlying `batch_eval` returns an error). Measurement shape matches
  `:done` so handlers can aggregate them together; `:stop_reason` is
  `:error` and the failure reason is in `:reason`.

  Metadata:

    * `:server` - PID of the server process.
    * `:seq_id` - Slot sequence ID.
    * `:mode` - `:generate` or `:stream`.
    * `:stop_reason` - `:error`.
    * `:reason` - The underlying failure term from the NIF.

  """

  use GenServer

  require Logger

  alias LlamaCppEx.{Model, Context, Sampler, Tokenizer}

  defstruct [
    :model,
    :ctx,
    :sampler_opts,
    slots: %{},
    queue: nil,
    n_parallel: 4,
    n_batch: 2048,
    chunk_size: 512,
    cache_prompt: false,
    # `:part`/`:rs` = partial seq_rm works; `:full` = whole-sequence only
    # (hybrid GDN models like Qwen 3.5/3.6); `:no` = no memory module. We
    # only do prefix-cache partial trims when this is `:part` or `:rs`.
    seq_rm_kind: :part,
    batch_strategy: LlamaCppEx.Server.Strategy.DecodeMaximal,
    tick_scheduled: false
  ]

  # --- Client API ---

  @doc """
  Starts the server.

  ## Options

    * `:model_path` (required) - Path to the GGUF model file.
    * `:n_gpu_layers` - GPU layers. Defaults to `99`.
    * `:n_ctx` - Total context size (shared across slots). Defaults to `8192`.
    * `:n_parallel` - Number of concurrent slots. Defaults to `4`.
    * `:n_batch` - Batch size. Defaults to `n_ctx`.
    * `:chunk_size` - Max prefill tokens per slot per tick. Defaults to `512`.
    * `:max_queue` - Max queued requests. `0` for unlimited. Defaults to `0`.
    * `:cache_prompt` - Retain KV cache between requests on the same slot for
      prefix reuse. Defaults to `false`. Set to `true` for multi-turn chat.
    * `:batch_strategy` - Batch building strategy module. Defaults to
      `LlamaCppEx.Server.Strategy.DecodeMaximal`. See `LlamaCppEx.Server.BatchStrategy`.
    * Sampling options: `:temp`, `:top_k`, `:top_p`, `:min_p`, `:seed`, `:penalty_repeat`,
      `:penalty_freq`, `:penalty_present`, `:grammar`, `:grammar_root`.
    * GenServer options like `:name`.

  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    {server_opts, gen_opts} = Keyword.split(opts, [:name])
    GenServer.start_link(__MODULE__, gen_opts, server_opts)
  end

  @doc """
  Generates text synchronously. Blocks until generation is complete.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:timeout` - Call timeout in ms. Defaults to `60_000`.

  """
  @spec generate(GenServer.server(), String.t(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def generate(server, prompt, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 60_000)
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    GenServer.call(server, {:generate, prompt, max_tokens}, timeout)
  end

  @doc """
  Returns a stream of generated text chunks.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:timeout` - Per-token timeout. Defaults to `30_000`.

  """
  @spec stream(GenServer.server(), String.t(), keyword()) :: Enumerable.t()
  def stream(server, prompt, opts \\ []) do
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    timeout = Keyword.get(opts, :timeout, 30_000)

    Stream.resource(
      fn ->
        ref = make_ref()
        :ok = GenServer.call(server, {:stream, prompt, max_tokens, self(), ref})
        {ref, timeout}
      end,
      fn {ref, timeout} = state ->
        receive do
          {^ref, {:token, text}} -> {[text], state}
          {^ref, :done} -> {:halt, state}
          {^ref, {:error, _reason}} -> {:halt, state}
        after
          timeout -> {:halt, state}
        end
      end,
      fn {ref, _timeout} ->
        receive do
          {^ref, _} -> :ok
        after
          0 -> :ok
        end
      end
    )
  end

  @doc """
  Generates text from pre-tokenized input. Blocks until generation is complete.

  Use `get_model/1` to obtain the model for tokenization outside the server.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:timeout` - Call timeout in ms. Defaults to `60_000`.

  """
  @spec generate_tokens(GenServer.server(), [integer()], keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def generate_tokens(server, token_ids, opts \\ []) when is_list(token_ids) do
    timeout = Keyword.get(opts, :timeout, 60_000)
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    GenServer.call(server, {:generate_tokens, token_ids, max_tokens}, timeout)
  end

  @doc """
  Returns a stream of generated text chunks from pre-tokenized input.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:timeout` - Per-token timeout. Defaults to `30_000`.

  """
  @spec stream_tokens(GenServer.server(), [integer()], keyword()) :: Enumerable.t()
  def stream_tokens(server, token_ids, opts \\ []) when is_list(token_ids) do
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    timeout = Keyword.get(opts, :timeout, 30_000)

    Stream.resource(
      fn ->
        ref = make_ref()
        :ok = GenServer.call(server, {:stream_tokens, token_ids, max_tokens, self(), ref})
        {ref, timeout}
      end,
      fn {ref, timeout} = state ->
        receive do
          {^ref, {:token, text}} -> {[text], state}
          {^ref, :done} -> {:halt, state}
          {^ref, {:error, _reason}} -> {:halt, state}
        after
          timeout -> {:halt, state}
        end
      end,
      fn {ref, _timeout} ->
        receive do
          {^ref, _} -> :ok
        after
          0 -> :ok
        end
      end
    )
  end

  @doc """
  Returns the model struct for external tokenization.

  The model resource is reference-counted and thread-safe for read-only
  operations like tokenization.
  """
  @spec get_model(GenServer.server()) :: Model.t()
  def get_model(server) do
    GenServer.call(server, :get_model)
  end

  @doc """
  Returns a snapshot of the server's current state.
  """
  @spec get_stats(GenServer.server()) :: map()
  def get_stats(server) do
    GenServer.call(server, :get_stats)
  end

  # --- Server callbacks ---

  @impl true
  def init(opts) do
    model_path = Keyword.fetch!(opts, :model_path)
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, 99)
    n_parallel = Keyword.get(opts, :n_parallel, 4)
    n_ctx = Keyword.get(opts, :n_ctx, 8192)
    n_batch = Keyword.get(opts, :n_batch, n_ctx)
    chunk_size = Keyword.get(opts, :chunk_size, 512)
    cache_prompt = Keyword.get(opts, :cache_prompt, false)
    batch_strategy = Keyword.get(opts, :batch_strategy, LlamaCppEx.Server.Strategy.DecodeMaximal)

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

    model_opts =
      Keyword.take(opts, [
        :main_gpu,
        :split_mode,
        :tensor_split,
        :use_mlock,
        :use_direct_io,
        :check_tensors
      ])

    context_opts =
      Keyword.take(opts, [
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
        :swa_full
      ])

    :ok = LlamaCppEx.init()
    {:ok, model} = Model.load(model_path, [n_gpu_layers: n_gpu_layers] ++ model_opts)

    {:ok, ctx} =
      Context.create(
        model,
        [n_ctx: n_ctx, n_batch: n_batch, n_seq_max: n_parallel] ++ context_opts
      )

    # Probe seq_rm support BEFORE any decode work — the call has the side
    # effect of clearing KV memory. Hybrid models (GDN, e.g. Qwen 3.5/3.6)
    # report `:full`, meaning partial range trims aren't supported; we'd
    # otherwise produce M-RoPE position-mismatch aborts in the prefix-cache
    # path when an old slot's KV tail extends past the new prompt's prefix
    # match.
    seq_rm_kind = LlamaCppEx.NIF.context_can_seq_rm(ctx.ref)

    if cache_prompt and seq_rm_kind == :full do
      Logger.info(
        "LlamaCppEx.Server: cache_prompt: true requested but model reports " <>
          "seq_rm support = :full (hybrid GDN). Prefix cache will only fire " <>
          "for exact-prefix continuations; cache hits requiring partial KV " <>
          "trim will fall back to a full slot reset."
      )
    end

    slots =
      for seq_id <- 0..(n_parallel - 1), into: %{} do
        {:ok, sampler} = Sampler.create(model, sampler_opts)

        slot = %{
          state: :idle,
          sampler: sampler,
          from: nil,
          stream_pid: nil,
          stream_ref: nil,
          prompt_tokens: [],
          prefill_pos: 0,
          pos: 0,
          pending_token: nil,
          batch_idx: -1,
          tokens_generated: 0,
          max_tokens: 0,
          accumulated_text: "",
          t_start: nil,
          t_first_token: nil,
          n_prompt_tokens: 0,
          # Prefix cache fields
          cached_tokens: [],
          cached_pos: 0,
          generated_token_ids: [],
          n_prefix_cache_tokens: 0
        }

        {seq_id, slot}
      end

    state = %__MODULE__{
      model: model,
      ctx: ctx,
      sampler_opts: sampler_opts,
      slots: slots,
      queue: :queue.new(),
      n_parallel: n_parallel,
      n_batch: n_batch,
      chunk_size: chunk_size,
      cache_prompt: cache_prompt,
      seq_rm_kind: seq_rm_kind,
      batch_strategy: batch_strategy
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:generate, prompt, max_tokens}, from, state) do
    {:ok, tokens} = Tokenizer.encode(state.model, prompt)

    case acquire_slot(state, tokens) do
      {:ok, seq_id, state} ->
        state = init_slot(state, seq_id, tokens, max_tokens, from, nil, nil)
        state = maybe_schedule_tick(state)
        {:noreply, state}

      :no_slots ->
        state = enqueue_request(state, {:generate, tokens, max_tokens, from, nil, nil})
        {:noreply, state}
    end
  end

  def handle_call({:generate_tokens, token_ids, max_tokens}, from, state) do
    if token_ids == [] do
      {:reply, {:error, "token list cannot be empty"}, state}
    else
      case acquire_slot(state, token_ids) do
        {:ok, seq_id, state} ->
          state = init_slot(state, seq_id, token_ids, max_tokens, from, nil, nil)
          state = maybe_schedule_tick(state)
          {:noreply, state}

        :no_slots ->
          state = enqueue_request(state, {:generate, token_ids, max_tokens, from, nil, nil})
          {:noreply, state}
      end
    end
  end

  def handle_call({:stream, prompt, max_tokens, pid, ref}, from, state) do
    {:ok, tokens} = Tokenizer.encode(state.model, prompt)

    case acquire_slot(state, tokens) do
      {:ok, seq_id, state} ->
        state = init_slot(state, seq_id, tokens, max_tokens, nil, pid, ref)
        GenServer.reply(from, :ok)
        state = maybe_schedule_tick(state)
        {:noreply, state}

      :no_slots ->
        GenServer.reply(from, :ok)
        state = enqueue_request(state, {:stream, tokens, max_tokens, nil, pid, ref})
        {:noreply, state}
    end
  end

  def handle_call({:stream_tokens, token_ids, max_tokens, pid, ref}, from, state) do
    case acquire_slot(state, token_ids) do
      {:ok, seq_id, state} ->
        state = init_slot(state, seq_id, token_ids, max_tokens, nil, pid, ref)
        GenServer.reply(from, :ok)
        state = maybe_schedule_tick(state)
        {:noreply, state}

      :no_slots ->
        GenServer.reply(from, :ok)
        state = enqueue_request(state, {:stream, token_ids, max_tokens, nil, pid, ref})
        {:noreply, state}
    end
  end

  def handle_call(:get_model, _from, state) do
    {:reply, state.model, state}
  end

  def handle_call(:get_stats, _from, state) do
    counts =
      Enum.reduce(state.slots, %{idle: 0, prefilling: 0, generating: 0}, fn {_id, slot}, acc ->
        Map.update!(acc, slot.state, &(&1 + 1))
      end)

    stats = %{
      active_slots: counts.prefilling + counts.generating,
      idle_slots: counts.idle,
      prefilling_slots: counts.prefilling,
      queue_depth: :queue.len(state.queue),
      n_parallel: state.n_parallel,
      n_batch: state.n_batch
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_info(:tick, state) do
    state = %{state | tick_scheduled: false}
    state = run_tick(state)
    {:noreply, state}
  end

  # --- Internal: Slot management ---

  defp acquire_slot(state, tokens) do
    idle_slots = Enum.filter(state.slots, fn {_id, slot} -> slot.state == :idle end)

    case idle_slots do
      [] ->
        :no_slots

      slots when state.cache_prompt and tokens != [] ->
        # Prefer the slot with the longest cached prefix match
        {best_id, _} =
          Enum.max_by(slots, fn {_id, slot} ->
            common_prefix_length(tokens, slot.cached_tokens)
          end)

        {:ok, best_id, state}

      [{seq_id, _} | _] ->
        {:ok, seq_id, state}
    end
  end

  defp init_slot(state, seq_id, tokens, max_tokens, from, stream_pid, stream_ref) do
    slot = state.slots[seq_id]

    # Prefix cache: find common prefix with cached KV. On models that only
    # support whole-sequence seq_rm (`:full`, e.g. hybrid GDN), a partial
    # trim would silently fail and leave stale KV past `n_match`, producing
    # an M-RoPE position-mismatch abort on the next decode. Disable the
    # cache hit in that case unless the new prompt extends the old one
    # exactly (no trim needed).
    raw_match =
      if state.cache_prompt do
        common_prefix_length(tokens, slot.cached_tokens)
      else
        0
      end

    needs_trim = raw_match > 0 and raw_match < slot.cached_pos

    n_match =
      if needs_trim and state.seq_rm_kind == :full do
        0
      else
        raw_match
      end

    cond do
      n_match > 0 and n_match < slot.cached_pos ->
        # Trim KV cache beyond the matched prefix (only safe on `:part`/`:rs`).
        true = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, n_match, -1)

      n_match > 0 ->
        # Exact-prefix continuation; nothing to trim.
        :noop

      true ->
        # No usable match — clear everything for this slot.
        _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
    end

    # Reset sampler for fresh generation
    Sampler.reset(slot.sampler)

    slot = %{
      slot
      | state: :prefilling,
        from: from,
        stream_pid: stream_pid,
        stream_ref: stream_ref,
        prompt_tokens: tokens,
        prefill_pos: n_match,
        pos: n_match,
        pending_token: nil,
        batch_idx: -1,
        tokens_generated: 0,
        max_tokens: max_tokens,
        accumulated_text: "",
        t_start: System.monotonic_time(),
        t_first_token: nil,
        n_prompt_tokens: length(tokens),
        generated_token_ids: [],
        n_prefix_cache_tokens: n_match
    }

    :telemetry.execute(
      [:llama_cpp_ex, :server, :request, :start],
      %{prompt_tokens: length(tokens), prefix_cache_tokens: n_match},
      %{server: self(), seq_id: seq_id, mode: slot_mode(slot)}
    )

    put_in(state.slots[seq_id], slot)
  end

  defp slot_mode(%{stream_pid: pid}) when is_pid(pid), do: :stream
  defp slot_mode(_slot), do: :generate

  defp enqueue_request(state, request) do
    %{state | queue: :queue.in(request, state.queue)}
  end

  defp dequeue_into_slot(state) do
    case :queue.out(state.queue) do
      {{:value, request}, queue} ->
        state = %{state | queue: queue}
        tokens = request_tokens(request)

        case acquire_slot(state, tokens) do
          {:ok, seq_id, state} ->
            state = assign_queued_request(state, seq_id, request)
            dequeue_into_slot(state)

          :no_slots ->
            # Put it back
            %{state | queue: :queue.in_r(request, state.queue)}
        end

      {:empty, _queue} ->
        state
    end
  end

  defp request_tokens({_type, tokens, _max, _from, _pid, _ref}), do: tokens

  defp assign_queued_request(state, seq_id, {:generate, tokens, max_tokens, from, _, _}) do
    init_slot(state, seq_id, tokens, max_tokens, from, nil, nil)
  end

  defp assign_queued_request(state, seq_id, {:stream, tokens, max_tokens, _, pid, ref}) do
    init_slot(state, seq_id, tokens, max_tokens, nil, pid, ref)
  end

  # --- Internal: Tick loop ---

  defp run_tick(state) do
    # Phase 1: Finish completed slots
    state = finish_completed_slots(state)

    # Phase 1b: Dequeue waiting requests into freed slots
    state = dequeue_into_slot(state)

    # Phase 2: Build batch
    {entries, state} = build_batch(state)

    # Capture TTFT immediately after the strategy ran. The strategy sends
    # each decode token piece to its stream consumer before returning, so
    # by now any slot whose first decode just streamed has tokens_generated
    # == 1 — that's the user-perceived first-token timestamp.
    state = mark_first_tokens(state)

    if entries == [] do
      state
    else
      # Count decode tokens before sampling (slots may transition after)
      n_decode =
        Enum.count(state.slots, fn {_id, s} ->
          s.state == :generating and s.batch_idx >= 0
        end)

      # Phase 3: Forward pass
      tick_start = System.monotonic_time()

      case LlamaCppEx.NIF.batch_eval(state.ctx.ref, entries) do
        :ok ->
          tick_end = System.monotonic_time()

          # Phase 4: Sample
          state = sample_generating_slots(state)
          state = sample_completed_prefills(state)
          state = advance_incomplete_prefills(state)

          # Emit tick telemetry
          :telemetry.execute(
            [:llama_cpp_ex, :server, :tick],
            %{
              batch_size: length(entries),
              decode_tokens: n_decode,
              prefill_tokens: length(entries) - n_decode,
              active_slots: Enum.count(state.slots, fn {_id, s} -> s.state != :idle end),
              queue_depth: :queue.len(state.queue),
              eval_ms: (tick_end - tick_start) / 1_000_000
            },
            %{server: self()}
          )

          # Phase 5: Continue
          if Enum.any?(state.slots, fn {_id, slot} -> slot.state != :idle end) do
            maybe_schedule_tick(state)
          else
            state
          end

        {:error, reason} ->
          Logger.error("batch forward pass failed: #{reason}")
          fail_all_active_slots(state, reason)
      end
    end
  end

  # Phase 1: Check generating slots for completion
  defp finish_completed_slots(state) do
    generating_slots =
      state.slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :generating and slot.pending_token != nil
      end)

    Enum.reduce(generating_slots, state, fn {seq_id, _slot}, state ->
      slot = state.slots[seq_id]
      token = slot.pending_token
      is_eog = LlamaCppEx.NIF.vocab_is_eog(state.model.ref, token)

      cond do
        is_eog ->
          # pending_token is the EOG control token — finish without streaming it.
          finish_slot(state, seq_id, :eog)

        slot.tokens_generated >= slot.max_tokens ->
          # We've already streamed max_tokens; pending_token is beyond the limit.
          finish_slot(state, seq_id, :max_tokens)

        true ->
          state
      end
    end)
  end

  # Phase 2: Build batch via pluggable strategy
  defp build_batch(state) do
    opts = [model_ref: state.model.ref, queue_depth: :queue.len(state.queue)]

    {entries, updated_slots} =
      state.batch_strategy.build_batch(state.slots, state.n_batch, state.chunk_size, opts)

    {entries, %{state | slots: updated_slots}}
  end

  defp mark_first_tokens(state) do
    now = System.monotonic_time()

    slots =
      Enum.reduce(state.slots, state.slots, fn {seq_id, slot}, acc ->
        if slot.t_first_token == nil and slot.tokens_generated >= 1 do
          Map.put(acc, seq_id, %{slot | t_first_token: now})
        else
          acc
        end
      end)

    %{state | slots: slots}
  end

  # Phase 4a: Sample for generating slots
  defp sample_generating_slots(state) do
    generating_slots =
      state.slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :generating and slot.batch_idx >= 0
      end)

    Enum.reduce(generating_slots, state, fn {seq_id, _slot}, state ->
      slot = state.slots[seq_id]

      next_token =
        LlamaCppEx.NIF.sampler_sample_at(slot.sampler.ref, state.ctx.ref, slot.batch_idx)

      LlamaCppEx.NIF.sampler_accept(slot.sampler.ref, next_token)

      slot = %{
        slot
        | pos: slot.pos + 1,
          pending_token: next_token,
          batch_idx: -1
      }

      put_in(state.slots[seq_id], slot)
    end)
  end

  # Phase 4b: Sample for prefilling slots that completed
  defp sample_completed_prefills(state) do
    completed_prefills =
      state.slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :prefilling and slot.batch_idx >= 0 and
          slot.prefill_pos >= length(slot.prompt_tokens)
      end)

    Enum.reduce(completed_prefills, state, fn {seq_id, _slot}, state ->
      slot = state.slots[seq_id]

      first_token =
        LlamaCppEx.NIF.sampler_sample_at(slot.sampler.ref, state.ctx.ref, slot.batch_idx)

      LlamaCppEx.NIF.sampler_accept(slot.sampler.ref, first_token)

      slot = %{
        slot
        | state: :generating,
          pending_token: first_token,
          pos: length(slot.prompt_tokens),
          batch_idx: -1
      }

      put_in(state.slots[seq_id], slot)
    end)
  end

  # Phase 4c: Advance incomplete prefills (no sampling needed)
  defp advance_incomplete_prefills(state) do
    incomplete_prefills =
      state.slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :prefilling and slot.prefill_pos < length(slot.prompt_tokens)
      end)

    Enum.reduce(incomplete_prefills, state, fn {seq_id, _slot}, state ->
      slot = state.slots[seq_id]
      slot = %{slot | batch_idx: -1}
      put_in(state.slots[seq_id], slot)
    end)
  end

  # --- Internal: Slot completion ---

  defp finish_slot(state, seq_id, stop_reason) do
    slot = state.slots[seq_id]
    t_end = System.monotonic_time()

    if slot.from do
      GenServer.reply(slot.from, {:ok, slot.accumulated_text})
    end

    if slot.stream_pid && slot.stream_ref do
      send(slot.stream_pid, {slot.stream_ref, :done})
    end

    # Emit telemetry
    emit_request_done(slot, seq_id, t_end, stop_reason)

    reset_slot(state, seq_id)
  end

  defp emit_request_done(slot, seq_id, t_end, stop_reason) do
    duration_ns = t_end - slot.t_start
    duration_ms = duration_ns / 1_000_000

    ttft_ms =
      if slot.t_first_token do
        (slot.t_first_token - slot.t_start) / 1_000_000
      else
        duration_ms
      end

    gen_duration_s = (t_end - (slot.t_first_token || slot.t_start)) / 1_000_000_000
    prompt_duration_s = ttft_ms / 1000

    prompt_eval_rate =
      if prompt_duration_s > 0, do: slot.n_prompt_tokens / prompt_duration_s, else: 0.0

    generation_rate =
      if gen_duration_s > 0, do: slot.tokens_generated / gen_duration_s, else: 0.0

    mode = slot_mode(slot)

    Logger.debug(
      "slot #{seq_id} done (#{stop_reason}): #{slot.n_prompt_tokens} prompt tokens (#{Float.round(prompt_eval_rate, 1)} t/s), " <>
        "#{slot.tokens_generated} generated (#{Float.round(generation_rate, 1)} t/s), " <>
        "ttft #{Float.round(ttft_ms, 1)}ms, total #{Float.round(duration_ms, 1)}ms"
    )

    prefix_cache_ratio =
      if slot.n_prompt_tokens > 0,
        do: slot.n_prefix_cache_tokens / slot.n_prompt_tokens,
        else: 0.0

    :telemetry.execute(
      [:llama_cpp_ex, :server, :request, :done],
      %{
        prompt_tokens: slot.n_prompt_tokens,
        generated_tokens: slot.tokens_generated,
        duration_ms: duration_ms,
        ttft_ms: ttft_ms,
        prompt_eval_rate: prompt_eval_rate,
        generation_rate: generation_rate,
        prefix_cache_tokens: slot.n_prefix_cache_tokens,
        prefix_cache_ratio: prefix_cache_ratio
      },
      %{server: self(), seq_id: seq_id, mode: mode, stop_reason: stop_reason}
    )
  end

  defp emit_request_exception(slot, seq_id, reason) do
    # Mirror `:done`'s measurement shape so dashboards can aggregate them.
    # Timings are best-effort: if generation never started, ttft/duration
    # fall back to the wall time since slot acquisition.
    t_end = System.monotonic_time()
    t_start = slot.t_start || t_end
    duration_ms = (t_end - t_start) / 1_000_000

    ttft_ms =
      if slot.t_first_token, do: (slot.t_first_token - t_start) / 1_000_000, else: duration_ms

    prompt_duration_s = ttft_ms / 1000
    gen_duration_s = (t_end - (slot.t_first_token || t_start)) / 1_000_000_000

    prompt_eval_rate =
      if prompt_duration_s > 0, do: slot.n_prompt_tokens / prompt_duration_s, else: 0.0

    generation_rate =
      if gen_duration_s > 0, do: slot.tokens_generated / gen_duration_s, else: 0.0

    prefix_cache_ratio =
      if slot.n_prompt_tokens > 0,
        do: slot.n_prefix_cache_tokens / slot.n_prompt_tokens,
        else: 0.0

    :telemetry.execute(
      [:llama_cpp_ex, :server, :request, :exception],
      %{
        prompt_tokens: slot.n_prompt_tokens,
        generated_tokens: slot.tokens_generated,
        duration_ms: duration_ms,
        ttft_ms: ttft_ms,
        prompt_eval_rate: prompt_eval_rate,
        generation_rate: generation_rate,
        prefix_cache_tokens: slot.n_prefix_cache_tokens,
        prefix_cache_ratio: prefix_cache_ratio
      },
      %{
        server: self(),
        seq_id: seq_id,
        mode: slot_mode(slot),
        stop_reason: :error,
        reason: reason
      }
    )
  end

  defp reset_slot(state, seq_id) do
    slot = state.slots[seq_id]
    Sampler.reset(slot.sampler)

    # Build full token history for prefix cache
    {cached_tokens, cached_pos} =
      if state.cache_prompt do
        all_tokens = slot.prompt_tokens ++ Enum.reverse(slot.generated_token_ids)
        {all_tokens, slot.pos}
      else
        LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
        {[], 0}
      end

    slot = %{
      slot
      | state: :idle,
        from: nil,
        stream_pid: nil,
        stream_ref: nil,
        prompt_tokens: [],
        prefill_pos: 0,
        pos: 0,
        pending_token: nil,
        batch_idx: -1,
        tokens_generated: 0,
        max_tokens: 0,
        accumulated_text: "",
        t_start: nil,
        t_first_token: nil,
        n_prompt_tokens: 0,
        cached_tokens: cached_tokens,
        cached_pos: cached_pos,
        generated_token_ids: [],
        n_prefix_cache_tokens: 0
    }

    put_in(state.slots[seq_id], slot)
  end

  defp fail_all_active_slots(state, reason) do
    active_slots =
      Enum.filter(state.slots, fn {_id, slot} -> slot.state != :idle end)

    Enum.reduce(active_slots, state, fn {seq_id, slot}, state ->
      if slot.from do
        GenServer.reply(slot.from, {:error, "inference failed: #{reason}"})
      end

      if slot.stream_pid && slot.stream_ref do
        send(slot.stream_pid, {slot.stream_ref, {:error, reason}})
      end

      emit_request_exception(slot, seq_id, reason)

      # Clear KV cache and reset — don't preserve cache on error
      LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
      Sampler.reset(slot.sampler)

      slot = %{
        slot
        | state: :idle,
          from: nil,
          stream_pid: nil,
          stream_ref: nil,
          prompt_tokens: [],
          prefill_pos: 0,
          pos: 0,
          pending_token: nil,
          batch_idx: -1,
          tokens_generated: 0,
          max_tokens: 0,
          accumulated_text: "",
          t_start: nil,
          t_first_token: nil,
          n_prompt_tokens: 0,
          cached_tokens: [],
          cached_pos: 0,
          generated_token_ids: [],
          n_prefix_cache_tokens: 0
      }

      put_in(state.slots[seq_id], slot)
    end)
  end

  defp maybe_schedule_tick(state) do
    if state.tick_scheduled do
      state
    else
      send(self(), :tick)
      %{state | tick_scheduled: true}
    end
  end

  @doc false
  def common_prefix_length(a, b) do
    a
    |> Enum.zip(b)
    |> Enum.take_while(fn {x, y} -> x == y end)
    |> length()
  end
end
