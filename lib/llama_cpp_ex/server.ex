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

  ### `[:llama_cpp_ex, :server, :kv_pressure]`

  Emitted when a forward pass hit KV-cache pressure (`llama_decode == 1`) and
  the server recovered by purging idle slots' cached prefixes and/or splitting
  the batch.

  Measurements:

    * `:purged_slots` - Number of idle slots whose cached KV was dropped.
    * `:batch_splits` - Number of times the batch was halved to fit.

  Metadata:

    * `:server` - PID of the server process.
    * `:purged_seq_ids` - Sequence IDs whose caches were purged.

  ### `[:llama_cpp_ex, :server, :prefix_instability]`

  Emitted when a cache-eligible request matches only 10–50% of a slot's
  cached history — the signature of a chat template that rewrites earlier
  turns (e.g. stripping thinking blocks), silently defeating prefix caching.

  Measurements:

    * `:matched_tokens` - Length of the common prefix actually reusable.
    * `:cached_tokens` - Length of the cached history that was expected to match.

  Metadata:

    * `:server` - PID of the server process.
    * `:seq_id` - Slot sequence ID.

  ### `[:llama_cpp_ex, :server, :ram_cache]`

  Emitted on level-2 RAM prompt cache activity (see `:prompt_cache_ram_mb`).

  Measurements:

    * `:bytes` - Size of the entry involved.
    * `:tokens` - Cached prefix length of the entry involved.
    * `:total_bytes` - Cache size after the operation.
    * `:entries` - Entry count after the operation.

  Metadata:

    * `:server` - PID of the server process.
    * `:op` - `:save`, `:restore`, or `:evict`.

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

  alias LlamaCppEx.{Context, Model, Sampler, Tokenizer}

  # Sampling options accepted both at server start (defaults) and per request
  # (overrides). Keep in sync with LlamaCppEx.Sampler.create/2.
  @sampler_opt_keys [
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
  ]

  # Per-request options accepted by generate/stream and carried through the
  # queue into the slot. Server-level options provide the defaults.
  @request_opt_keys [:cache_prompt, :session] ++ @sampler_opt_keys

  # Evicted caches below this many tokens are not worth a KV-sized state copy.
  @ram_cache_min_tokens 32

  # Restore a RAM cache entry only when the usable fraction clears this bar
  # (llama-server's f_keep heuristic) — restoring is a KV-sized memcpy.
  @ram_cache_min_keep 0.25

  # Prefix-instability heuristic: a cached history this long that matches
  # 10–50% of its tokens looks like a chat template rewriting history
  # (thinking-strip etc.) rather than a brand-new conversation (~0%) or a
  # clean continuation (~100%).
  @prefix_instability_min_cached 64

  defstruct [
    :model,
    :ctx,
    :sampler_opts,
    slots: %{},
    queue: nil,
    sessions: %{},
    n_parallel: 4,
    n_batch: 2048,
    chunk_size: 512,
    cache_prompt: true,
    # True when cross-slot prefix sharing is safe: unified KV (partial
    # cross-stream seq_cp aborts in split mode) AND partial seq_rm support
    # (hybrid GDN recurrent state cannot be partially copied).
    cross_slot_sharing: false,
    # Level-2 prompt cache: evicted slot KV states saved to RAM, FIFO under a
    # byte budget. 0 MB = disabled.
    prompt_cache_ram_mb: 0,
    ram_cache: [],
    ram_cache_bytes: 0,
    prefix_instability_warned: false,
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
    * `:n_batch` - Max tokens per forward pass. Defaults to `min(n_ctx, 2048)`.
      Bounds worst-case tick latency: one huge prompt can occupy at most
      `n_batch` tokens of a tick, so decode tokens of other slots are never
      delayed by more than one `n_batch`-sized pass. Raise it for pure batch
      throughput (fewer, larger passes); lower it (or lower `:chunk_size`) for
      smoother streaming latency under mixed load.
    * `:chunk_size` - Max prefill tokens per slot per tick. Defaults to `512`.
    * `:max_queue` - Max queued requests. `0` for unlimited. Defaults to `0`.
    * `:cache_prompt` - Retain KV cache between requests on the same slot for
      prefix reuse. Defaults to `true` (matching llama-server). Overridable
      per request via the `:cache_prompt` option on `generate/3` and friends.
    * `:kv_unified` - Share one KV buffer across all slots instead of splitting
      `n_ctx` evenly (`n_ctx/n_parallel` each). Enables cross-slot prefix
      sharing: a system prompt cached by any slot is adopted by every other
      slot via a metadata-only copy. Slots then compete for the shared `n_ctx`
      budget, and idle slots' caches are purged under KV pressure. Defaults to
      `true`. Set `false` for strictly isolated per-slot budgets.
    * `:prompt_cache_ram_mb` - Byte budget (in MB) for the level-2 RAM prompt
      cache: when a slot's cached prefix is about to be destroyed it is
      serialized to RAM and can be restored later instead of re-prefilling.
      State blobs are KV-sized (up to hundreds of MB for long contexts) —
      entries larger than the budget are never stored, so a small budget
      degrades to "disabled" rather than OOM. Defaults to `0` (off).
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
    * `:cache_prompt` - Reuse/retain this request's KV prefix on the slot.
      Defaults to the server-level `:cache_prompt` setting.
    * `:session` - Any term identifying a conversation. Requests with the same
      session are routed to the same slot whenever it is free, keeping their
      cached prefix intact under concurrency.
    * Sampling options (`:temp`, `:top_k`, `:top_p`, `:min_p`, `:seed`,
      `:penalty_repeat`, `:penalty_freq`, `:penalty_present`, `:grammar`,
      `:grammar_root`) - override the server-level defaults for this request.

  """
  @spec generate(GenServer.server(), String.t(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def generate(server, prompt, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 60_000)
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    req_opts = Keyword.take(opts, @request_opt_keys)

    # Tokenize in the caller: parallel across clients and off the server's
    # mailbox. The model handle comes from a :persistent_term cache.
    {:ok, token_ids} = Tokenizer.encode(get_model(server), prompt)
    GenServer.call(server, {:generate_tokens, token_ids, max_tokens, req_opts}, timeout)
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
    req_opts = Keyword.take(opts, @request_opt_keys)

    Stream.resource(
      fn ->
        {:ok, token_ids} = Tokenizer.encode(get_model(server), prompt)
        ref = make_ref()

        :ok =
          GenServer.call(server, {:stream_tokens, token_ids, max_tokens, self(), ref, req_opts})

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
    req_opts = Keyword.take(opts, @request_opt_keys)
    GenServer.call(server, {:generate_tokens, token_ids, max_tokens, req_opts}, timeout)
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
    req_opts = Keyword.take(opts, @request_opt_keys)

    Stream.resource(
      fn ->
        ref = make_ref()

        :ok =
          GenServer.call(server, {:stream_tokens, token_ids, max_tokens, self(), ref, req_opts})

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
  operations like tokenization. Served from a `:persistent_term` cache — no
  round-trip through the server's mailbox.
  """
  @spec get_model(GenServer.server()) :: Model.t()
  def get_model(server) do
    with pid when is_pid(pid) <- GenServer.whereis(server),
         %Model{} = model <- :persistent_term.get({__MODULE__, pid}, nil) do
      model
    else
      _ -> GenServer.call(server, :get_model)
    end
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
    n_batch = Keyword.get(opts, :n_batch, min(n_ctx, 2048))
    chunk_size = Keyword.get(opts, :chunk_size, 512)
    cache_prompt = Keyword.get(opts, :cache_prompt, true)
    kv_unified = Keyword.get(opts, :kv_unified, true)
    prompt_cache_ram_mb = Keyword.get(opts, :prompt_cache_ram_mb, 0)
    batch_strategy = Keyword.get(opts, :batch_strategy, LlamaCppEx.Server.Strategy.DecodeMaximal)

    sampler_opts = Keyword.take(opts, @sampler_opt_keys)

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

    # Trap exits so terminate/2 reliably erases the persistent_term model
    # cache on shutdown.
    Process.flag(:trap_exit, true)

    :ok = LlamaCppEx.init()
    {:ok, model} = Model.load(model_path, [n_gpu_layers: n_gpu_layers] ++ model_opts)

    # Cache the model handle for callers (get_model/1): tokenization and chat
    # templating happen client-side without a GenServer.call round-trip.
    :persistent_term.put({__MODULE__, self()}, model)

    {:ok, ctx} =
      Context.create(
        model,
        [n_ctx: n_ctx, n_batch: n_batch, n_seq_max: n_parallel, kv_unified: kv_unified] ++
          context_opts
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

    now = System.monotonic_time()

    slots =
      for seq_id <- 0..(n_parallel - 1), into: %{} do
        {:ok, sampler} = Sampler.create(model, sampler_opts)

        slot =
          idle_slot_fields([], 0)
          |> Map.put(:sampler, sampler)
          |> Map.put(:t_last_used, now)
          |> Map.put(:session, nil)

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
      cross_slot_sharing: kv_unified and seq_rm_kind == :part,
      prompt_cache_ram_mb: prompt_cache_ram_mb,
      seq_rm_kind: seq_rm_kind,
      batch_strategy: batch_strategy
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:generate_tokens, token_ids, max_tokens, req_opts}, from, state) do
    if token_ids == [] do
      {:reply, {:error, "token list cannot be empty"}, state}
    else
      case acquire_slot(state, token_ids, req_opts) do
        {:ok, seq_id, state} ->
          state = init_slot(state, seq_id, token_ids, max_tokens, from, nil, nil, req_opts)
          state = maybe_schedule_tick(state)
          {:noreply, state}

        :no_slots ->
          state =
            enqueue_request(state, {:generate, token_ids, max_tokens, from, nil, nil, req_opts})

          {:noreply, state}
      end
    end
  end

  def handle_call({:stream_tokens, token_ids, max_tokens, pid, ref, req_opts}, from, state) do
    case acquire_slot(state, token_ids, req_opts) do
      {:ok, seq_id, state} ->
        state = init_slot(state, seq_id, token_ids, max_tokens, nil, pid, ref, req_opts)
        GenServer.reply(from, :ok)
        state = maybe_schedule_tick(state)
        {:noreply, state}

      :no_slots ->
        GenServer.reply(from, :ok)

        state =
          enqueue_request(state, {:stream, token_ids, max_tokens, nil, pid, ref, req_opts})

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
      n_batch: state.n_batch,
      ram_cache_entries: length(state.ram_cache),
      ram_cache_bytes: state.ram_cache_bytes
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_info(:tick, state) do
    state = %{state | tick_scheduled: false}
    state = run_tick(state)
    {:noreply, state}
  end

  def handle_info({:EXIT, _pid, reason}, state) do
    # trap_exit is on (for terminate/2 cleanup) — honor exit signals.
    {:stop, reason, state}
  end

  @impl true
  def terminate(_reason, _state) do
    :persistent_term.erase({__MODULE__, self()})
    :ok
  end

  # --- Internal: Slot management ---

  defp request_cache_prompt?(state, req_opts) do
    Keyword.get(req_opts, :cache_prompt, state.cache_prompt)
  end

  defp acquire_slot(state, tokens, req_opts) do
    idle_slots = Enum.filter(state.slots, fn {_id, slot} -> slot.state == :idle end)

    case idle_slots do
      [] ->
        :no_slots

      slots ->
        session_pick = session_slot_if_idle(state, Keyword.get(req_opts, :session), slots)

        cond do
          session_pick != nil ->
            {:ok, session_pick, state}

          tokens != [] and request_cache_prompt?(state, req_opts) ->
            {:ok, pick_cached_slot(slots, tokens), state}

          true ->
            {:ok, pick_lru_slot(slots), state}
        end
    end
  end

  # Session affinity: the slot that served this session last, if it is idle.
  # Overrides the similarity rule — the session's cache lives there.
  defp session_slot_if_idle(_state, nil, _idle_slots), do: nil

  defp session_slot_if_idle(state, session, idle_slots) do
    with seq_id when seq_id != nil <- Map.get(state.sessions, session),
         true <- Enum.any?(idle_slots, fn {id, _} -> id == seq_id end) do
      seq_id
    else
      _ -> nil
    end
  end

  # llama-server's slot-pick rule (server-context.cpp): reuse the slot with the
  # best cached-prefix similarity only when it clears a threshold
  # (LCP/prompt_len > 0.1); otherwise take the least-recently-used idle slot so
  # a tiny unrelated request doesn't evict a valuable long cache.
  defp pick_cached_slot(idle_slots, tokens) do
    prompt_len = length(tokens)

    {best_id, best_lcp} =
      idle_slots
      |> Enum.map(fn {id, slot} -> {id, common_prefix_length(tokens, slot.cached_tokens)} end)
      |> Enum.max_by(fn {_id, lcp} -> lcp end)

    if best_lcp / prompt_len > 0.1 do
      best_id
    else
      pick_lru_slot(idle_slots)
    end
  end

  defp pick_lru_slot(idle_slots) do
    {seq_id, _} = Enum.min_by(idle_slots, fn {_id, slot} -> slot.t_last_used end)
    seq_id
  end

  defp init_slot(state, seq_id, tokens, max_tokens, from, stream_pid, stream_ref, req_opts) do
    state = update_session_mapping(state, seq_id, Keyword.get(req_opts, :session))
    slot = state.slots[seq_id]
    cache_prompt? = request_cache_prompt?(state, req_opts)

    # Prefix cache: find common prefix with cached KV. On models that only
    # support whole-sequence seq_rm (`:full`, e.g. hybrid GDN), a partial
    # trim would silently fail and leave stale KV past `n_match`, producing
    # an M-RoPE position-mismatch abort on the next decode. Disable the
    # cache hit in that case unless the new prompt extends the old one
    # exactly (no trim needed).
    #
    # A cached match may never cover the WHOLE prompt: at least the last
    # prompt token must be decoded to produce logits for sampling the first
    # generated token (llama-server does the same n_past-- adjustment). An
    # uncapped full match would enter the tick with nothing to prefill and no
    # logits to sample — a stuck slot.
    max_reuse = length(tokens) - 1

    raw_match =
      if cache_prompt? do
        min(common_prefix_length(tokens, slot.cached_tokens), max_reuse)
      else
        0
      end

    needs_trim = raw_match > 0 and raw_match < slot.cached_pos

    own_match =
      if needs_trim and state.seq_rm_kind == :full do
        0
      else
        raw_match
      end

    state = maybe_warn_prefix_instability(state, seq_id, slot, raw_match, cache_prompt?)

    {state, n_match} = resolve_prefix_cache(state, seq_id, tokens, own_match, cache_prompt?)
    slot = state.slots[seq_id]

    # Fresh sampler per request: request opts override server defaults, and a
    # new chain means clean grammar/penalty state and a fresh seed. The old
    # sampler resource is dropped and freed by GC.
    sampler_opts = Keyword.merge(state.sampler_opts, Keyword.take(req_opts, @sampler_opt_keys))
    {:ok, sampler} = Sampler.create(state.model, sampler_opts)

    slot = %{
      slot
      | state: :prefilling,
        from: from,
        stream_pid: stream_pid,
        stream_ref: stream_ref,
        sampler: sampler,
        cache_prompt: cache_prompt?,
        prompt_tokens: tokens,
        prompt_tokens_tuple: List.to_tuple(tokens),
        prefill_pos: n_match,
        pos: n_match,
        pending_token: nil,
        pending_eog: false,
        batch_idx: -1,
        tokens_generated: 0,
        max_tokens: max_tokens,
        accumulated_pieces: [],
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

  # Decides where this request's cached prefix comes from and prepares the
  # slot's KV accordingly. The three sources compete by match length, with
  # ties broken by cost: the slot's own cache (free) beats a donor slot
  # (metadata-only seq_cp under unified KV) beats the RAM prompt cache
  # (KV-sized memcpy). Whenever the slot's own cache is about to be destroyed
  # or truncated, it is offered to the RAM cache first.
  defp resolve_prefix_cache(state, seq_id, tokens, own_match, cache_prompt?) do
    slot = state.slots[seq_id]

    donor = best_donor(state, seq_id, tokens, own_match, cache_prompt?)
    donor_lcp = if donor, do: elem(donor, 1), else: 0

    ram = if cache_prompt?, do: best_ram_candidate(state, tokens), else: nil
    ram_lcp = if ram, do: elem(ram, 1), else: 0

    cond do
      donor_lcp > own_match and donor_lcp >= ram_lcp ->
        adopt_donor_cache(state, seq_id, slot, donor)

      ram_lcp > own_match ->
        adopt_ram_cache(state, seq_id, slot, ram)

      own_match > 0 ->
        keep_own_cache(state, seq_id, slot, own_match)

      true ->
        # No usable match anywhere — clear the slot.
        state = maybe_save_to_ram_cache(state, seq_id, slot)
        _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
        {state, 0}
    end
  end

  defp adopt_donor_cache(state, seq_id, slot, {donor_id, donor_lcp}) do
    state = maybe_save_to_ram_cache(state, seq_id, slot)
    _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
    :ok = LlamaCppEx.NIF.memory_seq_cp(state.ctx.ref, donor_id, seq_id, 0, donor_lcp)
    {state, donor_lcp}
  end

  defp adopt_ram_cache(state, seq_id, slot, {entry, ram_lcp}) do
    state = maybe_save_to_ram_cache(state, seq_id, slot)
    _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
    {state, apply_ram_restore(state, seq_id, entry, ram_lcp)}
  end

  defp keep_own_cache(state, seq_id, slot, own_match) do
    if own_match < slot.cached_pos do
      # Trim KV cache beyond the matched prefix (only safe on `:part`/`:rs`).
      # The truncated tail may still be valuable to another conversation —
      # offer the full state to the RAM cache before cutting it.
      state = maybe_save_to_ram_cache(state, seq_id, slot)
      true = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, own_match, -1)
      {state, own_match}
    else
      # Exact-prefix continuation; nothing to trim.
      {state, own_match}
    end
  end

  # A cache-eligible request whose prompt shares only a partial prefix with
  # the slot's cached history usually means the chat template rewrote earlier
  # turns (e.g. Qwen thinking-strip re-rendering) — every such request pays a
  # near-full re-prefill that the caller probably believes is cached. Emit
  # telemetry per occurrence and warn in the log once per server.
  defp maybe_warn_prefix_instability(state, seq_id, slot, raw_match, cache_prompt?) do
    suspicious? =
      cache_prompt? and slot.cached_pos >= @prefix_instability_min_cached and
        raw_match > slot.cached_pos * 0.1 and raw_match < slot.cached_pos * 0.5

    if suspicious? do
      :telemetry.execute(
        [:llama_cpp_ex, :server, :prefix_instability],
        %{matched_tokens: raw_match, cached_tokens: slot.cached_pos},
        %{server: self(), seq_id: seq_id}
      )

      warn_prefix_instability_once(state, raw_match, slot.cached_pos)
    else
      state
    end
  end

  defp warn_prefix_instability_once(%{prefix_instability_warned: true} = state, _match, _cached),
    do: state

  defp warn_prefix_instability_once(state, raw_match, cached_pos) do
    Logger.warning(
      "LlamaCppEx.Server: request matched only #{raw_match}/#{cached_pos} cached prompt " <>
        "tokens — the chat template may be rewriting history (e.g. stripping thinking " <>
        "blocks), which defeats prefix caching. Emitted per-request as " <>
        "[:llama_cpp_ex, :server, :prefix_instability]; this warning logs once."
    )

    %{state | prefix_instability_warned: true}
  end

  # Offers a slot's about-to-be-destroyed cache to the RAM prompt cache.
  # Skipped when the feature is off, the cache is too small to be worth a
  # KV-sized copy, an existing entry already covers it, or the blob exceeds
  # the whole budget (degrade to disabled, never OOM).
  defp maybe_save_to_ram_cache(%{prompt_cache_ram_mb: 0} = state, _seq_id, _slot), do: state

  defp maybe_save_to_ram_cache(state, seq_id, slot) do
    budget = state.prompt_cache_ram_mb * 1024 * 1024

    with true <- slot.cached_pos >= @ram_cache_min_tokens,
         false <- ram_cache_covers?(state.ram_cache, slot.cached_tokens, slot.cached_pos),
         bytes = LlamaCppEx.NIF.state_seq_get_size(state.ctx.ref, seq_id),
         true <- bytes > 0 and bytes <= budget,
         {:ok, bin} <- LlamaCppEx.NIF.state_seq_get_data(state.ctx.ref, seq_id) do
      entry = %{tokens: slot.cached_tokens, len: slot.cached_pos, bin: bin, bytes: bytes}

      state = %{
        state
        | ram_cache: state.ram_cache ++ [entry],
          ram_cache_bytes: state.ram_cache_bytes + bytes
      }

      state = evict_ram_cache_to_budget(state, budget)
      emit_ram_cache_telemetry(state, :save, entry)
      state
    else
      _ -> state
    end
  end

  defp ram_cache_covers?(entries, cached_tokens, cached_pos) do
    Enum.any?(entries, fn entry ->
      entry.len >= cached_pos and
        common_prefix_length(cached_tokens, entry.tokens) == cached_pos
    end)
  end

  # FIFO eviction: drop oldest entries until the cache fits the budget.
  defp evict_ram_cache_to_budget(state, budget) do
    if state.ram_cache_bytes <= budget do
      state
    else
      [evicted | rest] = state.ram_cache
      state = %{state | ram_cache: rest, ram_cache_bytes: state.ram_cache_bytes - evicted.bytes}
      emit_ram_cache_telemetry(state, :evict, evicted)
      evict_ram_cache_to_budget(state, budget)
    end
  end

  # Best RAM cache entry for this prompt, if one clears the f_keep bar
  # (restoring is a KV-sized memcpy — not worth it for a sliver) and its
  # unusable tail can actually be trimmed on this model.
  defp best_ram_candidate(%{ram_cache: []}, _tokens), do: nil

  defp best_ram_candidate(state, tokens) do
    # Cap at len-1: the last prompt token must be decoded for logits.
    max_reuse = length(tokens) - 1

    {entry, lcp} =
      state.ram_cache
      |> Enum.map(fn entry ->
        {entry, min(common_prefix_length(tokens, entry.tokens), max_reuse)}
      end)
      |> Enum.max_by(fn {_entry, lcp} -> lcp end)

    usable? =
      lcp > 0 and lcp / entry.len >= @ram_cache_min_keep and
        (lcp == entry.len or state.seq_rm_kind != :full)

    if usable?, do: {entry, lcp}
  end

  # Restores a RAM cache entry into an (empty) sequence and trims the unusable
  # tail. Returns the number of reusable prefix tokens.
  defp apply_ram_restore(state, seq_id, entry, lcp) do
    case LlamaCppEx.NIF.state_seq_set_data(state.ctx.ref, entry.bin, seq_id) do
      {:ok, _bytes} ->
        if lcp < entry.len do
          true = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, lcp, -1)
        end

        emit_ram_cache_telemetry(state, :restore, entry)
        lcp

      {:error, reason} ->
        # A partial restore could leave garbage in the sequence — clear it.
        Logger.warning("LlamaCppEx.Server: RAM cache restore failed: #{inspect(reason)}")
        _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
        0
    end
  end

  defp emit_ram_cache_telemetry(state, op, entry) do
    :telemetry.execute(
      [:llama_cpp_ex, :server, :ram_cache],
      %{
        bytes: entry.bytes,
        tokens: entry.len,
        total_bytes: state.ram_cache_bytes,
        entries: length(state.ram_cache)
      },
      %{server: self(), op: op}
    )
  end

  # Finds the slot whose in-KV tokens share the longest prefix with the new
  # prompt, when that beats the assigned slot's own match. Active donors count
  # too — only their FED tokens are in the KV, so the match is capped at the
  # fed length. The pos_max probe guards against any bookkeeping drift between
  # slot state and the actual KV contents.
  defp best_donor(state, dst_seq_id, tokens, own_match, cache_prompt?) do
    if cache_prompt? and state.cross_slot_sharing do
      # Cap at len-1: the last prompt token must be decoded for logits.
      max_reuse = length(tokens) - 1

      state.slots
      |> Enum.reject(fn {id, _slot} -> id == dst_seq_id end)
      |> Enum.map(fn {id, slot} -> {id, min(donor_prefix_match(slot, tokens), max_reuse)} end)
      |> Enum.max_by(fn {_id, lcp} -> lcp end, fn -> {nil, 0} end)
      |> validate_donor(state, own_match)
    else
      nil
    end
  end

  defp validate_donor({donor_id, lcp}, state, own_match) when lcp > own_match and lcp > 0 do
    if LlamaCppEx.NIF.memory_seq_pos_max(state.ctx.ref, donor_id) + 1 >= lcp do
      {donor_id, lcp}
    end
  end

  defp validate_donor(_candidate, _state, _own_match), do: nil

  defp donor_prefix_match(%{state: :idle} = slot, tokens) do
    common_prefix_length(tokens, slot.cached_tokens)
  end

  defp donor_prefix_match(%{state: :prefilling} = slot, tokens) do
    # Only positions 0..prefill_pos-1 are in the KV so far.
    min(common_prefix_length(tokens, slot.prompt_tokens), slot.prefill_pos)
  end

  defp donor_prefix_match(%{state: :generating} = slot, tokens) do
    fed = slot.prompt_tokens ++ Enum.reverse(slot.generated_token_ids)
    min(common_prefix_length(tokens, fed), slot.pos)
  end

  defp slot_mode(%{stream_pid: pid}) when is_pid(pid), do: :stream
  defp slot_mode(_slot), do: :generate

  # Keeps sessions ↔ slots consistent: the slot remembers its session (so an
  # idle slot can be re-picked by affinity) and the reverse map points each
  # session at the slot serving it. A slot taken over by a different session
  # drops its old mapping — but only if that mapping still points here (the
  # old session may already have moved to another slot).
  defp update_session_mapping(state, seq_id, session) do
    old_session = state.slots[seq_id].session

    sessions =
      if old_session != nil and old_session != session and
           Map.get(state.sessions, old_session) == seq_id do
        Map.delete(state.sessions, old_session)
      else
        state.sessions
      end

    sessions = if session != nil, do: Map.put(sessions, session, seq_id), else: sessions

    state = put_in(state.slots[seq_id].session, session)
    %{state | sessions: sessions}
  end

  defp enqueue_request(state, request) do
    %{state | queue: :queue.in(request, state.queue)}
  end

  # Freed slots serve queued requests session-first, then FIFO — otherwise a
  # busy queue scatters a conversation across slots and shreds its cache.
  defp dequeue_into_slot(state) do
    state
    |> dequeue_session_matches()
    |> dequeue_fifo()
  end

  defp dequeue_session_matches(state) do
    requests = :queue.to_list(state.queue)

    {state, remaining} =
      Enum.reduce(requests, {state, []}, fn request, {state, rest} ->
        opts = request_opts(request)

        case session_slot_if_idle(state, Keyword.get(opts, :session), idle_slots(state)) do
          nil ->
            {state, [request | rest]}

          seq_id ->
            {assign_queued_request(state, seq_id, request), rest}
        end
      end)

    %{state | queue: :queue.from_list(Enum.reverse(remaining))}
  end

  defp idle_slots(state) do
    Enum.filter(state.slots, fn {_id, slot} -> slot.state == :idle end)
  end

  defp dequeue_fifo(state) do
    case :queue.out(state.queue) do
      {{:value, request}, queue} ->
        state = %{state | queue: queue}
        tokens = request_tokens(request)

        case acquire_slot(state, tokens, request_opts(request)) do
          {:ok, seq_id, state} ->
            state = assign_queued_request(state, seq_id, request)
            dequeue_fifo(state)

          :no_slots ->
            # Put it back
            %{state | queue: :queue.in_r(request, state.queue)}
        end

      {:empty, _queue} ->
        state
    end
  end

  defp request_tokens({_type, tokens, _max, _from, _pid, _ref, _req_opts}), do: tokens

  defp request_opts({_type, _tokens, _max, _from, _pid, _ref, req_opts}), do: req_opts

  defp assign_queued_request(state, seq_id, {:generate, tokens, max_tokens, from, _, _, req_opts}) do
    init_slot(state, seq_id, tokens, max_tokens, from, nil, nil, req_opts)
  end

  defp assign_queued_request(state, seq_id, {:stream, tokens, max_tokens, _, pid, ref, req_opts}) do
    init_slot(state, seq_id, tokens, max_tokens, nil, pid, ref, req_opts)
  end

  # --- Internal: Tick loop ---

  defp run_tick(state) do
    # Phase 1: Finish completed slots
    state = finish_completed_slots(state)

    # Phase 1b: Dequeue waiting requests into freed slots
    state = dequeue_into_slot(state)

    # Phase 2: Build batch
    {entries, state} = build_batch(state)

    if entries == [] do
      state
    else
      run_forward_pass(state, entries)
    end
  end

  # Phases 3-5: fused forward pass + sampling, result handling, telemetry,
  # and continuation. One dirty-CPU NIF per tick does decode, per-slot
  # sampling, detokenization, and EOG checks; streaming sends happen here,
  # right after the NIF returns — one tick earlier than the previous design,
  # which deferred them to the next tick's batch building.
  defp run_forward_pass(state, entries) do
    # Count decode tokens before result handling (slots may transition after)
    n_decode =
      Enum.count(state.slots, fn {_id, s} ->
        s.state == :generating and s.batch_idx >= 0
      end)

    samplers = active_samplers(state)
    purgeable = purgeable_seq_ids(state)

    # Phase 3: Fused forward pass + sample
    tick_start = System.monotonic_time()

    case LlamaCppEx.NIF.batch_eval_sample(state.ctx.ref, entries, samplers, purgeable) do
      {:ok, results, purged, n_splits, failed} ->
        tick_end = System.monotonic_time()

        # Phase 4: Apply sampled results, stream pieces, clear tick markers
        state = handle_kv_pressure(state, purged, n_splits)
        state = apply_sample_results(state, results)
        state = fail_overflowed_slots(state, failed)
        state = clear_batch_indices(state)

        emit_tick_telemetry(state, entries, n_decode, tick_end - tick_start)

        # Phase 5: Continue
        continue_if_active(state)

      {:error, reason} ->
        Logger.error("batch forward pass failed: #{reason}")
        fail_all_active_slots(state, reason)
    end
  end

  # Sequences that could not fit a single further token even after purging and
  # batch splitting — their KV budget is exhausted. Fail just those requests
  # with a clean error; other slots keep generating.
  defp fail_overflowed_slots(state, []), do: state

  defp fail_overflowed_slots(state, failed) do
    Enum.reduce(failed, state, fn seq_id, state ->
      Logger.warning("LlamaCppEx.Server: slot #{seq_id} out of context — failing request")
      fail_slot(state, seq_id, :context_full)
    end)
  end

  defp fail_slot(state, seq_id, reason) do
    slot = state.slots[seq_id]

    if slot.from do
      GenServer.reply(slot.from, {:error, reason})
    end

    if slot.stream_pid && slot.stream_ref do
      send(slot.stream_pid, {slot.stream_ref, {:error, reason}})
    end

    emit_request_exception(slot, seq_id, reason)

    # Clear KV cache — don't preserve cache on error
    LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)

    slot =
      slot
      |> Map.merge(idle_slot_fields([], 0))
      |> Map.put(:t_last_used, System.monotonic_time())

    put_in(state.slots[seq_id], slot)
  end

  # Samplers for all active slots, keyed by seq_id — the NIF samples every
  # logits-requesting entry whose seq_id is registered here.
  defp active_samplers(state) do
    for {seq_id, slot} <- state.slots, slot.state != :idle do
      {seq_id, slot.sampler.ref}
    end
  end

  # Idle slots whose retained prefix-cache KV may be dropped by the NIF under
  # KV pressure (llama_decode == 1). Active slots are never purgeable.
  defp purgeable_seq_ids(state) do
    for {seq_id, slot} <- state.slots, slot.state == :idle, slot.cached_pos > 0 do
      seq_id
    end
  end

  # The NIF purged these idle slots' KV to relieve pressure — drop the
  # corresponding prefix-cache bookkeeping and surface telemetry.
  defp handle_kv_pressure(state, [], 0), do: state

  defp handle_kv_pressure(state, purged, n_splits) do
    Logger.warning(
      "LlamaCppEx.Server: KV pressure — purged #{length(purged)} idle slot cache(s), " <>
        "#{n_splits} batch split(s)"
    )

    :telemetry.execute(
      [:llama_cpp_ex, :server, :kv_pressure],
      %{purged_slots: length(purged), batch_splits: n_splits},
      %{server: self(), purged_seq_ids: purged}
    )

    Enum.reduce(purged, state, fn seq_id, state ->
      slot = state.slots[seq_id]
      put_in(state.slots[seq_id], %{slot | cached_tokens: [], cached_pos: 0})
    end)
  end

  # Phase 4: apply per-slot sampled tokens from the fused NIF. Each result is
  # {seq_id, token, piece, is_eog} for a slot that had a logits-requesting
  # entry this tick — either a generating slot's decode token or a slot whose
  # prefill just completed. Non-EOG pieces are streamed/accumulated
  # immediately; the token itself is fed to the KV on the NEXT tick (unless
  # the slot finishes first — see finish_completed_slots/1).
  defp apply_sample_results(state, results) do
    now = System.monotonic_time()

    Enum.reduce(results, state, fn {seq_id, token, piece, is_eog}, state ->
      slot = state.slots[seq_id]

      slot =
        case slot.state do
          :prefilling -> %{slot | state: :generating, pos: slot.n_prompt_tokens}
          :generating -> %{slot | pos: slot.pos + 1}
        end

      slot = %{slot | pending_token: token, pending_eog: is_eog}
      slot = if is_eog, do: slot, else: emit_piece(slot, piece, now)

      put_in(state.slots[seq_id], slot)
    end)
  end

  # Streams a sampled piece to the slot's subscriber (if any) and folds it
  # into the slot's accumulated output/counters.
  defp emit_piece(slot, piece, now) do
    if slot.stream_pid && slot.stream_ref do
      send(slot.stream_pid, {slot.stream_ref, {:token, piece}})
    end

    %{
      slot
      | accumulated_pieces: [piece | slot.accumulated_pieces],
        tokens_generated: slot.tokens_generated + 1,
        t_first_token: slot.t_first_token || now
    }
  end

  # batch_idx is only meaningful within a single tick (it marks slots the
  # strategy fed this batch). Clear it so the next tick's bookkeeping starts
  # clean even for slots that were skipped by budget limits.
  defp clear_batch_indices(state) do
    slots =
      Map.new(state.slots, fn
        {id, %{batch_idx: -1} = slot} -> {id, slot}
        {id, slot} -> {id, %{slot | batch_idx: -1}}
      end)

    %{state | slots: slots}
  end

  # Schedules the next tick while any slot is still active.
  defp continue_if_active(state) do
    if Enum.any?(state.slots, fn {_id, slot} -> slot.state != :idle end) do
      maybe_schedule_tick(state)
    else
      state
    end
  end

  defp emit_tick_telemetry(state, entries, n_decode, eval_native) do
    :telemetry.execute(
      [:llama_cpp_ex, :server, :tick],
      %{
        batch_size: length(entries),
        decode_tokens: n_decode,
        prefill_tokens: length(entries) - n_decode,
        active_slots: Enum.count(state.slots, fn {_id, s} -> s.state != :idle end),
        queue_depth: :queue.len(state.queue),
        eval_ms: eval_native / 1_000_000
      },
      %{server: self()}
    )
  end

  # Phase 1: Check generating slots for completion. EOG was determined by the
  # fused NIF at sample time (pending_eog) — no per-token NIF call here. A slot
  # whose pending token is EOG or whose streamed output hit max_tokens finishes
  # without feeding that pending token to the KV.
  defp finish_completed_slots(state) do
    generating_slots =
      state.slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :generating and slot.pending_token != nil
      end)

    Enum.reduce(generating_slots, state, fn {seq_id, _slot}, state ->
      slot = state.slots[seq_id]

      cond do
        slot.pending_eog ->
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

  # --- Internal: Slot completion ---

  defp finish_slot(state, seq_id, stop_reason) do
    slot = state.slots[seq_id]
    t_end = System.monotonic_time()

    if slot.from do
      GenServer.reply(slot.from, {:ok, accumulated_text(slot)})
    end

    if slot.stream_pid && slot.stream_ref do
      send(slot.stream_pid, {slot.stream_ref, :done})
    end

    # Emit telemetry
    emit_request_done(slot, seq_id, t_end, stop_reason)

    reset_slot(state, seq_id)
  end

  defp emit_request_done(slot, seq_id, t_end, stop_reason) do
    m = request_measurements(slot, t_end)

    Logger.debug(
      "slot #{seq_id} done (#{stop_reason}): #{slot.n_prompt_tokens} prompt tokens (#{Float.round(m.prompt_eval_rate, 1)} t/s), " <>
        "#{slot.tokens_generated} generated (#{Float.round(m.generation_rate, 1)} t/s), " <>
        "ttft #{Float.round(m.ttft_ms, 1)}ms, total #{Float.round(m.duration_ms, 1)}ms"
    )

    :telemetry.execute(
      [:llama_cpp_ex, :server, :request, :done],
      m,
      %{server: self(), seq_id: seq_id, mode: slot_mode(slot), stop_reason: stop_reason}
    )
  end

  defp emit_request_exception(slot, seq_id, reason) do
    # Mirrors `:done`'s measurement shape so dashboards can aggregate them.
    :telemetry.execute(
      [:llama_cpp_ex, :server, :request, :exception],
      request_measurements(slot, System.monotonic_time()),
      %{
        server: self(),
        seq_id: seq_id,
        mode: slot_mode(slot),
        stop_reason: :error,
        reason: reason
      }
    )
  end

  # Shared measurements for the :done / :exception request events. Timings are
  # best-effort: if generation never started, ttft/duration fall back to the
  # wall time since slot acquisition.
  defp request_measurements(slot, t_end) do
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

    %{
      prompt_tokens: slot.n_prompt_tokens,
      generated_tokens: slot.tokens_generated,
      duration_ms: duration_ms,
      ttft_ms: ttft_ms,
      prompt_eval_rate: prompt_eval_rate,
      generation_rate: generation_rate,
      prefix_cache_tokens: slot.n_prefix_cache_tokens,
      prefix_cache_ratio: prefix_cache_ratio
    }
  end

  defp reset_slot(state, seq_id) do
    slot = state.slots[seq_id]

    # Build full token history for prefix cache. Retention follows the
    # per-request setting: a cache_prompt: false request leaves nothing behind.
    # (No sampler reset — every request gets a fresh sampler at init_slot.)
    {cached_tokens, cached_pos} =
      if slot.cache_prompt do
        all_tokens = slot.prompt_tokens ++ Enum.reverse(slot.generated_token_ids)
        {all_tokens, slot.pos}
      else
        LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
        {[], 0}
      end

    slot =
      slot
      |> Map.merge(idle_slot_fields(cached_tokens, cached_pos))
      |> Map.put(:t_last_used, System.monotonic_time())

    put_in(state.slots[seq_id], slot)
  end

  # The single source of truth for a slot's per-request fields. init/1,
  # reset_slot/2, and the failure paths all build from this map, so a new
  # slot field cannot silently carry stale data across requests. The
  # prefix-cache carry-over is the only caller-controlled part; :sampler,
  # :t_last_used, and :session are the only fields that live outside it
  # (slot metadata that must survive request resets).
  defp idle_slot_fields(cached_tokens, cached_pos) do
    %{
      state: :idle,
      from: nil,
      stream_pid: nil,
      stream_ref: nil,
      cache_prompt: false,
      prompt_tokens: [],
      prompt_tokens_tuple: {},
      prefill_pos: 0,
      pos: 0,
      pending_token: nil,
      pending_eog: false,
      batch_idx: -1,
      tokens_generated: 0,
      max_tokens: 0,
      accumulated_pieces: [],
      t_start: nil,
      t_first_token: nil,
      n_prompt_tokens: 0,
      cached_tokens: cached_tokens,
      cached_pos: cached_pos,
      generated_token_ids: [],
      n_prefix_cache_tokens: 0
    }
  end

  # Builds the final completion string from the reverse-ordered piece list.
  defp accumulated_text(slot) do
    slot.accumulated_pieces |> Enum.reverse() |> IO.iodata_to_binary()
  end

  defp fail_all_active_slots(state, reason) do
    active_slots =
      Enum.filter(state.slots, fn {_id, slot} -> slot.state != :idle end)

    Enum.reduce(active_slots, state, fn {seq_id, _slot}, state ->
      fail_slot(state, seq_id, "inference failed: #{reason}")
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
  # Single-pass count of the shared prefix length — no intermediate zip list.
  def common_prefix_length(a, b), do: common_prefix_length(a, b, 0)

  defp common_prefix_length([x | a], [x | b], n), do: common_prefix_length(a, b, n + 1)
  defp common_prefix_length(_, _, n), do: n
end
