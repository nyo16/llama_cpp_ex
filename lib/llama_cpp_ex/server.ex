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
    * `:stop_reason` - `:eog` (end-of-generation token sampled),
      `:max_tokens` (request `max_tokens` reached), or `:cancelled` (consumer
      died or cancelled the request).

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

  alias LlamaCppEx.{Context, Model, Options, Sampler, Tokenizer, UTF8Stream}
  alias LlamaCppEx.Server.{PromptCache, Request, Slots}

  # Sampling options accepted both at server start (defaults) and per request
  # (overrides). Owned by Sampler, so this cannot drift out of sync; the
  # compile-time dependency recompiles this module when that list changes.
  @sampler_opt_keys Sampler.option_keys()

  # Per-request options accepted by generate/stream and carried through the
  # queue into the slot. Server-level options provide the defaults.
  @request_opt_keys [:cache_prompt, :session, :cache_scope] ++ @sampler_opt_keys

  # The complete accepted key set for the request-level public functions.
  @call_opt_keys Enum.uniq([:max_tokens, :timeout] ++ @request_opt_keys)

  @doc """
  The per-request options this server accepts, beyond `:max_tokens`/`:timeout`.

  Follows the same ownership rule as `LlamaCppEx.Sampler.option_keys/0`: callers
  that forward user options into a server — `LlamaCppEx.chat_completion/3` and
  `LlamaCppEx.stream_chat_completion/3` — select them with this function instead
  of keeping their own copy. Their copy had already drifted once, silently
  rejecting `:cache_scope`.
  """
  @spec request_option_keys() :: [atom()]
  def request_option_keys, do: @request_opt_keys

  # Options this module reads itself in init/1 and handle_continue/2, as opposed
  # to forwarding to Model/Context/Sampler.
  @own_start_opt_keys [
    :model_path,
    :n_gpu_layers,
    :n_parallel,
    :n_ctx,
    :n_batch,
    :chunk_size,
    :cache_prompt,
    :kv_unified,
    :prompt_cache_ram_mb,
    :max_queue,
    :batch_strategy
  ]

  # The complete accepted key set for start_link/1, assembled from the owning
  # modules so it cannot drift.
  @start_opt_keys Enum.uniq(
                    @own_start_opt_keys ++
                      @sampler_opt_keys ++
                      Model.tuning_option_keys() ++
                      Context.tuning_option_keys()
                  )

  @doc """
  The options `start_link/1` accepts.

  Exposed for the same reason as `request_option_keys/0`: callers that forward
  user options into a server must select them rather than guess. The one caller
  that guessed — `LlamaCppEx.ModelManager.ModelIO` — used a `Keyword.drop/2`
  denylist, so `:vocab_only` reached `init/1` and raised there.
  """
  @spec start_option_keys() :: [atom()]
  def start_option_keys, do: @start_opt_keys

  # A zero queue meant the reject branch was dead code and the documented
  # `:queue_full` error could never fire, so the queue was unbounded — and each
  # entry holds a full token list. 64 is deep enough that a burst is absorbed
  # rather than rejected, and shallow enough to bound worst-case memory at
  # 64 prompts.
  @default_max_queue 64

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
    max_queue: @default_max_queue,
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
    prompt_cache: nil,
    prefix_instability_warned: false,
    # `:part` = any position range; `:full` = whole-sequence only (hybrid GDN
    # models like Qwen 3.5/3.6); `:rs` = partial, but only within the last
    # `n_rs_seq` positions — beyond that `llama_memory_recurrent::seq_rm` returns
    # `false` rather than raising, which every partial-trim call site must handle;
    # `:no` = no memory module. Partial prefix-cache trims are attempted on
    # `:part` and `:rs`, and a refusal falls back to a full clear.
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
    * `:max_queue` - Max queued requests waiting for a slot. When the bound is
      hit, calls return `{:error, :queue_full}` immediately and streams emit a
      single `{:error, :queue_full}` element — no silent queueing until the
      call timeout. `0` means unlimited, which is **not** the default: at `0`
      the reject branch is dead code, the documented `:queue_full` error can
      never fire, and each queued entry holds a full token list, so a burst is
      bounded only by memory. Defaults to `#{@default_max_queue}`.
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
    * Context tuning options are forwarded to `LlamaCppEx.Context.create/2` —
      `:n_threads`, `:n_threads_batch`, `:n_ubatch`, `:type_k`, `:type_v`,
      `:flash_attn`, `:offload_kqv`, `:op_offload`, the RoPE/YaRN options,
      `:attention_type`, `:no_perf` and `:swa_full`. See
      `LlamaCppEx.Context.tuning_option_keys/0` for the authoritative list.
      `:n_ctx`, `:n_batch`, `:n_seq_max` and `:kv_unified` are set by the server
      from the options above and cannot be overridden here.
    * Model loading options are forwarded to `LlamaCppEx.Model.load/2` —
      `:main_gpu`, `:split_mode`, `:tensor_split`, `:use_mmap`, `:use_mlock`,
      `:use_direct_io`, `:check_tensors` and `:rpc_servers`. The three load flags
      collapse into llama.cpp's single `load_mode`: `:use_direct_io` wins
      outright, otherwise `:use_mlock` and `:use_mmap` combine; see
      `LlamaCppEx.Model.load/2`. `:rpc_servers` registers remote endpoints before
      the load so their devices can hold part of the model — see
      `LlamaCppEx.RPC`, including the caveat that a peer failure aborts the VM.
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
    * `:timeout` - Call timeout in ms. Defaults to
      `#{LlamaCppEx.Options.blocking_timeout()}`.
    * `:cache_prompt` - Reuse/retain this request's KV prefix on the slot.
      Defaults to the server-level `:cache_prompt` setting.
    * `:session` - Any term identifying a conversation. Requests with the same
      session are routed to the same slot whenever it is free, keeping their
      cached prefix intact under concurrency.
    * `:cache_scope` - Trust boundary for KV prefix reuse. A request only reuses
      a cached prefix — from its own slot, from another slot, or from the RAM
      prompt cache — when the cached content was produced under the *same*
      scope. Defaults to `nil`, a single shared pool, which is only safe when
      every caller of this server is in one trust domain. In a multi-tenant
      deployment set it to the tenant id: prefix reuse is a KV read, so two
      tenants whose prompts share a system prompt would otherwise be able to
      inherit each other's cache. Unlike `:session`, this does not affect slot
      routing, only what may be reused.
    * Sampling options (`:temp`, `:top_k`, `:top_p`, `:min_p`, `:seed`,
      `:penalty_repeat`, `:penalty_freq`, `:penalty_present`, `:grammar`,
      `:grammar_root`) - override the server-level defaults for this request.

  """
  @spec generate(GenServer.server(), String.t(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def generate(server, prompt, opts \\ []) do
    opts = Options.validate!(opts, @call_opt_keys, "LlamaCppEx.Server.generate/3")
    timeout = Options.timeout(opts, :blocking)
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    req_opts = Keyword.take(opts, @request_opt_keys)

    # Tokenize in the caller: parallel across clients and off the server's
    # mailbox. The model handle comes from LlamaCppEx.Registry.
    with {:ok, model} <- fetch_model(server),
         :ok <- Sampler.validate_grammar(model, req_opts),
         {:ok, token_ids} <- Tokenizer.encode(model, prompt) do
      GenServer.call(server, {:generate_tokens, token_ids, max_tokens, req_opts}, timeout)
    end
  end

  @doc """
  Returns a stream of generated text chunks.

  If the request is rejected (`:queue_full`), fails mid-generation, or a chunk
  does not arrive within `:timeout`, the stream emits a single
  `{:error, reason}` element and halts — consumers that need to distinguish
  errors from text should match on it. A per-token timeout emits
  `{:error, :timeout}` and cancels the request server-side; it used to truncate
  the stream silently, which is indistinguishable from a completed generation.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:timeout` - Per-token timeout, and the budget for being admitted to a slot
      or the queue. Defaults to `#{LlamaCppEx.Options.stream_timeout()}`.

  Also accepts the per-request options documented on `generate/3`.
  """
  @spec stream(GenServer.server(), String.t(), keyword()) :: Enumerable.t()
  def stream(server, prompt, opts \\ []) do
    opts = Options.validate!(opts, @call_opt_keys, "LlamaCppEx.Server.stream/3")
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    timeout = Options.timeout(opts, :stream)
    req_opts = Keyword.take(opts, @request_opt_keys)

    Stream.resource(
      fn ->
        with {:ok, model} <- fetch_model(server),
             :ok <- Sampler.validate_grammar(model, req_opts),
             {:ok, token_ids} <- Tokenizer.encode(model, prompt) do
          start_token_stream(server, token_ids, max_tokens, req_opts, timeout)
        else
          {:error, reason} -> {:rejected, make_ref(), reason}
        end
      end,
      &stream_next/1,
      &stream_cleanup/1
    )
  end

  # Shared stream start: a rejected request becomes a single {:error, reason}
  # element rather than silence followed by a timeout.
  #
  # The admission call is bounded by the caller's `:timeout` rather than
  # `GenServer.call/2`'s implicit 5000 ms. On a busy batching server the old
  # default *exited* from inside a `Stream.resource` start-function — which no
  # consumer can catch and which bypasses the `{:error, reason}` element the
  # `:rejected` branch below exists to produce. A caller who asked for a 60 s
  # budget meant it for getting into the queue too.
  defp start_token_stream(server, token_ids, max_tokens, req_opts, timeout) do
    ref = make_ref()
    message = {:stream_tokens, token_ids, max_tokens, self(), ref, req_opts}

    case call_or_error(server, message, timeout) do
      :ok -> {server, ref, timeout}
      {:error, reason} -> {:rejected, ref, reason}
    end
  end

  # A `GenServer.call/3` that reports an exit as a value. Every caller runs inside
  # a `Stream.resource` start-function or a `with` chain, where an exit escapes
  # past every element the stream would otherwise have emitted.
  defp call_or_error(server, message, timeout) do
    GenServer.call(server, message, timeout)
  catch
    :exit, {:timeout, _} -> {:error, :timeout}
    :exit, {reason, {GenServer, :call, _}} -> {:error, call_exit_reason(reason)}
    :exit, reason -> {:error, call_exit_reason(reason)}
  end

  # Maps a `GenServer.call/3` exit reason onto an error value.
  #
  # The three shapes anyone thinks of are `:noproc`, `:normal` and `:timeout`, and
  # those were the three that were handled. `handle_continue/2`'s
  # `{:stop, {:load_failed, reason}, state}` exits with
  # `{{:load_failed, reason}, {GenServer, :call, _}}`, and a supervisor shutting
  # the server down produces `{:shutdown, _}` or `:killed` — all of which escaped
  # from inside both `Stream.resource` start-functions past a `@spec` that
  # promised a total function.
  #
  # The catch-all is deliberate. An uncatalogued reason must still become a value,
  # because no caller here has another way to observe it. `:timeout` is left out:
  # it means "still loading" to `fetch_model/1` and "the mailbox is saturated" to
  # the streaming admission call, so each maps it itself.
  defp call_exit_reason(:normal), do: :noproc
  defp call_exit_reason(:noproc), do: :noproc
  defp call_exit_reason(:killed), do: :noproc
  defp call_exit_reason(:shutdown), do: :noproc
  defp call_exit_reason({:shutdown, _}), do: :noproc
  defp call_exit_reason(other), do: other

  # Admission-time grammar check for the entry points that do not already hold a
  # `%Model{}`. `:grammar` is a per-request *value* and `Options.validate!/3`
  # only checks key names, so an uncompilable grammar used to travel all the way
  # into `init_slot/4` — inside the GenServer that owns the model, where it
  # crashed the server rather than the request. Rejecting here means the request
  # never reaches a slot and the caller gets `{:error, :invalid_grammar}`
  # synchronously.
  #
  # The model is fetched only when a grammar is actually present, so a request
  # without one pays nothing.
  defp validate_request_grammar(server, req_opts) do
    case Keyword.get(req_opts, :grammar, "") do
      grammar when grammar in [nil, ""] ->
        :ok

      _ ->
        with {:ok, model} <- fetch_model(server) do
          Sampler.validate_grammar(model, req_opts)
        end
    end
  end

  # Shared next/cleanup functions for the piece-streaming Stream.resources.
  # A rejected request ({:error, :queue_full}) surfaces as a single error
  # element instead of silence-then-timeout. Halting an ACTIVE stream early
  # (cleanup before :done) cancels the request server-side — otherwise an
  # abandoned stream would keep consuming batch budget to max_tokens.
  defp stream_next({:rejected, ref, reason}), do: {[{:error, reason}], {:done, ref}}
  defp stream_next({:done, _ref} = state), do: {:halt, state}
  defp stream_next({:timed_out, _server, _ref} = state), do: {:halt, state}

  defp stream_next({server, ref, timeout}) do
    receive do
      {^ref, {:token, text}} -> {[text], {server, ref, timeout}}
      {^ref, {:done, _reason}} -> {:halt, {:done, ref}}
      {^ref, {:error, reason}} -> {[{:error, reason}], {:done, ref}}
    after
      # A per-token timeout is an error, not the end of the text. Truncating
      # silently here contradicted this function's own `@doc` and left these two
      # streams as the only ones in the library that ended a failed generation
      # indistinguishably from a successful one. The request is still in flight,
      # so the `:timed_out` state carries `server` for the cleanup to cancel.
      timeout -> {[{:error, :timeout}], {:timed_out, server, ref}}
    end
  end

  defp stream_cleanup({:rejected, ref, _reason}), do: drain_stream_messages(ref)
  defp stream_cleanup({:done, ref}), do: drain_stream_messages(ref)

  defp stream_cleanup({:timed_out, server, ref}) do
    cancel(server, ref)
    drain_stream_messages(ref)
  end

  defp stream_cleanup({server, ref, _timeout}) do
    cancel(server, ref)
    drain_stream_messages(ref)
  end

  defp drain_stream_messages(ref) do
    receive do
      {^ref, _} -> drain_stream_messages(ref)
    after
      0 -> :ok
    end
  end

  @doc """
  Generates text from pre-tokenized input. Blocks until generation is complete.

  Use `get_model/1` to obtain the model for tokenization outside the server.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:timeout` - Call timeout in ms. Defaults to
      `#{LlamaCppEx.Options.blocking_timeout()}`.

  """
  @spec generate_tokens(GenServer.server(), [integer()], keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def generate_tokens(server, token_ids, opts \\ []) when is_list(token_ids) do
    opts = Options.validate!(opts, @call_opt_keys, "LlamaCppEx.Server.generate_tokens/3")
    timeout = Options.timeout(opts, :blocking)
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    req_opts = Keyword.take(opts, @request_opt_keys)

    with :ok <- validate_request_grammar(server, req_opts) do
      GenServer.call(server, {:generate_tokens, token_ids, max_tokens, req_opts}, timeout)
    end
  end

  @doc """
  Like `generate_tokens/3`, but returns completion metadata alongside the text.

  Returns `{:ok, %{text: text, completion_tokens: n, finish_reason: reason}}`
  where `reason` is `:eog` or `:max_tokens`. Used by the OpenAI-shaped
  `LlamaCppEx.chat_completion/3` when routed through a server.
  """
  @spec complete_tokens(GenServer.server(), [integer()], keyword()) ::
          {:ok, %{text: String.t(), completion_tokens: non_neg_integer(), finish_reason: atom()}}
          | {:error, term()}
  def complete_tokens(server, token_ids, opts \\ []) when is_list(token_ids) do
    opts = Options.validate!(opts, @call_opt_keys, "LlamaCppEx.Server.complete_tokens/3")
    timeout = Options.timeout(opts, :blocking)
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    req_opts = Keyword.take(opts, @request_opt_keys) ++ [reply: :full]

    with :ok <- validate_request_grammar(server, req_opts) do
      GenServer.call(server, {:generate_tokens, token_ids, max_tokens, req_opts}, timeout)
    end
  end

  @doc false
  # Subscribes `self()` to a token stream using the raw message protocol:
  # {ref, {:token, piece}} per piece, then {ref, {:done, :eog | :max_tokens}}
  # or {ref, {:error, reason}}. Internal — used by the chat-completion
  # streaming path; library users should call stream/3 or stream_tokens/3.
  #
  # The `@spec` used to say `:ok`. The body returns `GenServer.call/2` verbatim
  # and its only caller depends on the `{:error, :queue_full}` the spec denied;
  # Dialyzer could not see the lie because `GenServer.call/2` is typed `term()`.
  #
  # It also never validated its options, so every key the facade accepted and
  # the server does not was silently dropped on the streaming path while the
  # blocking path raised for it.
  @spec subscribe_stream_tokens(GenServer.server(), [integer()], reference(), keyword()) ::
          :ok | {:error, term()}
  def subscribe_stream_tokens(server, token_ids, ref, opts \\ []) when is_list(token_ids) do
    opts = Options.validate!(opts, @call_opt_keys, "LlamaCppEx.Server.subscribe_stream_tokens/4")
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    req_opts = Keyword.take(opts, @request_opt_keys)

    # Bounded by the caller's `:timeout` for the same reason as
    # `start_token_stream/5`: the only caller runs this inside a
    # `Stream.resource` start-function, where the implicit 5000 ms default exited
    # instead of producing the `{:error, reason}` element the caller's
    # `:setup_failed` branch was written for.
    with :ok <- validate_request_grammar(server, req_opts) do
      call_or_error(
        server,
        {:stream_tokens, token_ids, max_tokens, self(), ref, req_opts},
        Options.timeout(opts, :stream)
      )
    end
  end

  @doc """
  Returns a stream of generated text chunks from pre-tokenized input.

  Emits a single `{:error, reason}` element and halts on rejection, mid-generation
  failure, or a per-token timeout — see `stream/3`.

  ## Options

    * `:max_tokens` - Maximum tokens to generate. Defaults to `256`.
    * `:timeout` - Per-token timeout, and the budget for being admitted to a slot
      or the queue. Defaults to `#{LlamaCppEx.Options.stream_timeout()}`.

  """
  @spec stream_tokens(GenServer.server(), [integer()], keyword()) :: Enumerable.t()
  def stream_tokens(server, token_ids, opts \\ []) when is_list(token_ids) do
    opts = Options.validate!(opts, @call_opt_keys, "LlamaCppEx.Server.stream_tokens/3")
    max_tokens = Keyword.get(opts, :max_tokens, 256)
    timeout = Options.timeout(opts, :stream)
    req_opts = Keyword.take(opts, @request_opt_keys)

    Stream.resource(
      fn ->
        case validate_request_grammar(server, req_opts) do
          :ok -> start_token_stream(server, token_ids, max_tokens, req_opts, timeout)
          {:error, reason} -> {:rejected, make_ref(), reason}
        end
      end,
      &stream_next/1,
      &stream_cleanup/1
    )
  end

  @doc """
  Cancels an in-flight or queued stream request by its subscription reference.

  The slot stops being scheduled immediately and is freed for other requests
  (its prefix cache is retained per the request's `:cache_prompt`). Consumer
  death is detected automatically via monitors — explicit cancel is for
  consumers that stop reading without exiting. `Server.stream/3` and
  `stream_tokens/3` call this from their cleanup, so halting those streams
  early (e.g. `Enum.take/2`) cancels generation instead of burning batch
  budget to `max_tokens`.
  """
  @spec cancel(GenServer.server(), reference()) :: :ok
  def cancel(server, ref) when is_reference(ref) do
    GenServer.cast(server, {:cancel, ref})
  end

  @doc """
  Returns the model struct for external tokenization, or an error.

  The model resource is reference-counted and thread-safe for read-only
  operations like tokenization. Served from `LlamaCppEx.Registry` — an ETS read,
  no round-trip through the server's mailbox.

  Returns `{:error, :noproc}` when `server` does not resolve to a live process or
  died on its way down, and `{:error, :not_ready}` when the server is still
  loading its model (the window between `start_link/1` returning and
  `handle_continue/2` finishing).

  A server whose model load *failed* returns `{:error, {:load_failed, reason}}`:
  `handle_continue/2` stops with that reason, so the `:get_model` call exits with
  it. That escaped the previous three `catch` clauses — from inside both
  `Stream.resource` start-functions, where nothing can catch it — even though the
  `@spec` promised a total function.
  """
  @spec fetch_model(GenServer.server()) :: {:ok, Model.t()} | {:error, term()}
  def fetch_model(server) do
    case GenServer.whereis(server) do
      pid when is_pid(pid) ->
        case Registry.lookup(LlamaCppEx.Registry, {__MODULE__, pid}) do
          [{^pid, %Model{} = model}] ->
            {:ok, model}

          [] ->
            # Either the server has not finished loading yet or the Registry
            # entry lost a race with a restart. Ask the server directly; it
            # answers only once handle_continue has published the model.
            fetch_model_via_call(pid)
        end

      _ ->
        {:error, :noproc}
    end
  end

  defp fetch_model_via_call(pid) do
    {:ok, GenServer.call(pid, :get_model, Options.blocking_timeout())}
  catch
    # A timeout means handle_continue/2 has not published the model yet: still
    # loading, not broken. Every other reason is classified by call_exit_reason/1.
    :exit, {:timeout, _} -> {:error, :not_ready}
    :exit, {reason, {GenServer, :call, _}} -> {:error, call_exit_reason(reason)}
    :exit, reason -> {:error, call_exit_reason(reason)}
  end

  @doc """
  Returns the model struct for external tokenization.

  Raises when the server is not running or has not finished loading. Use
  `fetch_model/1` when either is a possibility.

  The previous `@spec` claimed a total function while the implementation exited
  with `{:noproc, ...}` on a dead server, so callers had no documented way to
  handle it.
  """
  @spec get_model(GenServer.server()) :: Model.t()
  def get_model(server) do
    case fetch_model(server) do
      {:ok, model} ->
        model

      {:error, reason} ->
        raise ArgumentError,
              "LlamaCppEx.Server.get_model/1: no model available for #{inspect(server)} " <>
                "(#{inspect(reason)}). Use fetch_model/1 to handle this."
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
    # init/1 does only what is cheap and can fail fast, so start_link/1 still
    # reports a misconfiguration synchronously. Everything expensive —
    # backend init, the model load (hundreds of MB), context + KV allocation,
    # and n_parallel sampler chains — moves to handle_continue/2 so it does not
    # block the supervision tree's boot. These used to be hard-matched
    # (`{:ok, x} = ...`), turning a load failure into an opaque MatchError
    # instead of a `{:stop, reason}`. ModelManager already uses this shape.
    with :ok <- validate_server_opts(opts),
         {:ok, model_path} <- validate_model_path(opts),
         {:ok, batch_strategy} <- validate_batch_strategy(opts) do
      n_parallel = Keyword.get(opts, :n_parallel, 4)
      n_ctx = Keyword.get(opts, :n_ctx, 8192)

      # Trap exits so terminate/2 runs on shutdown — that is the only reason.
      #
      # It does NOT insulate the server from a linked process dying: the
      # `{:EXIT, _, reason}` clause in handle_info/2 honours the signal with
      # `{:stop, reason, state}`, so a link that breaks takes the model down. An
      # earlier version of this comment claimed the opposite, which matters because
      # the two readings imply opposite failure modes for anything that links
      # itself to a server.
      Process.flag(:trap_exit, true)

      state = %__MODULE__{
        sampler_opts: Keyword.take(opts, @sampler_opt_keys),
        queue: :queue.new(),
        max_queue: Keyword.get(opts, :max_queue, @default_max_queue),
        n_parallel: n_parallel,
        n_batch: Keyword.get(opts, :n_batch, min(n_ctx, 2048)),
        chunk_size: Keyword.get(opts, :chunk_size, 512),
        cache_prompt: Keyword.get(opts, :cache_prompt, true),
        prompt_cache: PromptCache.new(Keyword.get(opts, :prompt_cache_ram_mb, 0)),
        batch_strategy: batch_strategy
      }

      {:ok, state, {:continue, {:load, model_path, opts}}}
    end
  end

  @impl true
  def handle_continue({:load, model_path, opts}, state) do
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, 99)
    kv_unified = Keyword.get(opts, :kv_unified, true)
    model_opts = Keyword.take(opts, Model.tuning_option_keys())
    context_opts = Keyword.take(opts, Context.tuning_option_keys())

    with :ok <- LlamaCppEx.init(),
         {:ok, model} <- Model.load(model_path, [n_gpu_layers: n_gpu_layers] ++ model_opts),
         {:ok, ctx} <-
           Context.create(
             model,
             [
               n_ctx: Keyword.get(opts, :n_ctx, 8192),
               n_batch: state.n_batch,
               n_seq_max: state.n_parallel,
               kv_unified: kv_unified
             ] ++ context_opts
           ),
         {:ok, slots} <- create_slots(model, state.sampler_opts, state.n_parallel) do
      # Publish the model handle for callers (get_model/1): tokenization and
      # chat templating happen client-side without a GenServer.call round-trip.
      # The Registry entry is removed automatically when this process dies.
      {:ok, _} = Registry.register(LlamaCppEx.Registry, {__MODULE__, self()}, model)

      # Probe seq_rm support BEFORE any decode work — the call has the side
      # effect of clearing KV memory. Hybrid models (GDN, e.g. Qwen 3.5/3.6)
      # report `:full`, meaning partial range trims aren't supported; we'd
      # otherwise produce M-RoPE position-mismatch aborts in the prefix-cache
      # path when an old slot's KV tail extends past the new prompt's prefix
      # match.
      seq_rm_kind = LlamaCppEx.NIF.context_can_seq_rm(ctx.ref)

      if state.cache_prompt and seq_rm_kind == :full do
        Logger.info(
          "LlamaCppEx.Server: cache_prompt: true requested but model reports " <>
            "seq_rm support = :full (hybrid GDN). Prefix cache will only fire " <>
            "for exact-prefix continuations; cache hits requiring partial KV " <>
            "trim will fall back to a full slot reset."
        )
      end

      {:noreply,
       %{
         state
         | model: model,
           ctx: ctx,
           slots: slots,
           cross_slot_sharing: kv_unified and seq_rm_kind == :part,
           seq_rm_kind: seq_rm_kind
       }}
    else
      {:error, reason} ->
        {:stop, {:load_failed, reason}, state}
    end
  end

  defp create_slots(model, sampler_opts, n_parallel) do
    now = System.monotonic_time()

    Enum.reduce_while(0..(n_parallel - 1), {:ok, %{}}, fn seq_id, {:ok, acc} ->
      case Sampler.create(model, sampler_opts) do
        {:ok, sampler} ->
          slot =
            idle_slot_fields([], 0)
            |> Map.put(:sampler, sampler)
            |> Map.put(:t_last_used, now)
            |> Map.put(:session, nil)

          {:cont, {:ok, Map.put(acc, seq_id, slot)}}

        {:error, reason} ->
          {:halt, {:error, reason}}
      end
    end)
  end

  defp validate_server_opts(opts) do
    Options.validate!(opts, @start_opt_keys, "LlamaCppEx.Server.start_link/1")
    :ok
  end

  defp validate_model_path(opts) do
    case Keyword.fetch(opts, :model_path) do
      {:ok, path} when is_binary(path) ->
        # Checked here rather than in handle_continue so the overwhelmingly
        # common misconfiguration still fails start_link/1 synchronously.
        if File.regular?(path) do
          {:ok, path}
        else
          {:stop, {:model_not_found, path}}
        end

      {:ok, other} ->
        {:stop, {:invalid_model_path, other}}

      :error ->
        {:stop, {:missing_option, :model_path}}
    end
  end

  # The strategy layer is genuinely pluggable (a real @callback with @impl true
  # implementations, dispatched by module value), so a typo is a legitimate
  # configuration error — but it used to surface as an UndefinedFunctionError
  # inside handle_info(:tick), i.e. after the multi-hundred-MB model load.
  defp validate_batch_strategy(opts) do
    module = Keyword.get(opts, :batch_strategy, LlamaCppEx.Server.Strategy.DecodeMaximal)

    cond do
      not is_atom(module) ->
        {:stop, {:invalid_batch_strategy, module}}

      not Code.ensure_loaded?(module) ->
        {:stop, {:invalid_batch_strategy, {:module_not_available, module}}}

      not function_exported?(module, :build_batch, 4) ->
        {:stop, {:invalid_batch_strategy, {:build_batch_4_not_exported, module}}}

      true ->
        {:ok, module}
    end
  end

  @impl true
  def handle_call({:generate_tokens, token_ids, max_tokens, req_opts}, from, state) do
    if token_ids == [] do
      {:reply, {:error, :empty_prompt}, state}
    else
      request = Request.sync(token_ids, max_tokens, from, req_opts)

      case acquire_slot(state, token_ids, req_opts) do
        {:ok, seq_id, lcp, state} ->
          state = init_slot(state, seq_id, request, lcp)
          state = maybe_schedule_tick(state)
          {:noreply, state}

        :no_slots ->
          enqueue_or_reject(state, request, from, _reply_ok? = false)
      end
    end
  end

  def handle_call({:stream_tokens, token_ids, max_tokens, pid, ref, req_opts}, from, state) do
    if token_ids == [] do
      # Same guard as :generate_tokens — an empty prompt would enter the tick
      # with nothing to prefill and no logits to sample, hanging the consumer
      # until its stream timeout.
      {:reply, {:error, :empty_prompt}, state}
    else
      request = Request.stream(token_ids, max_tokens, pid, ref, req_opts)

      case acquire_slot(state, token_ids, req_opts) do
        {:ok, seq_id, lcp, state} ->
          state = init_slot(state, seq_id, request, lcp)
          GenServer.reply(from, :ok)
          state = maybe_schedule_tick(state)
          {:noreply, state}

        :no_slots ->
          enqueue_or_reject(state, request, from, _reply_ok? = true)
      end
    end
  end

  # Answers only once handle_continue/2 has published the model, because a
  # continue runs before any queued message. fetch_model/1 falls back to this
  # when the Registry entry is not there yet.
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
      ram_cache_entries: PromptCache.size(state.prompt_cache),
      ram_cache_bytes: state.prompt_cache.bytes
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:cancel, ref}, state) do
    state =
      case Enum.find(state.slots, fn {_id, slot} -> slot.stream_ref == ref end) do
        {seq_id, _slot} -> cancel_slot(state, seq_id)
        nil -> drop_queued_request(state, ref)
      end

    {:noreply, state}
  end

  @impl true
  def handle_info(:tick, state) do
    state = %{state | tick_scheduled: false}
    state = run_tick(state)
    {:noreply, state}
  end

  def handle_info({:DOWN, mref, :process, pid, _reason}, state) do
    # A request's consumer died — free its slot instead of generating into
    # the void until max_tokens. Queued requests from the dead pid are
    # dropped as well.
    state =
      case Enum.find(state.slots, fn {_id, slot} -> slot.monitor_ref == mref end) do
        {seq_id, _slot} -> cancel_slot(state, seq_id)
        nil -> state
      end

    {:noreply, drop_queued_requests_from(state, pid)}
  end

  def handle_info({:EXIT, _pid, reason}, state) do
    # trap_exit is on so terminate/2 runs on shutdown (see init/1); honouring the
    # signal here is what keeps a supervisor's `Process.exit(pid, :shutdown)`
    # working. The cost is that *every* exit signal stops the server, `:normal`
    # included — which an untrapped process would silently ignore.
    #
    # That makes this clause a landmine for anything that links itself here and
    # then exits normally. `LlamaCppEx.Generator` used to leave exactly such a
    # signal behind — an already-queued `{:EXIT, pid, :normal}` survives
    # `Process.unlink/1` — so a server driving a generator would have shut down
    # with reason `:normal` and no log line. `Generator.stop/1` drains it now.
    {:stop, reason, state}
  end

  # Catch-all. Without it, one stray message — a late `:ssl_closed`, a reply to a
  # timed-out call, anything a library sends to a pid it once saw — is a
  # FunctionClauseError that kills the server. That drops the %Model{} and
  # %Context{} refs and fails every in-flight request, and because backing
  # servers are started `restart: :temporary` (ModelManager.ModelIO), it never
  # comes back. ModelManager has had this clause since it was written.
  def handle_info(msg, state) do
    Logger.debug("LlamaCppEx.Server: ignoring unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, _state) do
    # The model handle lives in LlamaCppEx.Registry, which unregisters it when
    # this process dies — including on Process.exit(server, :kill), where
    # terminate/2 never runs. That leak, and the global GC that
    # :persistent_term.put/erase triggered on every model swap, is why the
    # Registry replaced it. Nothing left to clean up here.
    :ok
  end

  # --- Internal: Slot management ---

  defp request_cache_prompt?(state, req_opts) do
    Keyword.get(req_opts, :cache_prompt, state.cache_prompt)
  end

  # Queues a request for the next free slot, or rejects it when :max_queue is
  # bound and full. Stream subscriptions reply :ok up-front (tokens arrive as
  # messages); sync requests stay unanswered until finish_slot replies.
  defp enqueue_or_reject(state, request, from, reply_ok?) do
    if state.max_queue > 0 and :queue.len(state.queue) >= state.max_queue do
      {:reply, {:error, :queue_full}, state}
    else
      if reply_ok?, do: GenServer.reply(from, :ok)
      {:noreply, enqueue_request(state, request)}
    end
  end

  # Returns the chosen slot together with the prefix match already computed for
  # it, so init_slot/8 does not recompute an LCP the picker just discarded
  # (~69 µs per request at a 32k prompt). `nil` means "not computed here".
  defp acquire_slot(state, tokens, req_opts) do
    idle_slots = Enum.filter(state.slots, fn {_id, slot} -> slot.state == :idle end)

    case idle_slots do
      [] ->
        :no_slots

      slots ->
        session_pick = Slots.session_slot_if_idle(state.sessions, session_key(req_opts), slots)

        cond do
          session_pick != nil ->
            {:ok, session_pick, nil, state}

          tokens != [] and request_cache_prompt?(state, req_opts) ->
            {seq_id, lcp} = Slots.pick_cached_slot(slots, tokens)
            {:ok, seq_id, lcp, state}

          true ->
            {:ok, Slots.pick_lru_slot(slots), nil, state}
        end
    end
  end

  # Builds the request's sampler before anything else, so a rejection costs
  # nothing to undo, then hands off to install_slot/5.
  #
  # `:grammar` is a caller-supplied *value* and `Options.validate!/3` checks key
  # names only. The public entry points now reject an uncompilable grammar at
  # admission (validate_request_grammar/2); this `case` is the depth-2 defence
  # for the two disagreeing. The previous `{:ok, sampler} = Sampler.create(...)`
  # made that disagreement a crash of the GenServer holding the model — and
  # because backing servers run `restart: :temporary`, it never came back.
  defp init_slot(state, seq_id, %Request{opts: req_opts} = request, lcp) do
    # Fresh sampler per request: request opts override server defaults, and a
    # new chain means clean grammar/penalty state and a fresh seed. The old
    # sampler resource is dropped and freed by GC.
    sampler_opts = Keyword.merge(state.sampler_opts, Keyword.take(req_opts, @sampler_opt_keys))

    case Sampler.create(state.model, sampler_opts) do
      {:ok, sampler} ->
        install_slot(state, seq_id, request, lcp, sampler)

      {:error, reason} ->
        Logger.warning(
          "LlamaCppEx.Server: failing request for slot #{seq_id} — sampler " <>
            "creation refused #{inspect(reason)}"
        )

        reject_request(state, request, reason)
    end
  end

  # Fails a request that never entered its slot. No slot field was written and
  # no KV was touched, so unlike fail_slot/3 there is nothing to reset.
  defp reject_request(state, %Request{} = request, reason) do
    if request.from, do: GenServer.reply(request.from, {:error, reason})

    if request.stream_pid && request.stream_ref do
      send(request.stream_pid, {request.stream_ref, {:error, reason}})
    end

    state
  end

  # Takes the whole %Request{} rather than seven positional arguments: the old
  # `init_slot(state, seq_id, tokens, max_tokens, from, stream_pid, stream_ref,
  # req_opts, lcp)` was easy to call with `from` and `stream_pid` transposed,
  # and the two call shapes (sync vs stream) each passed `nil` for the other's
  # three fields.
  defp install_slot(state, seq_id, %Request{} = request, lcp, sampler) do
    %Request{tokens: tokens, max_tokens: max_tokens, opts: req_opts} = request

    state = update_session_mapping(state, seq_id, session_key(req_opts))
    slot = state.slots[seq_id]
    cache_prompt? = request_cache_prompt?(state, req_opts)

    # Computed once and threaded through: it was called six times below and in
    # resolve_prefix_cache's callees, at 3.7 µs per call on a 32k prompt.
    n_tokens = length(tokens)

    scope = Keyword.get(req_opts, :cache_scope)

    {raw_match, own_match} =
      own_cache_match(state, slot, tokens, n_tokens, cache_prompt?, scope, lcp)

    state = maybe_warn_prefix_instability(state, seq_id, slot, raw_match, cache_prompt?)

    {state, n_match} =
      resolve_prefix_cache(state, seq_id, tokens, own_match, cache_prompt?, scope)

    slot = state.slots[seq_id]

    # Watch the consumer (stream subscriber or sync caller) so its death
    # cancels the request instead of burning batch budget to max_tokens.
    monitor_ref =
      case Request.consumer_pid(request) do
        nil -> nil
        pid -> Process.monitor(pid)
      end

    slot = %{
      slot
      | state: :prefilling,
        from: request.from,
        stream_pid: request.stream_pid,
        stream_ref: request.stream_ref,
        monitor_ref: monitor_ref,
        sampler: sampler,
        reply_mode: Keyword.get(req_opts, :reply, :text),
        cache_prompt: cache_prompt?,
        prompt_tokens: tokens,
        prompt_tokens_tuple: List.to_tuple(tokens),
        # The previous request's history has served its purpose (raw_match above
        # and resolve_prefix_cache's RAM-cache offer both already read it) and
        # nothing reads either field while a slot is :prefilling or :generating —
        # donor_prefix_match/2 uses cached_tokens only for :idle slots, and
        # purgeable_seq_ids/1 only looks at idle ones. Dropping it here releases a
        # whole prompt-sized list per busy slot: it used to stay reachable
        # alongside the new prompt's list *and* tuple, so a slot held three
        # copies of a prompt for the entire request (~2.6 MB at n_parallel: 8
        # with 8k prompts, all of it garbage the process running every forward
        # pass had to scan). reset_slot/2 rebuilds it from prompt_tokens.
        cached_tokens: [],
        cached_pos: 0,
        cache_scope: scope,
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
        n_prompt_tokens: n_tokens,
        generated_token_ids: [],
        n_prefix_cache_tokens: n_match
    }

    :telemetry.execute(
      [:llama_cpp_ex, :server, :request, :start],
      %{prompt_tokens: n_tokens, prefix_cache_tokens: n_match},
      %{server: self(), seq_id: seq_id, mode: slot_mode(slot)}
    )

    put_in(state.slots[seq_id], slot)
  end

  # How much of this slot's own cached prefix this request may reuse.
  #
  # Returns `{raw_match, own_match}`. `raw_match` is what the tokens allow and is
  # what the prefix-instability telemetry reports; `own_match` is what is actually
  # safe to keep. They differ on models that only support whole-sequence seq_rm
  # (`:full`, e.g. hybrid GDN): a partial trim there silently fails and leaves
  # stale KV past the match, producing an M-RoPE position-mismatch abort on the
  # next decode, so a match that would need trimming is discarded outright.
  #
  # Both are capped at `n_tokens - 1`: at least the last prompt token must be
  # decoded to produce logits for sampling the first generated token (llama-server
  # makes the same n_past-- adjustment). An uncapped full match would enter the
  # tick with nothing to prefill and no logits to sample — a stuck slot.
  defp own_cache_match(state, slot, tokens, n_tokens, cache_prompt?, scope, lcp) do
    max_reuse = n_tokens - 1

    raw_match =
      cond do
        not cache_prompt? ->
          0

        # A cached prefix belongs to the scope that produced it. Reusing it for a
        # different scope hands one caller's KV to another — the tokens match, so
        # nothing downstream would notice.
        slot.cache_scope != scope ->
          0

        # acquire_slot/3 already matched `tokens` against this slot's cache.
        is_integer(lcp) ->
          min(lcp, max_reuse)

        true ->
          min(Slots.common_prefix_length(tokens, slot.cached_tokens), max_reuse)
      end

    needs_trim = raw_match > 0 and raw_match < slot.cached_pos
    own_match = if needs_trim and state.seq_rm_kind == :full, do: 0, else: raw_match

    {raw_match, own_match}
  end

  # Decides where this request's cached prefix comes from and prepares the
  # slot's KV accordingly. The three sources compete by match length, with
  # ties broken by cost: the slot's own cache (free) beats a donor slot
  # (metadata-only seq_cp under unified KV) beats the RAM prompt cache
  # (KV-sized memcpy). Whenever the slot's own cache is about to be destroyed
  # or truncated, it is offered to the RAM cache first.
  defp resolve_prefix_cache(state, seq_id, tokens, own_match, cache_prompt?, scope) do
    slot = state.slots[seq_id]

    donor = best_donor(state, seq_id, tokens, own_match, cache_prompt?, scope)
    donor_lcp = if donor, do: elem(donor, 1), else: 0

    ram =
      if cache_prompt? do
        PromptCache.best_candidate(state.prompt_cache, tokens, scope, state.seq_rm_kind)
      end

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

    # A partial cross-sequence copy is only expressible on a unified KV cache;
    # the NIF refuses it on a split cache rather than letting llama.cpp abort the
    # VM. `cross_slot_sharing` should already have kept us out of that case, so a
    # refusal here means the two disagree — fall back to a full re-prefill rather
    # than reporting a prefix we do not actually have.
    case LlamaCppEx.NIF.memory_seq_cp(state.ctx.ref, donor_id, seq_id, 0, donor_lcp) do
      :ok ->
        {state, donor_lcp}

      {:error, reason} ->
        Logger.warning(
          "LlamaCppEx.Server: donor cache adoption refused (#{inspect(reason)}); " <>
            "re-prefilling seq #{seq_id} from scratch"
        )

        _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
        {state, 0}
    end
  end

  defp adopt_ram_cache(state, seq_id, slot, {entry, ram_lcp}) do
    state = maybe_save_to_ram_cache(state, seq_id, slot)
    _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
    {state, apply_ram_restore(state, seq_id, entry, ram_lcp)}
  end

  defp keep_own_cache(state, seq_id, slot, own_match) do
    if own_match < slot.cached_pos do
      # Trim KV cache beyond the matched prefix. The truncated tail may still be
      # valuable to another conversation — offer the full state to the RAM cache
      # before cutting it.
      state = maybe_save_to_ram_cache(state, seq_id, slot)

      # `memory_seq_rm` *returns false*, it does not raise, when the memory module
      # refuses the range: `llama_memory_recurrent::seq_rm` honours a partial
      # rollback of at most `n_rs_seq` positions and returns false beyond that
      # (`vendor/llama.cpp/src/llama-memory-recurrent.cpp:181-187`). The
      # `seq_rm_kind` guard above covers only `:full`, so on an `:rs` context
      # `true = ...` was a MatchError inside the tick — a crash of the process
      # holding the model for a condition whose correct answer is "re-prefill".
      # Same shape as the donor-refusal branch above.
      if LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, own_match, -1) do
        {state, own_match}
      else
        Logger.warning(
          "LlamaCppEx.Server: partial seq_rm refused for seq #{seq_id} " <>
            "(trim to #{own_match} of #{slot.cached_pos} cached positions); " <>
            "re-prefilling from scratch"
        )

        _ = LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
        {state, 0}
      end
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

  # Offers a slot's about-to-be-destroyed cache to the RAM prompt cache. The
  # cache's own rules live in LlamaCppEx.Server.PromptCache; this function is the
  # NIF-and-telemetry shell around them.
  defp maybe_save_to_ram_cache(state, seq_id, slot) do
    {cache, saved, evicted} = PromptCache.save(state.prompt_cache, state.ctx.ref, seq_id, slot)
    state = %{state | prompt_cache: cache}

    Enum.each(evicted, &emit_ram_cache_telemetry(state, :evict, &1))
    if saved, do: emit_ram_cache_telemetry(state, :save, saved)

    state
  end

  # Restores a RAM cache entry into an (empty) sequence and trims the unusable
  # tail. Returns the number of reusable prefix tokens.
  defp apply_ram_restore(state, seq_id, entry, lcp) do
    case PromptCache.restore(state.ctx.ref, seq_id, entry, lcp) do
      {:ok, reusable} ->
        emit_ram_cache_telemetry(state, :restore, entry)
        reusable

      {:error, reason} ->
        Logger.warning("LlamaCppEx.Server: RAM cache restore failed: #{inspect(reason)}")
        0
    end
  end

  defp emit_ram_cache_telemetry(state, op, entry) do
    :telemetry.execute(
      [:llama_cpp_ex, :server, :ram_cache],
      %{
        bytes: entry.bytes,
        tokens: entry.len,
        total_bytes: state.prompt_cache.bytes,
        entries: PromptCache.size(state.prompt_cache)
      },
      %{server: self(), op: op}
    )
  end

  # Finds the slot whose in-KV tokens share the longest prefix with the new
  # prompt, when that beats the assigned slot's own match. Active donors count
  # too — only their FED tokens are in the KV, so the match is capped at the
  # fed length. The pos_max probe guards against any bookkeeping drift between
  # slot state and the actual KV contents.
  # Donors are additionally restricted to the same `:cache_scope`: adopting
  # another slot's KV is exactly the cross-request read the scope exists to
  # prevent, and here it is live KV rather than a saved blob.
  defp best_donor(state, dst_seq_id, tokens, own_match, cache_prompt?, scope) do
    if cache_prompt? and state.cross_slot_sharing do
      # Cap at len-1: the last prompt token must be decoded for logits.
      max_reuse = length(tokens) - 1

      state.slots
      |> Enum.reject(fn {id, slot} -> id == dst_seq_id or slot.cache_scope != scope end)
      |> Enum.map(fn {id, slot} ->
        {id, min(Slots.donor_prefix_match(slot, tokens), max_reuse)}
      end)
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

  defp slot_mode(%{stream_pid: pid}) when is_pid(pid), do: :stream
  defp slot_mode(_slot), do: :generate

  # Session affinity is keyed by `{cache_scope, session}`, never by the session id
  # alone.
  #
  # `:session` was a *global* keyspace: affinity routed on the session id while
  # only prefix *reuse* checked `:cache_scope`, so a guessed session id let one
  # scope claim the slot another scope was using and evict its prefix cache. The
  # scope check still cleared the KV on mismatch, so this is a denial of service
  # rather than a KV leak — worth closing while the feature is new.
  #
  # `nil` in, `nil` out: a request with no `:session` has no affinity, and pairing
  # a scope with a missing session id would invent one.
  defp session_key(req_opts) do
    case Keyword.get(req_opts, :session) do
      nil -> nil
      session -> {Keyword.get(req_opts, :cache_scope), session}
    end
  end

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
        opts = request.opts

        case Slots.session_slot_if_idle(state.sessions, session_key(opts), idle_slots(state)) do
          nil ->
            {state, [request | rest]}

          seq_id ->
            # Session affinity, not a similarity pick — no LCP was computed.
            {assign_queued_request(state, seq_id, request, nil), rest}
        end
      end)

    %{state | queue: :queue.from_list(Enum.reverse(remaining))}
  end

  defp idle_slots(state) do
    Enum.filter(state.slots, fn {_id, slot} -> slot.state == :idle end)
  end

  defp dequeue_fifo(state) do
    case :queue.out(state.queue) do
      {{:value, %Request{} = request}, queue} ->
        state = %{state | queue: queue}

        case acquire_slot(state, request.tokens, request.opts) do
          {:ok, seq_id, lcp, state} ->
            state = assign_queued_request(state, seq_id, request, lcp)
            dequeue_fifo(state)

          :no_slots ->
            # Put it back
            %{state | queue: :queue.in_r(request, state.queue)}
        end

      {:empty, _queue} ->
        state
    end
  end

  defp assign_queued_request(state, seq_id, %Request{} = request, lcp) do
    init_slot(state, seq_id, request, lcp)
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
    demonitor_slot(slot)

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
  # into the slot's accumulated output/counters. Streamed bytes pass through
  # a per-slot UTF-8 buffer so a codepoint split across tokens is never sent
  # partially; the accumulated text keeps raw pieces (whole result is valid).
  defp emit_piece(slot, piece, now) do
    utf8_pending =
      if slot.stream_pid && slot.stream_ref do
        {out, pending} = UTF8Stream.push(slot.utf8_pending, piece)

        if out != "" do
          send(slot.stream_pid, {slot.stream_ref, {:token, out}})
        end

        pending
      else
        slot.utf8_pending
      end

    %{
      slot
      | accumulated_pieces: [piece | slot.accumulated_pieces],
        utf8_pending: utf8_pending,
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
    demonitor_slot(slot)

    if slot.from do
      GenServer.reply(slot.from, {:ok, completion_reply(slot, stop_reason)})
    end

    if slot.stream_pid && slot.stream_ref do
      # Flush any held-back trailing bytes (best effort — an incomplete
      # codepoint at end-of-generation is the model's own output).
      if slot.utf8_pending != "" do
        send(slot.stream_pid, {slot.stream_ref, {:token, slot.utf8_pending}})
      end

      send(slot.stream_pid, {slot.stream_ref, {:done, stop_reason}})
    end

    # Emit telemetry
    emit_request_done(slot, seq_id, t_end, stop_reason)

    reset_slot(state, seq_id)
  end

  # Releases a slot whose consumer went away (died or explicitly cancelled).
  # Nothing to reply to — the prefix cache is retained per the request's
  # cache_prompt so a resumed conversation can still hit it, and the freed
  # slot immediately serves the queue.
  defp cancel_slot(state, seq_id) do
    slot = state.slots[seq_id]
    demonitor_slot(slot)
    emit_request_done(slot, seq_id, System.monotonic_time(), :cancelled)

    state
    |> reset_slot(seq_id)
    |> dequeue_into_slot()
    |> continue_if_active()
  end

  defp demonitor_slot(%{monitor_ref: nil}), do: :ok
  defp demonitor_slot(%{monitor_ref: mref}), do: Process.demonitor(mref, [:flush])

  # Drops a queued (not yet slotted) stream request by its subscription ref.
  defp drop_queued_request(state, ref) do
    queue =
      :queue.filter(
        fn %Request{stream_ref: req_ref} -> req_ref != ref end,
        state.queue
      )

    %{state | queue: queue}
  end

  # Drops queued requests whose consumer (stream pid or sync caller) died.
  defp drop_queued_requests_from(state, pid) do
    queue =
      :queue.filter(
        fn %Request{} = request -> Request.consumer_pid(request) != pid end,
        state.queue
      )

    %{state | queue: queue}
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

    # Build the token history matching what is actually in the KV. Retention
    # follows the per-request setting: a cache_prompt: false request leaves
    # nothing behind. A slot cancelled mid-prefill only has prefill_pos
    # positions in KV — caching the full prompt would advertise positions
    # that don't exist. (No sampler reset — every request gets a fresh
    # sampler at init_slot.)
    {cached_tokens, cached_pos} =
      cond do
        not slot.cache_prompt ->
          LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)
          {[], 0}

        slot.state == :prefilling ->
          {Enum.take(slot.prompt_tokens, slot.prefill_pos), slot.prefill_pos}

        true ->
          {slot.prompt_tokens ++ Enum.reverse(slot.generated_token_ids), slot.pos}
      end

    slot =
      slot
      |> Map.merge(idle_slot_fields(cached_tokens, cached_pos, slot.cache_scope))
      |> Map.put(:t_last_used, System.monotonic_time())

    put_in(state.slots[seq_id], slot)
  end

  # The single source of truth for a slot's per-request fields. init/1,
  # reset_slot/2, and the failure paths all build from this map, so a new
  # slot field cannot silently carry stale data across requests. The
  # prefix-cache carry-over is the only caller-controlled part; :sampler,
  # :t_last_used, and :session are the only fields that live outside it
  # (slot metadata that must survive request resets).
  #
  # `cache_scope` travels with `cached_tokens` because it describes whose tokens
  # they are: it is the `:cache_scope` of the request that produced them, and
  # every prefix-reuse path refuses to cross it.
  defp idle_slot_fields(cached_tokens, cached_pos, cache_scope \\ nil) do
    %{
      state: :idle,
      from: nil,
      stream_pid: nil,
      stream_ref: nil,
      monitor_ref: nil,
      reply_mode: :text,
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
      utf8_pending: "",
      t_start: nil,
      t_first_token: nil,
      n_prompt_tokens: 0,
      cached_tokens: cached_tokens,
      cached_pos: cached_pos,
      cache_scope: cache_scope,
      generated_token_ids: [],
      n_prefix_cache_tokens: 0
    }
  end

  # Builds the final completion string from the reverse-ordered piece list.
  defp accumulated_text(slot) do
    slot.accumulated_pieces |> Enum.reverse() |> IO.iodata_to_binary()
  end

  # generate/generate_tokens reply with plain text; complete_tokens gets
  # completion metadata for the OpenAI-shaped API.
  defp completion_reply(%{reply_mode: :full} = slot, stop_reason) do
    %{
      text: accumulated_text(slot),
      completion_tokens: slot.tokens_generated,
      finish_reason: stop_reason
    }
  end

  defp completion_reply(slot, _stop_reason), do: accumulated_text(slot)

  # Error replies are shaped {:error, atom | {atom, detail}} across all
  # failure paths: :context_full, :queue_full, :empty_prompt, and
  # {:inference_failed, reason} here.
  defp fail_all_active_slots(state, reason) do
    active_slots =
      Enum.filter(state.slots, fn {_id, slot} -> slot.state != :idle end)

    Enum.reduce(active_slots, state, fn {seq_id, _slot}, state ->
      fail_slot(state, seq_id, {:inference_failed, reason})
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
end
