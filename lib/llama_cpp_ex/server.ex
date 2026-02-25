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
    * Sampling options: `:temp`, `:top_k`, `:top_p`, `:min_p`, `:seed`, `:penalty_repeat`,
      `:penalty_freq`, `:penalty_present`, `:grammar`, `:grammar_root`.
    * GenServer options like `:name`.

  """
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
      Keyword.take(opts, [:main_gpu, :split_mode, :tensor_split, :use_mlock, :use_direct_io])

    :ok = LlamaCppEx.init()
    {:ok, model} = Model.load(model_path, [n_gpu_layers: n_gpu_layers] ++ model_opts)

    {:ok, ctx} =
      Context.create(model,
        n_ctx: n_ctx,
        n_batch: n_batch,
        n_seq_max: n_parallel
      )

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
          n_prompt_tokens: 0
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
      chunk_size: chunk_size
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:generate, prompt, max_tokens}, from, state) do
    case acquire_slot(state) do
      {:ok, seq_id, state} ->
        state = init_slot(state, seq_id, prompt, max_tokens, from, nil, nil)
        state = maybe_schedule_tick(state)
        {:noreply, state}

      :no_slots ->
        state = enqueue_request(state, {:generate, prompt, max_tokens, from, nil, nil})
        {:noreply, state}
    end
  end

  def handle_call({:stream, prompt, max_tokens, pid, ref}, from, state) do
    case acquire_slot(state) do
      {:ok, seq_id, state} ->
        state = init_slot(state, seq_id, prompt, max_tokens, nil, pid, ref)
        GenServer.reply(from, :ok)
        state = maybe_schedule_tick(state)
        {:noreply, state}

      :no_slots ->
        GenServer.reply(from, :ok)
        state = enqueue_request(state, {:stream, prompt, max_tokens, nil, pid, ref})
        {:noreply, state}
    end
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

  defp acquire_slot(state) do
    case Enum.find(state.slots, fn {_id, slot} -> slot.state == :idle end) do
      {seq_id, _slot} -> {:ok, seq_id, state}
      nil -> :no_slots
    end
  end

  defp init_slot(state, seq_id, prompt, max_tokens, from, stream_pid, stream_ref) do
    {:ok, tokens} = Tokenizer.encode(state.model, prompt)

    # Clear KV cache for this sequence
    LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)

    # Reset sampler for fresh generation
    slot = state.slots[seq_id]
    Sampler.reset(slot.sampler)

    slot = %{
      slot
      | state: :prefilling,
        from: from,
        stream_pid: stream_pid,
        stream_ref: stream_ref,
        prompt_tokens: tokens,
        prefill_pos: 0,
        pos: 0,
        pending_token: nil,
        batch_idx: -1,
        tokens_generated: 0,
        max_tokens: max_tokens,
        accumulated_text: "",
        t_start: System.monotonic_time(),
        t_first_token: nil,
        n_prompt_tokens: length(tokens)
    }

    put_in(state.slots[seq_id], slot)
  end

  defp enqueue_request(state, request) do
    %{state | queue: :queue.in(request, state.queue)}
  end

  defp dequeue_into_slot(state) do
    case :queue.out(state.queue) do
      {{:value, request}, queue} ->
        state = %{state | queue: queue}

        case acquire_slot(state) do
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

  defp assign_queued_request(state, seq_id, {:generate, prompt, max_tokens, from, _, _}) do
    init_slot(state, seq_id, prompt, max_tokens, from, nil, nil)
  end

  defp assign_queued_request(state, seq_id, {:stream, prompt, max_tokens, _, pid, ref}) do
    init_slot(state, seq_id, prompt, max_tokens, nil, pid, ref)
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
      # Count decode tokens before sampling (slots may transition after)
      n_decode =
        Enum.count(state.slots, fn {_id, s} ->
          s.state == :generating and s.batch_idx >= 0
        end)

      # Phase 3: Forward pass
      tick_start = System.monotonic_time()
      :ok = LlamaCppEx.NIF.batch_eval(state.ctx.ref, entries)
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

      if is_eog or slot.tokens_generated >= slot.max_tokens do
        # pending_token is the NEXT token to process:
        # - if EOG: don't stream (it's a control token)
        # - if max_tokens reached: don't stream (it's beyond our limit)
        finish_slot(state, seq_id)
      else
        state
      end
    end)
  end

  # Phase 2: Build batch with decode tokens first, then prefill chunks
  defp build_batch(state) do
    budget = state.n_batch

    # 2a: Decode tokens (priority)
    {entries, state, budget} = add_decode_tokens(state, [], budget)

    # 2b: Prefill chunks (fill remaining budget)
    {entries, state, _budget} = add_prefill_chunks(state, entries, budget)

    {Enum.reverse(entries), state}
  end

  defp add_decode_tokens(state, entries, budget) do
    generating_slots =
      state.slots
      |> Enum.filter(fn {_id, slot} ->
        slot.state == :generating and slot.pending_token != nil
      end)
      |> Enum.sort_by(&elem(&1, 0))

    Enum.reduce(generating_slots, {entries, state, budget}, fn {seq_id, _slot},
                                                               {entries, state, budget} ->
      if budget <= 0 do
        {entries, state, budget}
      else
        slot = state.slots[seq_id]
        token = slot.pending_token

        # Detokenize and stream/accumulate the pending token
        piece = LlamaCppEx.NIF.token_to_piece(state.model.ref, token)

        if slot.stream_pid && slot.stream_ref do
          send(slot.stream_pid, {slot.stream_ref, {:token, piece}})
        end

        batch_idx = length(entries)

        slot = %{
          slot
          | accumulated_text: slot.accumulated_text <> piece,
            batch_idx: batch_idx,
            tokens_generated: slot.tokens_generated + 1
        }

        # Record first token time
        slot =
          if slot.t_first_token == nil do
            %{slot | t_first_token: System.monotonic_time()}
          else
            slot
          end

        entry = {token, slot.pos, seq_id, true}
        state = put_in(state.slots[seq_id], slot)

        {[entry | entries], state, budget - 1}
      end
    end)
  end

  defp add_prefill_chunks(state, entries, budget) do
    prefilling_slots =
      state.slots
      |> Enum.filter(fn {_id, slot} -> slot.state == :prefilling end)
      |> Enum.sort_by(&elem(&1, 0))

    Enum.reduce(prefilling_slots, {entries, state, budget}, fn {seq_id, _slot},
                                                               {entries, state, budget} ->
      if budget <= 0 do
        {entries, state, budget}
      else
        slot = state.slots[seq_id]
        remaining = length(slot.prompt_tokens) - slot.prefill_pos
        chunk_len = min(budget, min(state.chunk_size, remaining))
        is_last_chunk = slot.prefill_pos + chunk_len >= length(slot.prompt_tokens)

        chunk_tokens = Enum.slice(slot.prompt_tokens, slot.prefill_pos, chunk_len)

        # Add chunk tokens to entries
        {new_entries, last_batch_idx} =
          chunk_tokens
          |> Enum.with_index()
          |> Enum.reduce({entries, -1}, fn {token, i}, {entries, _last_idx} ->
            pos = slot.prefill_pos + i
            batch_idx = length(entries)
            is_last_token_of_last_chunk = is_last_chunk and i == chunk_len - 1
            logits = is_last_token_of_last_chunk
            entry = {token, pos, seq_id, logits}
            {[entry | entries], batch_idx}
          end)

        slot =
          if is_last_chunk do
            %{slot | batch_idx: last_batch_idx, prefill_pos: slot.prefill_pos + chunk_len}
          else
            %{slot | batch_idx: -1, prefill_pos: slot.prefill_pos + chunk_len}
          end

        state = put_in(state.slots[seq_id], slot)
        {new_entries, state, budget - chunk_len}
      end
    end)
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
          prompt_tokens: [],
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

  defp finish_slot(state, seq_id) do
    slot = state.slots[seq_id]
    t_end = System.monotonic_time()

    if slot.from do
      GenServer.reply(slot.from, {:ok, slot.accumulated_text})
    end

    if slot.stream_pid && slot.stream_ref do
      send(slot.stream_pid, {slot.stream_ref, :done})
    end

    # Emit telemetry
    emit_request_done(slot, seq_id, t_end)

    reset_slot(state, seq_id)
  end

  defp emit_request_done(slot, seq_id, t_end) do
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

    mode = if slot.stream_pid, do: :stream, else: :generate

    Logger.debug(
      "slot #{seq_id} done: #{slot.n_prompt_tokens} prompt tokens (#{Float.round(prompt_eval_rate, 1)} t/s), " <>
        "#{slot.tokens_generated} generated (#{Float.round(generation_rate, 1)} t/s), " <>
        "ttft #{Float.round(ttft_ms, 1)}ms, total #{Float.round(duration_ms, 1)}ms"
    )

    :telemetry.execute(
      [:llama_cpp_ex, :server, :request, :done],
      %{
        prompt_tokens: slot.n_prompt_tokens,
        generated_tokens: slot.tokens_generated,
        duration_ms: duration_ms,
        ttft_ms: ttft_ms,
        prompt_eval_rate: prompt_eval_rate,
        generation_rate: generation_rate
      },
      %{server: self(), seq_id: seq_id, mode: mode}
    )
  end

  defp reset_slot(state, seq_id) do
    slot = state.slots[seq_id]
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
        n_prompt_tokens: 0
    }

    put_in(state.slots[seq_id], slot)
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
