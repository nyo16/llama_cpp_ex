defmodule LlamaCppEx.Server do
  @moduledoc """
  GenServer for batched multi-sequence inference.

  Manages a shared model/context and serves multiple concurrent callers
  using a slot pool, inspired by llama-server's continuous batching.

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

  alias LlamaCppEx.{Model, Context, Sampler, Tokenizer}

  defstruct [
    :model,
    :ctx,
    :sampler_opts,
    slots: %{},
    n_parallel: 4,
    tick_scheduled: false
  ]

  # --- Client API ---

  @doc """
  Starts the server.

  ## Options

    * `:model_path` (required) - Path to the GGUF model file.
    * `:n_gpu_layers` - GPU layers. Defaults to `0`.
    * `:n_ctx` - Total context size (shared across slots). Defaults to `8192`.
    * `:n_parallel` - Number of concurrent slots. Defaults to `4`.
    * `:n_batch` - Batch size. Defaults to `n_ctx`.
    * Sampling options: `:temp`, `:top_k`, `:top_p`, `:min_p`, `:seed`, `:penalty_repeat`,
      `:grammar`, `:grammar_root`.
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

  # --- Server callbacks ---

  @impl true
  def init(opts) do
    model_path = Keyword.fetch!(opts, :model_path)
    n_gpu_layers = Keyword.get(opts, :n_gpu_layers, 0)
    n_parallel = Keyword.get(opts, :n_parallel, 4)
    n_ctx = Keyword.get(opts, :n_ctx, 8192)
    n_batch = Keyword.get(opts, :n_batch, n_ctx)

    sampler_opts =
      Keyword.take(opts, [
        :seed,
        :temp,
        :top_k,
        :top_p,
        :min_p,
        :penalty_repeat,
        :grammar,
        :grammar_root
      ])

    :ok = LlamaCppEx.init()
    {:ok, model} = Model.load(model_path, n_gpu_layers: n_gpu_layers)

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
          tokens_generated: 0,
          max_tokens: 0,
          pos: 0,
          pending_token: nil,
          accumulated_text: ""
        }

        {seq_id, slot}
      end

    state = %__MODULE__{
      model: model,
      ctx: ctx,
      sampler_opts: sampler_opts,
      slots: slots,
      n_parallel: n_parallel
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:generate, prompt, max_tokens}, from, state) do
    case acquire_slot(state) do
      {:ok, seq_id, state} ->
        state = start_slot(state, seq_id, prompt, max_tokens, from, nil, nil)
        {:noreply, state}

      :no_slots ->
        {:reply, {:error, :no_slots}, state}
    end
  end

  def handle_call({:stream, prompt, max_tokens, pid, ref}, from, state) do
    case acquire_slot(state) do
      {:ok, seq_id, state} ->
        state = start_slot(state, seq_id, prompt, max_tokens, nil, pid, ref)
        GenServer.reply(from, :ok)
        {:noreply, state}

      :no_slots ->
        {:reply, {:error, :no_slots}, state}
    end
  end

  @impl true
  def handle_info(:tick, state) do
    state = %{state | tick_scheduled: false}
    state = run_tick(state)
    {:noreply, state}
  end

  # --- Internal ---

  defp acquire_slot(state) do
    case Enum.find(state.slots, fn {_id, slot} -> slot.state == :idle end) do
      {seq_id, _slot} -> {:ok, seq_id, state}
      nil -> :no_slots
    end
  end

  defp start_slot(state, seq_id, prompt, max_tokens, from, stream_pid, stream_ref) do
    {:ok, tokens} = Tokenizer.encode(state.model, prompt)

    # Clear KV cache for this sequence
    LlamaCppEx.NIF.memory_seq_rm(state.ctx.ref, seq_id, 0, -1)

    # Reset sampler for fresh generation
    slot = state.slots[seq_id]
    Sampler.reset(slot.sampler)

    # Prefill prompt tokens
    case LlamaCppEx.NIF.prefill(state.ctx.ref, tokens, seq_id) do
      {:ok, n_past} ->
        # Sample first token IMMEDIATELY (logits are fresh from prefill)
        first_token = LlamaCppEx.NIF.sampler_sample(slot.sampler.ref, state.ctx.ref)
        LlamaCppEx.NIF.sampler_accept(slot.sampler.ref, first_token)

        slot = %{
          slot
          | state: :generating,
            from: from,
            stream_pid: stream_pid,
            stream_ref: stream_ref,
            tokens_generated: 0,
            max_tokens: max_tokens,
            pos: n_past,
            pending_token: first_token,
            accumulated_text: ""
        }

        state = put_in(state.slots[seq_id], slot)
        maybe_schedule_tick(state)

      {:error, reason} ->
        reply_error(from, stream_pid, stream_ref, reason)
        state
    end
  end

  defp run_tick(state) do
    active_slots =
      state.slots
      |> Enum.filter(fn {_id, slot} -> slot.state == :generating end)
      |> Enum.sort_by(&elem(&1, 0))

    if active_slots == [] do
      state
    else
      # Process each slot sequentially (decode + sample is atomic per slot)
      state = Enum.reduce(active_slots, state, &process_slot/2)

      # Schedule another tick if any slots still generating
      if Enum.any?(state.slots, fn {_id, slot} -> slot.state == :generating end) do
        maybe_schedule_tick(state)
      else
        state
      end
    end
  end

  defp process_slot({seq_id, _slot}, state) do
    slot = state.slots[seq_id]
    token = slot.pending_token
    is_eog = LlamaCppEx.NIF.vocab_is_eog(state.model.ref, token)

    if is_eog or slot.tokens_generated >= slot.max_tokens do
      finish_slot(state, seq_id)
    else
      # Detokenize and stream/accumulate the pending token
      piece = LlamaCppEx.NIF.token_to_piece(state.model.ref, token)

      if slot.stream_pid && slot.stream_ref do
        send(slot.stream_pid, {slot.stream_ref, {:token, piece}})
      end

      # Decode the pending token (adds to KV cache, produces new logits)
      case LlamaCppEx.NIF.decode_token(state.ctx.ref, token, slot.pos, seq_id) do
        :ok ->
          # Sample next token IMMEDIATELY after decode
          next_token = LlamaCppEx.NIF.sampler_sample(slot.sampler.ref, state.ctx.ref)
          LlamaCppEx.NIF.sampler_accept(slot.sampler.ref, next_token)

          slot = %{
            slot
            | tokens_generated: slot.tokens_generated + 1,
              pos: slot.pos + 1,
              pending_token: next_token,
              accumulated_text: slot.accumulated_text <> piece
          }

          put_in(state.slots[seq_id], slot)

        {:error, reason} ->
          finish_slot_with_error(state, seq_id, reason)
      end
    end
  end

  defp finish_slot(state, seq_id) do
    slot = state.slots[seq_id]

    if slot.from do
      GenServer.reply(slot.from, {:ok, slot.accumulated_text})
    end

    if slot.stream_pid && slot.stream_ref do
      send(slot.stream_pid, {slot.stream_ref, :done})
    end

    reset_slot(state, seq_id)
  end

  defp finish_slot_with_error(state, seq_id, reason) do
    slot = state.slots[seq_id]
    reply_error(slot.from, slot.stream_pid, slot.stream_ref, reason)
    reset_slot(state, seq_id)
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
        tokens_generated: 0,
        max_tokens: 0,
        pos: 0,
        pending_token: nil,
        accumulated_text: ""
    }

    put_in(state.slots[seq_id], slot)
  end

  defp reply_error(from, stream_pid, stream_ref, reason) do
    if from, do: GenServer.reply(from, {:error, reason})

    if stream_pid && stream_ref do
      send(stream_pid, {stream_ref, {:error, reason}})
    end
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
