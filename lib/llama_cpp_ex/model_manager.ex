defmodule LlamaCppEx.ModelManager do
  @moduledoc """
  Holds multiple models resident and routes requests to them by id.

  The manager is a node-wide singleton `GenServer` that owns an ETS table of
  loaded models. Following the otp-thinking ETS pattern, **lifecycle writes
  serialize through the GenServer, while inference-time lookups read the ETS
  table directly from the caller** — so the manager never becomes a throughput
  bottleneck for `generate/3`, `stream/3`, `chat/3`, or `embed/3`.

  It is a singleton by design: the client API targets the manager by its module
  name, and the backing `Registry`/`DynamicSupervisor` use fixed names. Start at
  most one per node — `init/1` refuses a second instance with a clear error.

  Because the slow parts of `load/3` (Hub download + native model load) run in a
  supervised `Task` rather than the GenServer process, a long load does **not**
  block other lifecycle calls (`unload/1`, `set_default/1`, or concurrent
  `load/3`s). The memory-budget reservation and the ETS commit are still
  serialized on the GenServer, so resident models are always accounted for. (The
  budget remains advisory: a model's footprint is only reserved once its size is
  known — after resolve — so two models *downloading* at once may momentarily
  under-count each other.)

  Start it as part of `LlamaCppEx.ModelSupervisor`, which also starts the
  `Registry` and `DynamicSupervisor` that server-backed models need:

      children = [
        {LlamaCppEx.ModelSupervisor,
         memory_budget: :auto,
         models: [
           {"chat", {:hub, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"}, n_gpu_layers: -1},
           {"embed", {:path, "/models/nomic-embed.gguf"}, capabilities: [:embed]}
         ]}
      ]

  ## Backing modes

    * `:server` (default for generation/chat) — backs the model with a supervised
      `LlamaCppEx.Server`, giving continuous batching, streaming, prefix cache,
      and telemetry.
    * `:direct` (auto-selected when `:embed` is in `:capabilities`) — holds the
      `%LlamaCppEx.Model{}` and runs stateless `LlamaCppEx.generate/3` /
      `LlamaCppEx.Embedding.embed/2`. Mandatory for embeddings, since the server
      has no embedding path.

  ## Routing

    * Explicit id: `generate("chat", prompt)`.
    * Default model: `generate(:default, prompt)` routes to the model marked
      `default: true` at load, or set via `set_default/1`.

  ## Unloading and memory

  Model cleanup is GC-based: `unload/1` stops the backing server (dropping its
  context and model refs) and removes the ETS entry, then forces a GC. Because
  reclamation is by garbage collection, **any caller still holding a `%Model{}`
  obtained via `fetch_model/1` keeps the underlying model alive** past `unload/1`.
  Prefer id-based dispatch and avoid holding raw refs.

  Loads are checked against an advisory memory budget (see
  `LlamaCppEx.ModelManager.Budget`); over-budget loads are refused with
  `{:error, {:insufficient_memory, ...}}`.
  """

  use GenServer

  require Logger

  alias LlamaCppEx.ModelManager.{Budget, Entry, ModelIO}

  @table :llama_cpp_ex_models
  @lru :llama_cpp_ex_models_lru
  @default_key {:__meta__, :default}

  @typedoc """
  A model identifier. Any term works as a key; strings (e.g. `"chat"`) or atoms
  are conventional. Ids flow through as raw terms and are never converted to
  atoms, so user-supplied strings are safe.
  """
  @type id :: term()
  @type source :: Entry.source()

  # --- Client API: lifecycle (serialized through the GenServer) ---

  @doc """
  Starts the manager. Normally started by `LlamaCppEx.ModelSupervisor`.

  ## Options

    * `:memory_budget` - `:infinity` (default), `:auto` (~80% of system RAM), or
      a byte limit.
    * `:models` - List of `{id, source}` or `{id, source, opts}` to auto-load
      after start. `source` is `{:path, p}` or `{:hub, repo, file}`.
    * `:io` - Backend module (`LlamaCppEx.ModelManager.Backend`). Defaults to
      `LlamaCppEx.ModelManager.ModelIO`; overridden in tests.
    * `:name` - GenServer name. Defaults to `LlamaCppEx.ModelManager`.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Loads a model and keeps it resident under `id`.

  ## Options

    * `:mode` - `:server` or `:direct`. Defaults to `:direct` when
      `:capabilities` includes `:embed`, otherwise `:server`.
    * `:capabilities` - List of `:generate`, `:chat`, `:embed`. Defaults to
      `[:generate, :chat]`.
    * `:default` - When `true`, mark this model as the default route.
    * Hub options (`:cache_dir`, `:token`, `:revision`, `:force`) when `source`
      is `{:hub, repo, file}`.
    * Any `LlamaCppEx.Model.load/2` or `LlamaCppEx.Server.start_link/1` options
      (e.g. `:n_gpu_layers`, `:n_ctx`, `:n_parallel`).
  """
  @spec load(id(), source(), keyword()) :: {:ok, id()} | {:error, term()}
  def load(id, source, opts \\ []) do
    GenServer.call(__MODULE__, {:load, id, source, opts}, :infinity)
  end

  @doc """
  Unloads a model and frees its backing resources (GC-based).

  Stopping a backing server can take a moment for large models, so this accepts
  an optional `timeout` (default 30s) rather than the 5s `GenServer` default.
  """
  @spec unload(id(), timeout()) :: :ok | {:error, :not_loaded}
  def unload(id, timeout \\ 30_000), do: GenServer.call(__MODULE__, {:unload, id}, timeout)

  @doc "Sets the default model used by `:default` routing."
  @spec set_default(id()) :: :ok | {:error, :not_loaded}
  def set_default(id), do: GenServer.call(__MODULE__, {:set_default, id})

  # --- Client API: reads (bypass the GenServer, hit ETS directly) ---

  @doc "Lists resident models as sanitized maps (no raw refs)."
  @spec list() :: [map()]
  def list do
    @table
    |> safe_tab2list()
    |> Enum.flat_map(fn
      {_id, %Entry{} = e} -> [Entry.to_public(%{e | last_used: last_used(e.id)})]
      _ -> []
    end)
  end

  @doc "Returns a sanitized view of one model, or `{:error, :not_loaded}`."
  @spec info(id()) :: {:ok, map()} | {:error, :not_loaded}
  def info(id) do
    case lookup_entry(id) do
      {:ok, e} -> {:ok, Entry.to_public(%{e | last_used: last_used(id)})}
      {:error, _} = err -> err
    end
  end

  @doc "Returns whether a model is loaded and `:ready`."
  @spec loaded?(id()) :: boolean()
  def loaded?(id) do
    match?({:ok, %Entry{status: :ready}}, lookup_entry(id))
  end

  @doc """
  Returns the raw `%LlamaCppEx.Model{}` for advanced use.

  Holding the returned ref keeps the model alive past `unload/1` — prefer
  id-based dispatch where possible.
  """
  @spec fetch_model(id()) :: {:ok, LlamaCppEx.Model.t()} | {:error, term()}
  def fetch_model(id) do
    case route(id) do
      {:ok, {:server, pid, _e}} -> {:ok, LlamaCppEx.Server.get_model(pid)}
      {:ok, {:direct, model, _e}} -> {:ok, model}
      {:error, _} = err -> err
    end
  end

  @doc "Returns the current default model id, or `nil`."
  @spec default() :: id() | nil
  def default do
    case safe_lookup(@table, @default_key) do
      [{@default_key, id}] -> id
      _ -> nil
    end
  end

  # --- Client API: inference dispatch (bypass the GenServer) ---

  @doc """
  Routes a generation request to model `id` (or `:default`).

  Dispatches to `LlamaCppEx.Server.generate/3` (`:server` mode) or
  `LlamaCppEx.generate/3` (`:direct` mode).
  """
  @spec generate(id(), String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def generate(id, prompt, opts \\ []) do
    with_route(
      id,
      &LlamaCppEx.Server.generate(&1, prompt, opts),
      &LlamaCppEx.generate(&1, prompt, opts)
    )
  end

  @doc """
  Routes a streaming generation request to model `id` (or `:default`).

  Raises `ArgumentError` if the model is not loaded and ready (a lazy stream
  cannot carry an error tuple).
  """
  @spec stream(id(), String.t(), keyword()) :: Enumerable.t()
  def stream(id, prompt, opts \\ []) do
    case with_route(
           id,
           &LlamaCppEx.Server.stream(&1, prompt, opts),
           &LlamaCppEx.stream(&1, prompt, opts)
         ) do
      {:error, reason} ->
        raise ArgumentError, "cannot stream from model #{inspect(id)}: #{inspect(reason)}"

      stream ->
        stream
    end
  end

  @doc "Routes a chat request to model `id` (or `:default`)."
  @spec chat(id(), [LlamaCppEx.Chat.message()], keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def chat(id, messages, opts \\ []) do
    with_route(
      id,
      &server_chat(&1, messages, opts),
      &LlamaCppEx.chat(&1, messages, opts)
    )
  end

  @doc """
  Routes an OpenAI-shaped chat completion to model `id` (or `:default`).

  Server-backed models serve it through the batching server (prefix caching,
  `:session` affinity); direct models use the stateless path.
  """
  @spec chat_completion(id(), [LlamaCppEx.Chat.message()], keyword()) ::
          {:ok, LlamaCppEx.ChatCompletion.t()} | {:error, term()}
  def chat_completion(id, messages, opts \\ []) do
    with_route(
      id,
      &LlamaCppEx.chat_completion(&1, messages, opts),
      &LlamaCppEx.chat_completion(&1, messages, opts)
    )
  end

  @doc """
  Routes a streaming OpenAI-shaped chat completion to model `id` (or `:default`).

  Raises `ArgumentError` if the model is not loaded and ready (a lazy stream
  cannot carry an error tuple).
  """
  @spec stream_chat_completion(id(), [LlamaCppEx.Chat.message()], keyword()) :: Enumerable.t()
  def stream_chat_completion(id, messages, opts \\ []) do
    case with_route(
           id,
           &LlamaCppEx.stream_chat_completion(&1, messages, opts),
           &LlamaCppEx.stream_chat_completion(&1, messages, opts)
         ) do
      {:error, reason} ->
        raise ArgumentError,
              "cannot stream chat completion from model #{inspect(id)}: #{inspect(reason)}"

      stream ->
        stream
    end
  end

  # A server-backed model exposes generate/stream only, so chat templating
  # happens here before handing the rendered prompt to the server.
  defp server_chat(pid, messages, opts) do
    {chat_opts, gen_opts} =
      Keyword.split(opts, [:add_assistant, :enable_thinking, :chat_template_kwargs])

    model = LlamaCppEx.Server.get_model(pid)

    with {:ok, prompt} <- LlamaCppEx.Chat.apply_template(model, messages, chat_opts) do
      LlamaCppEx.Server.generate(pid, prompt, gen_opts)
    end
  end

  @doc """
  Routes an embedding request to model `id` (or `:default`).

  The model must have been loaded with `:embed` in its `:capabilities` (which
  forces `:direct` mode).
  """
  @spec embed(id(), String.t(), keyword()) :: {:ok, [float()]} | {:error, term()}
  def embed(id, text, opts \\ []) do
    case route(id) do
      {:ok, {:direct, model, e}} ->
        if :embed in e.capabilities do
          touch(e.id)
          LlamaCppEx.Embedding.embed(model, text, opts)
        else
          {:error, :not_embedding_model}
        end

      {:ok, {:server, _pid, _e}} ->
        {:error, :not_embedding_model}

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Resolves `id` (or `:default`) to its dispatch target.

  Returns `{:ok, {:server, pid, entry}}`, `{:ok, {:direct, model, entry}}`, or
  `{:error, :not_loaded | {:not_ready, status}}`. Primarily for dispatch and
  testing.
  """
  @spec route(id()) ::
          {:ok, {:server, pid(), Entry.t()}}
          | {:ok, {:direct, LlamaCppEx.Model.t(), Entry.t()}}
          | {:error, term()}
  def route(id) do
    with {:ok, e} <- lookup_entry(resolve_id(id)),
         {:ok, e} <- ready(e) do
      case e.mode do
        :server -> {:ok, {:server, e.server_pid, e}}
        :direct -> {:ok, {:direct, e.model, e}}
      end
    end
  end

  # Shared dispatch skeleton: resolve the route, mark the model used, and hand
  # the server pid or direct model to the matching callback. Errors pass through.
  defp with_route(id, server_fun, direct_fun) do
    case route(id) do
      {:ok, {:server, pid, e}} ->
        touch(e.id)
        server_fun.(pid)

      {:ok, {:direct, model, e}} ->
        touch(e.id)
        direct_fun.(model)

      {:error, _} = err ->
        err
    end
  end

  # --- Server callbacks ---

  @impl true
  def init(opts) do
    if :ets.whereis(@table) != :undefined do
      {:stop,
       "LlamaCppEx.ModelManager is a node-wide singleton and is already running " <>
         "(it owns the #{inspect(@table)} ETS table). Start at most one per node."}
    else
      Process.flag(:trap_exit, true)

      # A previous manager incarnation may have left backing servers behind under
      # the (process-independent) DynamicSupervisor; reclaim their VRAM now that
      # our ETS table starts empty. No-op when started standalone (no DynSup).
      cleanup_orphaned_servers()

      # Supervises the per-load tasks that run the slow resolve/native-load off
      # the GenServer process. Linked here so it dies with the manager.
      {:ok, task_sup} = Task.Supervisor.start_link()

      table = :ets.new(@table, [:named_table, :protected, :set, read_concurrency: true])

      _lru =
        :ets.new(@lru, [
          :named_table,
          :public,
          :set,
          read_concurrency: true,
          write_concurrency: true
        ])

      # Idempotent native backend init; harmless if already initialized.
      _ = safe_backend_init()

      gpu_devices = gpu_devices()

      state = %{
        table: table,
        io: Keyword.get(opts, :io, ModelIO),
        budget: Budget.resolve(Keyword.get(opts, :memory_budget, :infinity), gpu_devices),
        n_gpus: length(gpu_devices),
        monitors: %{},
        task_sup: task_sup,
        loads: %{}
      }

      {:ok, state, {:continue, {:autoload, Keyword.get(opts, :models, [])}}}
    end
  end

  @impl true
  def handle_continue({:autoload, models}, state) do
    state =
      Enum.reduce(models, state, fn spec, acc ->
        {id, source, opts} = normalize_spec(spec)

        case do_load(acc, id, source, opts) do
          {:ok, _id, new_state} ->
            new_state

          {:error, reason, new_state} ->
            Logger.warning("ModelManager auto-load of #{inspect(id)} failed: #{inspect(reason)}")
            new_state
        end
      end)

    {:noreply, state}
  end

  @impl true
  def handle_call({:load, id, source, opts}, from, state) do
    case lookup_entry(id) do
      {:ok, %Entry{status: status}} when status in [:ready, :loading] ->
        {:reply, {:error, :already_loaded}, state}

      _ ->
        {:noreply, start_async_load(state, id, source, opts, from)}
    end
  end

  def handle_call({:unload, id}, _from, state) do
    case lookup_entry(id) do
      {:ok, entry} ->
        {:reply, :ok, do_unload(state, entry)}

      {:error, _} ->
        {:reply, {:error, :not_loaded}, state}
    end
  end

  def handle_call({:set_default, id}, _from, state) do
    case lookup_entry(id) do
      {:ok, _entry} ->
        :ets.insert(@table, {@default_key, id})
        {:reply, :ok, state}

      {:error, _} ->
        {:reply, {:error, :not_loaded}, state}
    end
  end

  # Serialized memory-budget reservation, called by a load task once it knows the
  # model's placement. Recording the placement on the :loading entry means
  # concurrent reservations account for it, so the budget can't be oversubscribed.
  def handle_call({:reserve, id, placement}, _from, state) do
    case Budget.check(state.budget, placement, used_placement(state)) do
      :ok ->
        case lookup_entry(id) do
          {:ok, entry} ->
            :ets.insert(
              @table,
              {id, %{entry | placement: placement, est_bytes: placement_total(placement)}}
            )

          {:error, _} ->
            :ok
        end

        {:reply, :ok, state}

      {:error, _} = error ->
        {:reply, error, state}
    end
  end

  @impl true
  # A load task finished. async_nolink delivers {ref, result}; flush the paired
  # :DOWN so it doesn't fall through to the server-monitor clause below.
  def handle_info({ref, result}, state) when is_map_key(state.loads, ref) do
    Process.demonitor(ref, [:flush])
    {load, loads} = Map.pop(state.loads, ref)
    {:noreply, finalize_load(%{state | loads: loads}, load, result)}
  end

  # A load task crashed before returning a result.
  def handle_info({:DOWN, ref, :process, _pid, reason}, state)
      when is_map_key(state.loads, ref) do
    {load, loads} = Map.pop(state.loads, ref)
    :ets.delete(@table, load.id)
    reply_load(load.from, {:error, {:load_crashed, reason}})
    {:noreply, %{state | loads: loads}}
  end

  def handle_info({:DOWN, ref, :process, pid, reason}, state) do
    case Map.pop(state.monitors, ref) do
      {nil, _} ->
        {:noreply, state}

      {id, monitors} ->
        Logger.warning(
          "ModelManager: backing server for #{inspect(id)} (#{inspect(pid)}) went down: " <>
            "#{inspect(reason)}; marking :error"
        )

        case lookup_entry(id) do
          {:ok, entry} ->
            :ets.insert(
              @table,
              {id, %{entry | status: :error, server_pid: nil, monitor_ref: nil, error: reason}}
            )

          {:error, _} ->
            :ok
        end

        {:noreply, %{state | monitors: monitors}}
    end
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # --- Load / unload internals ---

  # Synchronous load used only by boot-time autoload (`handle_continue`), where
  # blocking the manager until models are resident is acceptable. Interactive
  # `load/3` goes through the async path (`start_async_load/5`) instead.
  defp do_load(state, id, source, opts) do
    case lookup_entry(id) do
      {:ok, %Entry{status: status}} when status in [:ready, :loading] ->
        {:error, :already_loaded, state}

      _ ->
        mode = resolve_mode(opts)
        capabilities = Keyword.get(opts, :capabilities, [:generate, :chat])

        with {:ok, path, file_bytes} <- state.io.resolve_source(source, opts),
             placement = Budget.distribute(file_bytes, [{:mode, mode} | opts], state.n_gpus),
             :ok <- Budget.check(state.budget, placement, used_placement(state)),
             {:ok, backing} <- start_backing_io(state.io, id, path, mode, opts) do
          {entry, state} =
            build_ready_entry(
              state,
              id,
              source,
              backing,
              file_bytes,
              placement,
              capabilities,
              opts
            )

          :ets.insert(@table, {id, entry})
          maybe_set_default(id, opts)
          {:ok, id, state}
        else
          {:error, reason} -> {:error, reason, state}
        end
    end
  end

  # --- Async interactive load ---

  defp start_async_load(state, id, source, opts, from) do
    mode = resolve_mode(opts)
    capabilities = Keyword.get(opts, :capabilities, [:generate, :chat])

    # Claim the id synchronously so a concurrent load of the same id is rejected
    # while this one is in flight.
    :ets.insert(
      @table,
      {id,
       %Entry{id: id, status: :loading, mode: mode, source: source, capabilities: capabilities}}
    )

    manager = self()

    task =
      Task.Supervisor.async_nolink(state.task_sup, fn ->
        run_load(manager, state.io, id, source, opts, mode, state.n_gpus)
      end)

    load = %{id: id, source: source, capabilities: capabilities, opts: opts, from: from}
    %{state | loads: Map.put(state.loads, task.ref, load)}
  end

  # Runs in the task process: the slow resolve + native load, with a serialized
  # budget reservation in between, keeping the GenServer mailbox free.
  defp run_load(manager, io, id, source, opts, mode, n_gpus) do
    with {:ok, path, file_bytes} <- io.resolve_source(source, opts),
         placement = Budget.distribute(file_bytes, [{:mode, mode} | opts], n_gpus),
         :ok <- GenServer.call(manager, {:reserve, id, placement}, :infinity),
         {:ok, backing} <- start_backing_io(io, id, path, mode, opts) do
      {:ok, backing, file_bytes, placement}
    end
  end

  defp finalize_load(state, load, {:ok, backing, file_bytes, placement}) do
    {entry, state} =
      build_ready_entry(
        state,
        load.id,
        load.source,
        backing,
        file_bytes,
        placement,
        load.capabilities,
        load.opts
      )

    :ets.insert(@table, {load.id, entry})
    maybe_set_default(load.id, load.opts)
    reply_load(load.from, {:ok, load.id})
    state
  end

  defp finalize_load(state, load, {:error, reason}) do
    # Drop the :loading placeholder, releasing any reservation it held.
    :ets.delete(@table, load.id)
    reply_load(load.from, {:error, reason})
    state
  end

  defp reply_load(nil, _msg), do: :ok
  defp reply_load(from, msg), do: GenServer.reply(from, msg)

  # --- Backing start (shared by the sync and async load paths) ---

  defp start_backing_io(io, _id, path, :direct, opts) do
    case io.load_model(path, opts) do
      {:ok, model} -> {:ok, {:direct, model}}
      {:error, _} = err -> err
    end
  end

  defp start_backing_io(io, id, path, :server, opts) do
    case io.start_server(id, path, opts) do
      {:ok, pid} -> {:ok, {:server, pid}}
      {:error, _} = err -> err
    end
  end

  defp build_ready_entry(state, id, source, backing, file_bytes, placement, capabilities, opts) do
    base = [
      id: id,
      status: :ready,
      source: source,
      capabilities: capabilities,
      byte_size: file_bytes,
      est_bytes: placement_total(placement),
      placement: placement,
      n_gpu_layers: Keyword.get(opts, :n_gpu_layers, 99),
      loaded_at: System.system_time(:second)
    ]

    case backing do
      {:server, pid} ->
        ref = Process.monitor(pid)
        entry = struct!(Entry, [mode: :server, server_pid: pid, monitor_ref: ref] ++ base)
        {entry, %{state | monitors: Map.put(state.monitors, ref, id)}}

      {:direct, model} ->
        {struct!(Entry, [mode: :direct, model: model] ++ base), state}
    end
  end

  defp placement_total(%{ram: ram, vram: vram}), do: ram + Enum.sum(Map.values(vram))

  defp do_unload(state, %Entry{} = entry) do
    state =
      case entry do
        %Entry{mode: :server, server_pid: pid, monitor_ref: ref} when is_pid(pid) ->
          if ref, do: Process.demonitor(ref, [:flush])
          _ = state.io.stop_server(pid)
          %{state | monitors: Map.delete(state.monitors, ref)}

        _ ->
          state
      end

    # Clear the default pointer before the entry row, so a crash mid-unload can't
    # leave @default_key referencing a removed model.
    if default() == entry.id, do: :ets.delete(@table, @default_key)

    :ets.delete(@table, entry.id)
    :ets.delete(@lru, entry.id)

    :erlang.garbage_collect()
    state
  end

  # --- Helpers ---

  defp resolve_mode(opts) do
    capabilities = Keyword.get(opts, :capabilities, [:generate, :chat])

    cond do
      Keyword.get(opts, :mode) in [:server, :direct] -> Keyword.get(opts, :mode)
      :embed in capabilities -> :direct
      true -> :server
    end
  end

  defp maybe_set_default(id, opts) do
    if Keyword.get(opts, :default, false), do: :ets.insert(@table, {@default_key, id})
  end

  defp used_placement(_state) do
    @table
    |> safe_tab2list()
    |> Enum.reduce(Budget.empty_usage(), fn
      {_id, %Entry{status: status, placement: placement}}, acc
      when status in [:ready, :loading] ->
        Budget.add_usage(acc, placement)

      _, acc ->
        acc
    end)
  end

  defp gpu_devices do
    LlamaCppEx.devices()
    |> Enum.filter(&(&1.type in [:gpu, :igpu]))
  rescue
    # The device NIF isn't loaded (test env / unsupported platform): degrade to
    # "no GPUs" so budgeting treats everything as RAM. Other errors propagate.
    _ in [ErlangError, UndefinedFunctionError] -> []
  end

  # Reclaim backing servers left under the (process-independent) DynamicSupervisor
  # by a previous manager incarnation. Our ETS table starts empty on init, so any
  # surviving children are orphans holding VRAM with no owner. No-op when started
  # standalone (e.g. tests), where the DynamicSupervisor isn't running.
  defp cleanup_orphaned_servers do
    dynsup = ModelIO.dynamic_supervisor()

    if is_pid(Process.whereis(dynsup)) do
      for {_, pid, _, _} <- DynamicSupervisor.which_children(dynsup), is_pid(pid) do
        DynamicSupervisor.terminate_child(dynsup, pid)
      end
    end

    :ok
  end

  defp normalize_spec({id, source, opts}), do: {id, source, opts}
  defp normalize_spec({id, source}), do: {id, source, []}

  defp resolve_id(:default), do: default()
  defp resolve_id(id), do: id

  defp lookup_entry(nil), do: {:error, :not_loaded}

  defp lookup_entry(id) do
    case safe_lookup(@table, id) do
      [{^id, %Entry{} = e}] -> {:ok, e}
      _ -> {:error, :not_loaded}
    end
  end

  defp ready(%Entry{status: :ready} = e), do: {:ok, e}
  defp ready(%Entry{status: status}), do: {:error, {:not_ready, status}}

  defp touch(id), do: :ets.insert(@lru, {id, System.monotonic_time(:millisecond)})

  defp last_used(id) do
    case safe_lookup(@lru, id) do
      [{^id, ts}] -> ts
      _ -> 0
    end
  end

  defp safe_lookup(table, key) do
    :ets.lookup(table, key)
  rescue
    ArgumentError -> []
  end

  defp safe_tab2list(table) do
    :ets.tab2list(table)
  rescue
    ArgumentError -> []
  end

  # Best-effort: the NIF may be missing entirely (test env / unsupported
  # platform) — degrade to :ok like gpu_devices/0, but log what was skipped.
  # init/0 is a thin NIF call, so only these two exceptions can occur;
  # anything else propagates.
  defp safe_backend_init do
    LlamaCppEx.init()
  rescue
    e in [ErlangError, UndefinedFunctionError] ->
      Logger.debug("backend init skipped: #{Exception.message(e)}")
      :ok
  end
end
