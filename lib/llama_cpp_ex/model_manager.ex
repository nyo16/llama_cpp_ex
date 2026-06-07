defmodule LlamaCppEx.ModelManager do
  @moduledoc """
  Holds multiple models resident and routes requests to them by id.

  The manager is a singleton `GenServer` that owns an ETS table of loaded
  models. Following the otp-thinking ETS pattern, **load/unload writes serialize
  through the GenServer, while inference-time lookups read the ETS table directly
  from the caller** — so the manager never becomes a throughput bottleneck for
  `generate/3`, `stream/3`, `chat/3`, or `embed/3`.

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

  @doc "Unloads a model and frees its backing resources (GC-based)."
  @spec unload(id()) :: :ok | {:error, :not_loaded}
  def unload(id), do: GenServer.call(__MODULE__, {:unload, id})

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
    case route(id) do
      {:ok, {:server, pid, e}} ->
        touch(e.id)
        LlamaCppEx.Server.generate(pid, prompt, opts)

      {:ok, {:direct, model, e}} ->
        touch(e.id)
        LlamaCppEx.generate(model, prompt, opts)

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Routes a streaming generation request to model `id` (or `:default`).

  Raises `ArgumentError` if the model is not loaded and ready (a lazy stream
  cannot carry an error tuple).
  """
  @spec stream(id(), String.t(), keyword()) :: Enumerable.t()
  def stream(id, prompt, opts \\ []) do
    case route(id) do
      {:ok, {:server, pid, e}} ->
        touch(e.id)
        LlamaCppEx.Server.stream(pid, prompt, opts)

      {:ok, {:direct, model, e}} ->
        touch(e.id)
        LlamaCppEx.stream(model, prompt, opts)

      {:error, reason} ->
        raise ArgumentError, "cannot stream from model #{inspect(id)}: #{inspect(reason)}"
    end
  end

  @doc "Routes a chat request to model `id` (or `:default`)."
  @spec chat(id(), [LlamaCppEx.Chat.message()], keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def chat(id, messages, opts \\ []) do
    case route(id) do
      {:ok, {:server, pid, e}} ->
        touch(e.id)

        {chat_opts, gen_opts} =
          Keyword.split(opts, [:add_assistant, :enable_thinking, :chat_template_kwargs])

        with model <- LlamaCppEx.Server.get_model(pid),
             {:ok, prompt} <- LlamaCppEx.Chat.apply_template(model, messages, chat_opts) do
          LlamaCppEx.Server.generate(pid, prompt, gen_opts)
        end

      {:ok, {:direct, model, e}} ->
        touch(e.id)
        LlamaCppEx.chat(model, messages, opts)

      {:error, _} = err ->
        err
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

  # --- Server callbacks ---

  @impl true
  def init(opts) do
    Process.flag(:trap_exit, true)

    table = :ets.new(@table, [:named_table, :protected, :set, read_concurrency: true])

    _lru =
      :ets.new(@lru, [
        :named_table,
        :public,
        :set,
        read_concurrency: true,
        write_concurrency: true
      ])

    state = %{
      table: table,
      io: Keyword.get(opts, :io, ModelIO),
      budget: Budget.resolve(Keyword.get(opts, :memory_budget, :infinity)),
      monitors: %{}
    }

    # Idempotent native backend init; harmless if already initialized.
    _ = safe_backend_init()

    {:ok, state, {:continue, {:autoload, Keyword.get(opts, :models, [])}}}
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
  def handle_call({:load, id, source, opts}, _from, state) do
    case do_load(state, id, source, opts) do
      {:ok, id, new_state} -> {:reply, {:ok, id}, new_state}
      {:error, reason, new_state} -> {:reply, {:error, reason}, new_state}
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
    if match?({:ok, _}, lookup_entry(id)) do
      :ets.insert(@table, {@default_key, id})
      {:reply, :ok, state}
    else
      {:reply, {:error, :not_loaded}, state}
    end
  end

  @impl true
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

  defp do_load(state, id, source, opts) do
    cond do
      match?({:ok, %Entry{status: status}} when status in [:ready, :loading], lookup_entry(id)) ->
        {:error, :already_loaded, state}

      true ->
        mode = resolve_mode(opts)
        capabilities = Keyword.get(opts, :capabilities, [:generate, :chat])

        with {:ok, path, file_bytes} <- state.io.resolve_source(source, opts),
             est = Budget.estimate(file_bytes, [{:mode, mode} | opts]),
             :ok <- Budget.check(state.budget, est, used_bytes(state)),
             {:ok, entry, state} <-
               start_backing(state, id, source, path, mode, capabilities, file_bytes, est, opts) do
          :ets.insert(@table, {id, entry})
          maybe_set_default(id, opts)
          {:ok, id, state}
        else
          {:error, reason} -> {:error, reason, state}
          {:error, reason, state} -> {:error, reason, state}
        end
    end
  end

  defp start_backing(state, id, source, path, :server, capabilities, file_bytes, est, opts) do
    case state.io.start_server(id, path, opts) do
      {:ok, pid} ->
        ref = Process.monitor(pid)

        entry = %Entry{
          id: id,
          status: :ready,
          mode: :server,
          model: nil,
          server_pid: pid,
          monitor_ref: ref,
          source: source,
          capabilities: capabilities,
          byte_size: file_bytes,
          est_bytes: est,
          n_gpu_layers: Keyword.get(opts, :n_gpu_layers, 99),
          loaded_at: System.system_time(:second)
        }

        {:ok, entry, %{state | monitors: Map.put(state.monitors, ref, id)}}

      {:error, reason} ->
        {:error, reason, state}
    end
  end

  defp start_backing(state, id, source, path, :direct, capabilities, file_bytes, est, opts) do
    case state.io.load_model(path, opts) do
      {:ok, model} ->
        entry = %Entry{
          id: id,
          status: :ready,
          mode: :direct,
          model: model,
          server_pid: nil,
          source: source,
          capabilities: capabilities,
          byte_size: file_bytes,
          est_bytes: est,
          n_gpu_layers: Keyword.get(opts, :n_gpu_layers, 99),
          loaded_at: System.system_time(:second)
        }

        {:ok, entry, state}

      {:error, reason} ->
        {:error, reason, state}
    end
  end

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

    :ets.delete(@table, entry.id)
    :ets.delete(@lru, entry.id)

    if default() == entry.id, do: :ets.delete(@table, @default_key)

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

  defp used_bytes(_state) do
    @table
    |> safe_tab2list()
    |> Enum.reduce(0, fn
      {_id, %Entry{status: status, est_bytes: bytes}}, acc when status in [:ready, :loading] ->
        acc + bytes

      _, acc ->
        acc
    end)
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

  defp safe_backend_init do
    LlamaCppEx.init()
  rescue
    _ -> :ok
  catch
    _, _ -> :ok
  end
end
