defmodule LlamaCppEx.ModelManager.ModelIO do
  @moduledoc """
  Default `LlamaCppEx.ModelManager.Backend` implementation.

  Delegates to `LlamaCppEx.Hub` for downloads, `LlamaCppEx.Model` for direct
  loading, and `LlamaCppEx.Server` (under the manager's `DynamicSupervisor`,
  named via the manager's `Registry`) for server-backed models.
  """

  @behaviour LlamaCppEx.ModelManager.Backend

  alias LlamaCppEx.{Hub, Model, Options, Server}

  @registry LlamaCppEx.ModelRegistry
  @dynsup LlamaCppEx.ModelDynSup

  # Hub-only options, stripped before handing opts to Model.load / Server.
  @hub_keys [:cache_dir, :token, :revision, :force, :progress]

  # Manager-only options, never forwarded to the native layer.
  @manager_keys [:mode, :capabilities, :default, :memory_budget, :io]

  # What each destination actually reads — allowlists, not a `Keyword.drop/2`
  # denylist.
  #
  # The denylist named the Hub and manager keys and let everything else through,
  # so `:vocab_only` — which `LlamaCppEx.Model` classifies as structural
  # precisely because it must not be forwarded blindly — reached
  # `Server.start_link/1` and raised inside `init/1`, even though `load/3`
  # documents accepting "any `Model.load/2` or `Server.start_link/1` options".
  # A denylist can only be complete for the option set that existed when it was
  # written. `Server.start_option_keys/0` is the server's own list, for the same
  # reason.
  @model_load_keys Model.tuning_option_keys() ++ Model.structural_option_keys()

  # The denylist doubled as the typo gate: forwarding everything it did not
  # recognise meant `Server.start_link/1`'s own validation caught `n_paralell`.
  # Two allowlists on their own would have silently ignored it — a server quietly
  # running with the default `:n_parallel` is exactly the class of bug the
  # option-forwarding work exists to remove. So the union is validated here,
  # naming the function the caller actually called, and each destination then
  # takes only the keys it reads.
  @known_keys Enum.uniq(
                @hub_keys ++ @manager_keys ++ @model_load_keys ++ Server.start_option_keys()
              )

  @doc "Every option `LlamaCppEx.ModelManager.load/3` accepts for this backend."
  @spec option_keys() :: [atom()]
  def option_keys, do: @known_keys

  @doc "The registry name backing servers are registered under."
  def registry, do: @registry

  @doc "The dynamic supervisor name backing servers run under."
  def dynamic_supervisor, do: @dynsup

  @impl true
  def resolve_source({:path, path}, _opts) do
    case File.stat(path) do
      {:ok, %{size: size}} -> {:ok, path, size}
      {:error, reason} -> {:error, {:stat_failed, path, reason}}
    end
  end

  def resolve_source({:hub, repo_id, filename}, opts) do
    hub_opts = Keyword.take(opts, @hub_keys)

    with {:ok, path} <- Hub.download(repo_id, filename, hub_opts),
         {:ok, %{size: size}} <- File.stat(path) do
      {:ok, path, size}
    end
  end

  @impl true
  def load_model(path, opts) do
    opts = validate!(opts)
    Model.load(path, Keyword.take(opts, @model_load_keys))
  end

  @impl true
  def start_server(id, path, opts) do
    opts = validate!(opts)

    server_opts =
      [model_path: path, name: via(id)] ++ Keyword.take(opts, Server.start_option_keys())

    spec = %{
      id: {:llama_server, id},
      start: {Server, :start_link, [server_opts]},
      restart: :temporary
    }

    DynamicSupervisor.start_child(@dynsup, spec)
  end

  @impl true
  def stop_server(pid) when is_pid(pid) do
    case DynamicSupervisor.terminate_child(@dynsup, pid) do
      :ok -> :ok
      {:error, :not_found} -> :ok
    end
  end

  defp via(id), do: {:via, Registry, {@registry, id}}

  defp validate!(opts), do: Options.validate!(opts, @known_keys, "LlamaCppEx.ModelManager.load/3")
end
