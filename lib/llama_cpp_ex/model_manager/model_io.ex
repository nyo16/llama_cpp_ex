defmodule LlamaCppEx.ModelManager.ModelIO do
  @moduledoc """
  Default `LlamaCppEx.ModelManager.Backend` implementation.

  Delegates to `LlamaCppEx.Hub` for downloads, `LlamaCppEx.Model` for direct
  loading, and `LlamaCppEx.Server` (under the manager's `DynamicSupervisor`,
  named via the manager's `Registry`) for server-backed models.
  """

  @behaviour LlamaCppEx.ModelManager.Backend

  alias LlamaCppEx.{Hub, Model, Server}

  @registry LlamaCppEx.ModelRegistry
  @dynsup LlamaCppEx.ModelDynSup

  # Hub-only options, stripped before handing opts to Model.load / Server.
  @hub_keys [:cache_dir, :token, :revision, :force, :progress]
  # Manager-only options, never forwarded to the native layer.
  @manager_keys [:mode, :capabilities, :default, :memory_budget, :io]

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
    Model.load(path, native_opts(opts))
  end

  @impl true
  def start_server(id, path, opts) do
    server_opts = [model_path: path, name: via(id)] ++ native_opts(opts)

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

  defp native_opts(opts) do
    Keyword.drop(opts, @hub_keys ++ @manager_keys)
  end
end
