defmodule LlamaCppEx.ModelSupervisor do
  @moduledoc """
  Opt-in supervisor for the multi-model manager.

  Starts, in order, the `Registry` and `DynamicSupervisor` that server-backed
  models need, then `LlamaCppEx.ModelManager`. Add it to your application's
  supervision tree:

      children = [
        {LlamaCppEx.ModelSupervisor,
         memory_budget: :auto,
         models: [
           {"chat", {:hub, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"}, n_gpu_layers: -1},
           {"embed", {:path, "/models/nomic-embed.gguf"}, capabilities: [:embed]}
         ]}
      ]

  For quick scripts or IEx, start it directly:

      {:ok, _sup} = LlamaCppEx.ModelSupervisor.start_link([])
      {:ok, "chat"} = LlamaCppEx.ModelManager.load("chat", {:path, "model.gguf"})

  ## Options

    * `:memory_budget` - Forwarded to `LlamaCppEx.ModelManager` (`:infinity`,
      `:auto`, or a byte limit).
    * `:models` - Models to auto-load after start (loaded by the manager so the
      supervisor itself does not block on downloads).
    * `:name` - Supervisor name. Defaults to `LlamaCppEx.ModelSupervisor`.
  """

  use Supervisor

  alias LlamaCppEx.ModelManager.ModelIO

  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts \\ []) do
    {name, opts} = Keyword.pop(opts, :name, __MODULE__)
    Supervisor.start_link(__MODULE__, opts, name: name)
  end

  @impl true
  def init(opts) do
    manager_opts = Keyword.take(opts, [:memory_budget, :models, :io])

    children = [
      {Registry, keys: :unique, name: ModelIO.registry()},
      {DynamicSupervisor, strategy: :one_for_one, name: ModelIO.dynamic_supervisor()},
      {LlamaCppEx.ModelManager, manager_opts}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
