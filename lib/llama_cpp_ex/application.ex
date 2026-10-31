defmodule LlamaCppEx.Application do
  @moduledoc """
  Library-level supervision tree.

  Starts exactly one child: `LlamaCppEx.Registry`, a unique `Registry` that
  `LlamaCppEx.Server` uses to publish its `%LlamaCppEx.Model{}` handle so callers
  can tokenize and apply chat templates without a round-trip through the server's
  mailbox (see `LlamaCppEx.Server.get_model/1`).

  This replaced a `:persistent_term` entry written on every server start and
  erased on every stop. `:persistent_term.put/1` and `erase/1` each trigger a
  **global** garbage collection — every process in the VM scans its heap — and
  `LlamaCppEx.ModelManager`'s load/unload path does exactly that per model swap,
  so a swap under load cost system-wide latency. A `Registry` entry is an ETS
  write with the same O(1) lock-free read, and it is removed automatically when
  the owning process dies, which also closes the leak `:persistent_term` had on
  `Process.exit(server, :kill)` (`terminate/2` never runs, so the entry and its
  reference to the model resource survived the server).

  Nothing here holds model state, so a restart is cheap: servers re-register on
  their own `handle_continue`.

  This is separate from `LlamaCppEx.ModelSupervisor`, which is opt-in and starts
  the `Registry`/`DynamicSupervisor` pair that the multi-model manager needs.

  When `config :llama_cpp_ex, log_level: level` is set, it is applied here with
  `LlamaCppEx.set_log_level/1`. Unset, the NIF's own default (`:info`) stands
  and starting the application does not touch the NIF.
  """

  use Application

  @impl true
  def start(_type, _args) do
    case Application.fetch_env(:llama_cpp_ex, :log_level) do
      {:ok, level} -> :ok = LlamaCppEx.set_log_level(level)
      :error -> :ok
    end

    children = [
      {Registry, keys: :unique, name: LlamaCppEx.Registry}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: __MODULE__.Supervisor)
  end
end
