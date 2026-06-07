defmodule LlamaCppEx.ModelManager.Entry do
  @moduledoc """
  A single resident-model record held in the `LlamaCppEx.ModelManager` ETS table.

  Entries carry raw NIF resources (`:model`) and backing process pids
  (`:server_pid`). `to_public/1` produces a sanitized, reference-free view for
  `LlamaCppEx.ModelManager.list/0` and `info/1` — handing out the raw `%Model{}`
  would keep the model alive past `unload/1` and defeat GC-based reclamation.
  """

  alias LlamaCppEx.Model

  @type status :: :loading | :ready | :unloading | :error
  @type mode :: :server | :direct
  @type capability :: :generate | :chat | :embed
  @type source :: {:path, String.t()} | {:hub, String.t(), String.t()}

  @type t :: %__MODULE__{
          id: term(),
          status: status(),
          mode: mode(),
          model: Model.t() | nil,
          server_pid: pid() | nil,
          monitor_ref: reference() | nil,
          source: source() | nil,
          capabilities: [capability()],
          byte_size: non_neg_integer(),
          est_bytes: non_neg_integer(),
          n_gpu_layers: integer(),
          error: term(),
          loaded_at: integer() | nil,
          last_used: integer()
        }

  @enforce_keys [:id, :status, :mode]
  defstruct id: nil,
            status: :loading,
            mode: :server,
            model: nil,
            server_pid: nil,
            monitor_ref: nil,
            source: nil,
            capabilities: [:generate, :chat],
            byte_size: 0,
            est_bytes: 0,
            n_gpu_layers: 99,
            error: nil,
            loaded_at: nil,
            last_used: 0

  @doc """
  Returns a sanitized map view of an entry — no `%Model{}` refs or pids.
  """
  @spec to_public(t()) :: map()
  def to_public(%__MODULE__{} = e) do
    %{
      id: e.id,
      status: e.status,
      mode: e.mode,
      capabilities: e.capabilities,
      source: e.source,
      byte_size: e.byte_size,
      est_bytes: e.est_bytes,
      n_gpu_layers: e.n_gpu_layers,
      error: e.error,
      loaded_at: e.loaded_at,
      last_used: e.last_used
    }
  end
end
